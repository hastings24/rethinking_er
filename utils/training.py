# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils.status import progress_bar, load_backup, create_stash, save_backup
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from utils.status import LastWish
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from torch import nn
import time
import sys


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    if type(dataset.N_CLASSES_PER_TASK) == list:
        THIS_TASK_START = int(np.sum(dataset.N_CLASSES_PER_TASK[:k]))
        THIS_TASK_END = int(np.sum(dataset.N_CLASSES_PER_TASK[:k+1]))
    else:
        THIS_TASK_START = k * dataset.N_CLASSES_PER_TASK
        THIS_TASK_END = (k + 1) * dataset.N_CLASSES_PER_TASK

    outputs[:, :THIS_TASK_START] = -float('inf')
    outputs[:, THIS_TASK_END:] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    last_wish = LastWish()
    model.net.to(model.device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        get_par_fn = model.net.get_params
        feat_fn = model.net.features
        class_fn = model.net.classifier
        set_par_fn = model.net.set_params
        model.net = nn.DataParallel(model.net)
        setattr(model.net, 'get_params', get_par_fn)
        setattr(model.net, 'features', feat_fn)
        setattr(model.net, 'classifier', class_fn)
        setattr(model.net, 'set_params', set_par_fn)

    if args.checkpoint_path is not None:
        model_stash = load_backup(model, args)
    else:
        model_stash = create_stash(model, args, dataset)

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
    
    last_wish.register_action(lambda: save_backup(model, model_stash))

    print(file=sys.stderr)

    tasks = dataset.N_TASKS

    for t in range(tasks):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t < model_stash['task_idx']:
            print('skipping task', t, file=sys.stderr)
            if t > 0 and args.csv_log:
                csv_logger.log(model_stash['mean_accs'][t - 1])
            if hasattr(model, 'end_task'):
                model.end_task(dataset)
            continue

        n_epochs = args.n_epochs
        if args.model == 'joint':
            n_epochs = 0

        for epoch in range(n_epochs):
            if t <= model_stash['task_idx'] and epoch < model_stash['epoch_idx']:
                print('skipping epoch', epoch, file=sys.stderr)
                continue

            for i, allData in enumerate(train_loader):
                data = allData

                if t <= model_stash['task_idx'] and epoch < model_stash[
                        'epoch_idx'] and i < model_stash['batch_idx']:
                    print('batch', epoch, file=sys.stderr)
                    continue

                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, t, loss)
                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
            model.net.train()
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        mean_acc = np.mean(accs, axis=1)
        if n_epochs or t == dataset.N_TASKS - 1:
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(accs, mean_acc, args, t)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
