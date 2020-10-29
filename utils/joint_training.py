# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils.status import progress_bar, load_backup, create_stash, save_backup
from utils.tb_logger import *
from utils.loggers import CsvLogger
from utils.status import LastWish
import torch.nn as nn
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from torch.utils.data import DataLoader
import numpy as np
import sys
from copy import deepcopy


def evaluate(model: ContinualModel, test_loader: DataLoader, net=None):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param test_loader: the test dataloader
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    if net is None:
        net = model.net
    mode = net.training
    net.eval()
    correct, total = 0, 0
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = net(inputs)

        _, pred = torch.max(outputs.data, 1)
        correct += torch.sum(pred == labels).item()
        total += labels.shape[0]

    net.train(mode)
    return (correct / total * 100, correct / total * 100)


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
    train_loader, test_loader = dataset.get_joint_loaders()

    for epoch in range(args.n_epochs):
        if epoch < model_stash['epoch_idx']:
            print('skipping epoch', epoch, file=sys.stderr)
            continue
        for i, data in enumerate(train_loader):
            if epoch < model_stash['epoch_idx'] and i < model_stash['batch_idx']:
                print('batch', epoch, file=sys.stderr)
                continue

            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            loss = model.observe(inputs, labels, not_aug_inputs)
            progress_bar(i, len(train_loader), epoch, 0, loss)

            if args.tensorboard:
                tb_logger.log_loss(loss, args, epoch, 0, i)

            model_stash['batch_idx'] = i + 1
        model_stash['epoch_idx'] = epoch + 1
        model_stash['batch_idx'] = 0

        if epoch and not epoch % 50 or epoch == args.n_epochs - 1:
            accs = evaluate(model, test_loader)
            print('\nAccuracy after {} epochs: {}'.format(epoch + 1, accs[0]))
            model_stash['mean_accs'].append(accs[0])
            if args.csv_log:
                csv_logger.log(np.array(accs))
            if args.tensorboard:
                tb_logger.log_accuracy(np.array(accs), np.array(accs), args, 0)

    if hasattr(model, 'end_task'):
        model.end_task(dataset)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
