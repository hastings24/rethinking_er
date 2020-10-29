# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import importlib
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.joint_training import train as jtrain
from utils.best_args import best_args
from utils.conf import set_random_seed
import torch


def main():
    if torch.cuda.device_count() > 1:
        torch.set_num_threads(6 * torch.cuda.device_count())
    else:
        torch.set_num_threads(2)
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        model = args.model
        if model == 'joint':
            model = 'sgd'
        best = best_args[args.dataset][model]
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    off_joint = False
    if args.model == 'joint' and args.dataset == 'seq-core50':
        args.dataset = 'seq-core50j'
        args.model = 'sgd'
        off_joint = True

    dataset = get_dataset(args)

    # continual learning
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    if off_joint:
        print('BEGIN JOINT TRAINING')
        jtrain(model, dataset, args)
    else:
        print('BEGIN CONTINUAL TRAINING')
        train(model, dataset, args)


if __name__ == '__main__':
    main()
