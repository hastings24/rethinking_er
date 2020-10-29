# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_fmnist import SequentialFMNIST
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.utils.continual_dataset import ContinualDataset
from datasets.seq_core50 import SequentialCore50
from datasets.seq_core50j import SequentialCore50j
from argparse import Namespace

NAMES = {
    SequentialFMNIST.NAME: SequentialFMNIST,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialCore50.NAME: SequentialCore50,
    SequentialCore50j.NAME: SequentialCore50j
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)

