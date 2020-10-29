# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torchvision.datasets import FashionMNIST as FMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone.MNISTMLP import MNISTMLP
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple


class MyFMNIST(FMNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MyFMNIST, self).__init__(root, train,
                                      transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, original_img


class SequentialFMNIST(ContinualDataset):

    NAME = 'seq-fmnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    def get_data_loaders(self, nomask=False):
        transform = transforms.ToTensor()
        train_dataset = MyFMNIST(base_path() + 'FMNIST',
                                train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            test_dataset = FMNIST(base_path() + 'FMNIST',
                                train=False, download=True, transform=transform)

        if not nomask:
            train, test = store_masked_loaders(train_dataset, test_dataset, self)
            return train, test
        else:
            return train_dataset, test_dataset

    def not_aug_dataloader(self, args, batch_size):
        if hasattr(args, 'iba') and args.iba:
            raise ValueError('IBA is not compatible with F-MNIST')
        transform = transforms.ToTensor()
        train_dataset = MyFMNIST(base_path() + 'FMNIST',
                                train=True, download=True, transform=transform)

        train_mask = np.logical_and(np.array(train_dataset.targets) >= (self.i - 1)
                                    * self.N_CLASSES_PER_TASK, np.array(train_dataset.targets) < self.i * self.N_CLASSES_PER_TASK)

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, SequentialFMNIST.N_TASKS
                        * SequentialFMNIST.N_CLASSES_PER_TASK, hidden_size=256)


    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None
