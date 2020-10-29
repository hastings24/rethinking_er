# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from backbone.ResNet import ResNet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import socket
import os
import time
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
import sys

#
#   MODIFIED VERSION OF seq-core50 for joint training only
#   (your RAM will live to fight another day if you use this)
#

class Core50(Dataset):
    TRAIN_LENGTH = 119894
    TRAIN_MAP = {0: [0], 1: [0], 2: [0], 3: [0], 4: [0, 1], 5: [1], 6: [1], 7: [1, 2], 8: [2], 9: [2, 3], 10: [3],
                 11: [3, 4], 12: [4], 13: [4], 14: [4, 5], 15: [5], 16: [5, 6], 17: [6], 18: [6], 19: [6, 7], 20: [7],
                 21: [7, 8], 22: [8], 23: [8]}
    TEST_LENGTH  =  44972


    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1rm2gE74AeEXaHRP8tHvtYnmokZ5Td-Yb
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1rm2gE74AeEXaHRP8tHvtYnmokZ5Td-Yb',

                    dest_path=os.path.join(root, 'core-50-processed.zip'),
                    unzip=True)

        self.targets = []
        for num in range(24 if self.train else 9):
            self.targets.append(np.load(os.path.join(
                self.root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'test', num))))
        self.targets = np.concatenate(np.array(self.targets))
        print("TARGETS SHAPE:", self.targets.shape, file=sys.stderr)

        # placeholder
        self.data = np.arange(self.TRAIN_LENGTH if self.train else self.TEST_LENGTH)
        self.loaded_data = {}
        self.in_memory = []

    def __len__(self):
        return len(self.targets)

    def fetch_and_load(self, index):
        memory_bank_index = index // 5000
        memory_bank_offset = index % 5000
        if memory_bank_index not in self.in_memory:

            if self.train:
                # clean memory
                if len(self.in_memory) == 999999:
                    del self.loaded_data[self.in_memory[0]]
                    self.in_memory = self.in_memory[1:]

                self.loaded_data[memory_bank_index] = np.load(
                    os.path.join(self.root, 'processed/x_train_%02d.npy' % memory_bank_index))
                self.in_memory.append(memory_bank_index)

            else:
                # keep <=999999 banks in memory at all times
                if len(self.in_memory) == 999999:
                    del self.loaded_data[self.in_memory[0]]
                    self.in_memory = self.in_memory[1:]

                self.loaded_data[memory_bank_index] = np.load(os.path.join(self.root, 'processed/x_test_%02d.npy' % memory_bank_index))
                self.in_memory.append(memory_bank_index)

        else:
            pass
        return self.loaded_data[memory_bank_index][memory_bank_offset]

    def __getitem__(self, index):
        img, target = self.fetch_and_load(self.data[index]), self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MyCore50(Core50):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        super(MyCore50, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.fetch_and_load(self.data[index]), self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target,  not_aug_img


class SequentialCore50j(ContinualDataset):

    NAME = 'seq-core50j'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = [10, 5, 5, 5, 5, 5, 5, 5, 5]
    N_TASKS = 9
    _mean = (0.59998563, 0.56810559, 0.54106508)
    _std = (0.07111129, 0.0552458, 0.06024752)
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(128, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(_mean,
                                  _std)])


    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCore50(base_path() + 'CORE50',
                                 train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = Core50(base_path() + 'CORE50',
                        train=False, download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_joint_loaders(self, nomask=False):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCore50(base_path() + 'CORE50',
                                 train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = Core50(base_path() + 'CORE50',
                                  train=False, download=True, transform=test_transform)

        train_loader = DataLoader(train_dataset,
                                  batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset,
                                 batch_size=32, shuffle=False, num_workers=2)
        return train_loader, test_loader

    def not_aug_dataloader(self, args, batch_size):
        if hasattr(args, 'iba') and args.iba:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            self.get_normalization_transform()])

        train_dataset = MyCore50(base_path() + 'CORE50',
                            train=True, download=True, transform=transform)

        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader


    @staticmethod
    def get_backbone():
        return ResNet18(np.sum(SequentialCore50j.N_CLASSES_PER_TASK))

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCore50j._mean,
                                         SequentialCore50j._std)
        return transform
