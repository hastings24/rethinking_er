# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils.ring_buffer import RingBuffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
import numpy as np
from utils.status import progress_bar
from torch.optim import SGD
import sys

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--hal_lambda', type=float, default=0.1)

    return parser


class HAL(ContinualModel):
    NAME = 'hal'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(HAL, self).__init__(backbone, loss, args, transform)
        self.task_number = 0
        self.buffer = RingBuffer(self.args.buffer_size, self.device, get_dataset(args).N_TASKS)
        # self.anchors = torch.zeros([0, 1, 28, 28]).to(self.device)
        self.hal_lambda = args.hal_lambda
        self.beta = .5
        self.gamma = 1 #.1
        self.anchor_optimization_steps = 100
        self.finetuning_epochs = 50

    def end_task(self, dataset):
        self.task_number += 1
        # ring buffer mgmt
        self.buffer.num_seen_examples = 0
        self.buffer.task_number += 1

        # get anchors
        e_t = self.get_anchors(dataset)
        del self.phi

    def get_anchors(self, dataset):
        theta_t = self.net.get_params().detach().clone()

        for _ in range(self.finetuning_epochs):
            inputs, labels = self.buffer.get_data(self.args.batch_size, transform=self.transform)
            self.opt.zero_grad()
            out = self.net(inputs)
            loss = self.loss(out, labels)
            loss.backward()
            self.opt.step()

        theta_m = self.net.get_params().detach().clone()

        classes_for_this_task = np.unique(dataset.train_loader.dataset.targets)

        for a_class in classes_for_this_task:
            e_t = torch.rand(self.input_shape, requires_grad=True, device=self.device)
            e_t_opt = SGD([e_t], lr=self.args.lr)
            print(file=sys.stderr)
            for i in range(self.anchor_optimization_steps):
                e_t_opt.zero_grad()
                cum_loss = 0

                self.opt.zero_grad()
                self.net.set_params(theta_m)
                loss = -torch.sum(self.loss(self.net(e_t.unsqueeze(0)), torch.tensor([a_class]).to(self.device)))
                loss.backward()
                cum_loss += loss.item()

                self.opt.zero_grad()
                self.net.set_params(theta_t)
                loss = torch.sum(self.loss(self.net(e_t.unsqueeze(0)), torch.tensor([a_class]).to(self.device)))
                loss.backward()
                cum_loss += loss.item()

                self.opt.zero_grad()
                loss = torch.sum(self.gamma * (self.net.features(e_t.unsqueeze(0)) - self.phi) ** 2)
                assert not self.phi.requires_grad
                loss.backward()
                cum_loss += loss.item()

                if i % 10 == 9:
                    progress_bar(i, self.anchor_optimization_steps, i, 'A' + str(a_class), cum_loss)

                e_t_opt.step()

            e_t = e_t.detach()
            e_t.requires_grad = False
            self.anchors = torch.cat((self.anchors, e_t.unsqueeze(0)))

        self.net.set_params(theta_t)

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        if not hasattr(self, 'input_shape'):
            self.input_shape = inputs.shape[1:]
            self.anchors = torch.zeros(tuple([0] + list(self.input_shape))).to(self.device)
        if not hasattr(self, 'phi'):
            self.phi = torch.zeros_like(self.net.features(inputs[0].unsqueeze(0)), requires_grad=False)
            assert not self.phi.requires_grad

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        old_weights = self.net.get_params().detach().clone()

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        first_loss = 0

        if len(self.anchors) > 0:
            first_loss = loss.item()
            with torch.no_grad():
                pred_anchors = self.net(self.anchors)

            self.net.set_params(old_weights)
            pred_anchors -= self.net(self.anchors)
            loss = self.hal_lambda * (pred_anchors ** 2).mean()
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            self.phi = self.beta * self.phi + (1 - self.beta) * self.net.features(inputs[:real_batch_size]).mean(0)

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return first_loss + loss.item()
