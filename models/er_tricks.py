# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils.buffer_tricks import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from torch.optim import SGD
from datasets import get_dataset
from utils import apply_decay


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='A bag of tricks for '
                                        'Continual learning.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--bic_epochs', type=int, default=50,
                        help='bias injector.')
    parser.add_argument('--elrd', type=float, default=1)

    return parser


class ErTricks(ContinualModel):
    NAME = 'er_tricks'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErTricks, self).__init__(backbone, loss, args, transform)
        dd = get_dataset(args)
        self.n_tasks = dd.N_TASKS
        self.cpt = dd.N_CLASSES_PER_TASK

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0

        # BIC
        self.bic_params = torch.zeros(2, device=self.device, requires_grad=True)
        self.bic_opt = SGD([self.bic_params], lr=0.5)

    def end_task(self, dataset):
        self.current_task += 1
        self.net.eval()
        for l in range(self.args.bic_epochs):
            data = self.buffer.get_data(self.args.buffer_size, transform=dataset.get_normalization_transform())
            while data[0].shape[0] > 0:
                inputs, labels = data[0][:self.args.batch_size], data[1][:self.args.batch_size]
                data = (data[0][self.args.batch_size:], data[1][self.args.batch_size:])

                self.bic_opt.zero_grad()
                with torch.no_grad():
                    out = self.net(inputs)

                if type(self.cpt) == list:
                    start_last_task = int(torch.sum(torch.tensor(self.cpt[:self.current_task - 1])))
                    end_last_task = int(torch.sum(torch.tensor(self.cpt[:self.current_task])))
                else:
                    start_last_task = (self.current_task - 1) * self.cpt
                    end_last_task = self.current_task * self.cpt
                out[:, start_last_task:end_last_task] *= self.bic_params[1].repeat_interleave(end_last_task - start_last_task)
                out[:, start_last_task:end_last_task] += self.bic_params[0].repeat_interleave(end_last_task - start_last_task)

                loss_bic = self.loss(out, labels)
                loss_bic.backward()
                self.bic_opt.step()

        self.net.train()

    def forward(self, x):
        ret = super(ErTricks, self).forward(x)
        if ret.shape[0] > 0:
            if type(self.cpt) == list:
                start_last_task = int(torch.sum(torch.tensor(self.cpt[:self.current_task - 1])))
                end_last_task = int(torch.sum(torch.tensor(self.cpt[:self.current_task])))
            else:
                start_last_task = (self.current_task - 1) * self.cpt
                end_last_task = self.current_task * self.cpt
            ret[:, start_last_task:end_last_task] *= self.bic_params[1].repeat_interleave(end_last_task - start_last_task)
            ret[:, start_last_task:end_last_task] += self.bic_params[0].repeat_interleave(end_last_task - start_last_task)

        return ret

    def observe(self, inputs, labels, not_aug_inputs):

        apply_decay(self.args.elrd, self.args.lr, self.opt, self.buffer.num_seen_examples)

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_indexes = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform, return_indexes=True)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss_scores = self.loss(outputs, labels, reduction='none')
        loss = loss_scores.mean()
        loss.backward()
        self.opt.step()

        if not self.buffer.is_empty():
            self.buffer.update_scores(buf_indexes, -loss_scores.detach()[real_batch_size:])
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size],
                             loss_scores=-loss_scores.detach()[:real_batch_size])

        return loss.item()
