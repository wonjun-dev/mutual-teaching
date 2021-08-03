import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import numpy as np


class MutualTeaching:
    def __init__(self, model_1, model_2, optimizer, augment_func):
        self.model_1 = model_1
        self.model_2 = model_2
        self.mean_model_1 = model_1.parameters()
        self.mean_model_2 = model_2.parameters()
        self.optimizer = optimizer
        self.augment_fn = augment_func
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_id = 0.5

    def training_loop(self, data):
        # Generate hard pseudo label
        # TODO use clustering algorithm
        pseudo_label = torch.arange(0, 100) % 3
        # iteration
        for idx, x in enumerate(data):
            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(pseudo_label[idx], 0)
            x_prime = self.augment_fn(x)
            out_1, out_2 = self._forward(x, x_prime)
            hard_loss_1, hard_loss_2 = self._calculate_hard_loss(out_1, out_2, y)
            soft_loss_1, soft_loss_2 = self._calculate_soft_loss(out_1, out_2)
            loss = (1 - self.lambda_id) * (hard_loss_1 + hard_loss_2) + self.lambda_id * (
                soft_loss_1 + soft_loss_2
            )
            self._step(loss)
            # self._mean_parameters()

    def _forward(self, x, x_prime):
        out_1 = self.model_1(x)
        out_2 = self.model_2(x_prime)

        return out_1, out_2

    def _calculate_hard_loss(self, out_1, out_2, pseudo_label):
        loss_1 = self.loss_fn(out_1, pseudo_label)
        loss_2 = self.loss_fn(out_2, pseudo_label)
        return loss_1, loss_2

    def _calculate_soft_loss(self, out_1, out_2):
        loss_1 = torch.mean(torch.sum(F.softmax(out_2, dim=1) * F.log_softmax(out_1, dim=1), dim=1))
        loss_2 = torch.mean(torch.sum(F.softmax(out_1, dim=1) * F.log_softmax(out_2, dim=1), dim=1))
        return loss_1, loss_2

    def _step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _mean_parameters(self):
        pass


def main():
    model_1 = nn.Linear(10, 3)
    model_2 = nn.Linear(10, 3)
    optimizer = torch.optim.Adam(
        [{"params": model_1.parameters()}, {"params": model_2.parameters()}]
    )
    augment_fn = lambda x: x + 1  # tmp function
    dummy_data = torch.randn(100, 10)

    mt = MutualTeaching(model_1, model_2, optimizer, augment_fn)

    for e in range(10):
        mt.training_loop(dummy_data)


if __name__ == "__main__":
    main()
