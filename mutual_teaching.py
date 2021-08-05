import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader

from utils.data import Market1501


class MutualTeaching:
    def __init__(self, model_1, model_2, optimizer, augment_fn):
        self.model_1 = model_1
        self.model_2 = model_2
        self.mean_model_1 = model_1.parameters()
        self.mean_model_2 = model_2.parameters()
        self.optimizer = optimizer
        self.augment_fn = augment_fn
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_id = 0.5

    def training_loop(self, dataloader):
        # Generate hard pseudo label
        # TODO use clustering algorithm
        pseudo_label = self._generate_pseudo_labels(dataloader.dataset.__len__())
        # iteration
        for idx, x in enumerate(dataloader):
            y = pseudo_label[idx * x.shape[0] : (idx + 1) * x.shape[0]]
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

    def _generate_pseudo_labels(self, size):
        pseudo_label = torch.arange(0, size) % 3
        return pseudo_label


def main():
    model_1 = nn.Sequential(nn.Conv2d(3, 10, 3), nn.ReLU(), nn.Flatten(), nn.Linear(78120, 3))
    model_2 = nn.Sequential(nn.Conv2d(3, 10, 3), nn.ReLU(), nn.Flatten(), nn.Linear(78120, 3))
    optimizer = torch.optim.Adam(
        [{"params": model_1.parameters()}, {"params": model_2.parameters()}]
    )
    augment_fn = lambda x: x + 1  # tmp function

    train_dataset = Market1501(
        "datasets/market1501/Market-1501-v15.09.15", data_name="bounding_box_train"
    )
    train_loader = DataLoader(train_dataset, batch_size=128)

    mt = MutualTeaching(model_1, model_2, optimizer, augment_fn)

    for e in range(10):
        mt.training_loop(train_loader)


if __name__ == "__main__":
    main()
