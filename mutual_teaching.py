import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from utils.data import Market1501
from utils.clustering import KMeansCluster
from model import ReidResNet


class MutualTeaching:
    def __init__(self, model_1, model_2, model_cluster, optimizer, augment_fn, device):
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_cluster = model_cluster
        self.mean_model_1 = copy.deepcopy(self.model_1)
        self.mean_model_2 = copy.deepcopy(self.model_2)

        for param in self.mean_model_1.parameters():
            param.detach_()
        for param in self.mean_model_2.parameters():
            param.detach_()

        self.optimizer = optimizer
        self.augment_fn = augment_fn
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_id = 0.5

    def training_loop(self, dataloader, epoch):
        # Generate hard pseudo label
        self._generate_pseudo_labels(dataloader, self.device)
        # iteration
        for idx, (x, y_tilde) in enumerate(dataloader):
            x, y_tilde = x.to(self.device), y_tilde.to(self.device)
            x_prime = self.augment_fn(x)
            out_1, out_2, mean_out_1, mean_out_2 = self._forward(x, x_prime)
            hard_loss_1, hard_loss_2 = self._calculate_hard_loss(out_1, out_2, y_tilde)
            soft_loss_1, soft_loss_2 = self._calculate_soft_loss(
                out_1, out_2, mean_out_1, mean_out_2
            )
            loss = (1 - self.lambda_id) * (hard_loss_1 + hard_loss_2) + self.lambda_id * (
                soft_loss_1 + soft_loss_2
            )
            print(loss.item())
            self._step(loss)

            # self._mean_parameters()
            self._mean_parameters(
                self.model_1, self.mean_model_1, step=epoch * len(dataloader) + idx
            )
            self._mean_parameters(
                self.model_2, self.mean_model_2, step=epoch * len(dataloader) + idx
            )

    def _forward(self, x, x_prime):
        out_1 = self.model_1(x)
        out_2 = self.model_2(x_prime)

        mean_out_1 = self.mean_model_1(x)
        mean_out_2 = self.mean_model_2(x_prime)

        return out_1, out_2, mean_out_1, mean_out_2

    def _calculate_hard_loss(self, out_1, out_2, y_tilde):
        loss_1 = self.loss_fn(out_1, y_tilde)
        loss_2 = self.loss_fn(out_2, y_tilde)
        return loss_1, loss_2

    def _calculate_soft_loss(self, out_1, out_2, mean_out_1, mean_out_2):
        loss_1 = torch.mean(
            torch.sum(-F.softmax(mean_out_2, dim=1).detach() * F.log_softmax(out_1, dim=1), dim=1)
        )
        loss_2 = torch.mean(
            torch.sum(-F.softmax(mean_out_1, dim=1).detach() * F.log_softmax(out_2, dim=1), dim=1)
        )
        return loss_1, loss_2

    def _step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _mean_parameters(self, model, mean_model, step: int, alpha: float = 0.999):
        alpha = min(1 - 1 / (step + 1), alpha)
        print("alpha: ", alpha)
        for mean_param, param in zip(mean_model.parameters(), model.parameters()):
            mean_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _generate_pseudo_labels(self, dataloader, device):
        # pseudo_label = torch.arange(0, size) % 3
        print("Encoding features for clustering.")
        full_features = []
        for samples, _ in dataloader:
            samples = samples.to(device)
            self.mean_model_1(samples)
            batch_features = self.mean_model_1.hooks.numpy()
            for f in batch_features:
                full_features.append(f)
        full_features = np.array(full_features, dtype=object)

        pseudo_label = self.model_cluster.generate_pseudo_labels(full_features)
        dataloader.dataset.pseudo_labels = torch.tensor(pseudo_label, dtype=torch.long)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_1 = ReidResNet()
    model_2 = ReidResNet()
    model_1.to(device)
    model_2.to(device)
    model_cluster = KMeansCluster(n_clusters=500)
    optimizer = torch.optim.Adam(
        [{"params": model_1.parameters()}, {"params": model_2.parameters()}]
    )
    augment_fn = lambda x: x + 1  # tmp function

    train_dataset = Market1501(
        "datasets/market1501/Market-1501-v15.09.15", data_name="bounding_box_train"
    )
    train_loader = DataLoader(train_dataset, batch_size=128)

    mt = MutualTeaching(model_1, model_2, model_cluster, optimizer, augment_fn, device)

    for e in range(10):
        mt.training_loop(train_loader, epoch=e)


if __name__ == "__main__":
    main()
