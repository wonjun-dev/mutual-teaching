import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np

from utils.data import Market1501
from utils.clustering import KMeansCluster
from model import ReidResNet


class MutualTeaching:
    def __init__(self, model_1, model_2, model_cluster, optimizer, device):
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
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.triplet_loss_fn = nn.BCELoss()
        self.lambda_id = 0.5

    def training_loop(self, dataloader, epoch):
        # Generate hard pseudo label
        self._generate_pseudo_labels(dataloader, self.device)
        # iteration
        dataloader.dataset.mutual = True
        for idx, (x_1, x_2, y_tilde) in enumerate(dataloader):
            x_1, x_2, y_tilde = x_1.to(self.device), x_2.to(self.device), y_tilde.to(self.device)
            out_1, out_2, mean_out_1, mean_out_2 = self._forward(x_1, x_2)
            hard_loss_1, hard_loss_2 = self._calculate_hard_loss(out_1, out_2, y_tilde)
            soft_loss_1, soft_loss_2 = self._calculate_soft_loss(
                out_1, out_2, mean_out_1, mean_out_2
            )
            hard_tri_loss_1 = self._calculate_hard_triplet_loss(out_1, y_tilde)
            hard_tri_loss_2 = self._calculate_hard_triplet_loss(out_2, y_tilde)

            id_loss = (1 - self.lambda_id) * (hard_loss_1 + hard_loss_2) + self.lambda_id * (
                soft_loss_1 + soft_loss_2
            )

            tri_loss = hard_tri_loss_1 + hard_tri_loss_2
            print(id_loss.item())
            print(tri_loss.item())
            loss = id_loss + tri_loss
            self._step(loss)

            # self._mean_parameters()
            self._mean_parameters(
                self.model_1, self.mean_model_1, step=epoch * len(dataloader) + idx
            )
            self._mean_parameters(
                self.model_2, self.mean_model_2, step=epoch * len(dataloader) + idx
            )

    def _forward(self, x_1, x_2):
        out_1 = self.model_1(x_1)
        out_2 = self.model_2(x_2)

        mean_out_1 = self.mean_model_1(x_1)
        mean_out_2 = self.mean_model_2(x_2)

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

    def _calculate_hard_triplet_loss(self, out, y_tilde):

        triplet_loss = torch.tensor(0, dtype=torch.float)
        hard_pos_dist = torch.tensor(0, dtype=torch.float)
        hard_neg_dist = torch.tensor(0, dtype=torch.float)

        # get hardest positive/negative samples in mini-batch.
        for sample, y in zip(out, y_tilde):
            # find idxs positive/negative
            pos_idxs = (y == y_tilde).nonzero(as_tuple=True)[0]
            neg_idxs = (y != y_tilde).nonzero(as_tuple=True)[0]

            for idx in pos_idxs:
                pos = out[idx]
                print("s", sample)
                print("p", pos)
                dist = torch.norm(sample - pos, p=2)  # l2-norm
                print("d", dist)
                if dist > hard_pos_dist:
                    hard_pos_dist = dist

            for idx in neg_idxs:
                neg = out[idx]
                dist = torch.norm(sample - neg, p=2)  # l2-norm
                if dist > hard_neg_dist:
                    hard_neg_dist = dist

            # calculate
            pred = hard_neg_dist.exp() / (hard_pos_dist.exp() + hard_neg_dist.exp())
            print(pred)
            print(torch.tensor(1, dtype=torch.float, device=self.device))
            triplet_loss += self.triplet_loss_fn(
                pred, torch.tensor(1, dtype=torch.float, device=self.device)
            )

        return triplet_loss

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
        # TODO Use average feature of two mean models
        print("Encoding features for clustering.")
        dataloader.dataset.mutual = False
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
        "datasets/market1501/Market-1501-v15.09.15", data_name="bounding_box_train", mutual=True
    )
    train_loader = DataLoader(train_dataset, batch_size=32)

    mt = MutualTeaching(model_1, model_2, model_cluster, optimizer, device)

    for e in range(10):
        mt.training_loop(train_loader, epoch=e)


if __name__ == "__main__":
    main()
