import os
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


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
        self.lambda_tri = 0.5
        self.alpha = 0
        self.pseudo_labels = OrderedDict()
        self.best_score = 100000

    def training_loop(self, train_loader, cluster_loader, epoch):
        # Generate hard pseudo label
        self._generate_pseudo_labels(cluster_loader, self.device)
        # iteration
        train_loader.dataset.mutual = True

        self.model_1.train()
        self.model_2.train()
        for idx, (x_1, x_2, gt, key) in enumerate(train_loader):
            # y_tilde = torch.randint(0, 10, (32,))
            y_tilde = torch.tensor([self.pseudo_labels[k] for k in key], dtype=torch.long)
            x_1, x_2, y_tilde = x_1.to(self.device), x_2.to(self.device), y_tilde.to(self.device)
            out_1, out_2, mean_out_1, mean_out_2 = self._forward(x_1, x_2)
            hard_loss_1, hard_loss_2 = self._calculate_hard_loss(out_1, out_2, y_tilde)
            soft_loss_1, soft_loss_2 = self._calculate_soft_loss(
                out_1, out_2, mean_out_1, mean_out_2
            )

            # Triplet loss
            preds_1 = self._calculate_triplet_distance(self.model_1.hooks, y_tilde)
            preds_2 = self._calculate_triplet_distance(self.model_2.hooks, y_tilde)
            mean_preds_1 = self._calculate_triplet_distance(self.mean_model_1.hooks, y_tilde)
            mean_preds_2 = self._calculate_triplet_distance(self.mean_model_2.hooks, y_tilde)

            hard_tri_loss_1 = self._calculate_hard_triplet_loss(preds_1)
            hard_tri_loss_2 = self._calculate_hard_triplet_loss(preds_2)
            soft_tri_loss_1 = self._calculate_soft_triplet_loss(preds_1, mean_preds_2)
            soft_tri_loss_2 = self._calculate_soft_triplet_loss(preds_2, mean_preds_1)

            id_loss = (1 - self.lambda_id) * (hard_loss_1 + hard_loss_2) + self.lambda_id * (
                soft_loss_1 + soft_loss_2
            )

            tri_loss = (1 - self.lambda_tri) * (
                hard_tri_loss_1 + hard_tri_loss_2
            ) + self.lambda_tri * (soft_tri_loss_1 + soft_tri_loss_2)

            loss = id_loss + tri_loss
            print(
                f"loss: {loss.item()}, id_loss: {id_loss.item()}, tri_loss: {tri_loss.item()}, alpha: {self.alpha}"
            )
            self._step(loss)

            # self._mean_parameters()
            self._mean_parameters(
                self.model_1, self.mean_model_1, step=epoch * len(train_loader) + idx
            )
            self._mean_parameters(
                self.model_2, self.mean_model_2, step=epoch * len(train_loader) + idx
            )

            if loss.item() < self.best_score:
                print(f"Save model...")
                self.save_model(self.model_1, name=f"1_{epoch}")
                self.save_model(self.model_1, name=f"2_{epoch}")
                self.best_score = loss.item()

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

    def _calculate_hard_triplet_loss(self, preds):
        cnt = len(preds)
        triplet_loss = torch.tensor(0, dtype=torch.float)

        if cnt > 0:
            preds = torch.tensor(preds)
            preds = torch.nan_to_num(preds, nan=1.0)  # nan has 0 loss
            targets = torch.ones(cnt)
            triplet_loss = self.triplet_loss_fn(preds, targets)

        return triplet_loss.to(self.device)

    def _calculate_soft_triplet_loss(self, preds, targets):
        cnt = len(preds)
        triplet_loss = torch.tensor(0, dtype=torch.float)

        if cnt > 0:
            preds = torch.tensor(preds)
            targets = torch.tensor(targets)

            preds_nan_idx = self.__get_nan_idx(preds)
            targets_nan_idx = self.__get_nan_idx(targets)

            # nan has 0 loss
            preds[preds_nan_idx] = 1.0
            preds[targets_nan_idx] = 1.0
            targets[preds_nan_idx] = 1.0
            targets[targets_nan_idx] = 1.0

            triplet_loss = self.triplet_loss_fn(preds, targets)

        return triplet_loss.to(self.device)

    def _calculate_triplet_distance(self, hook, y_tilde):
        preds = list()

        # get hardest positive/negative samples in mini-batch.
        for sample, y in zip(hook, y_tilde):
            hard_pos_dist = torch.tensor(0, dtype=torch.float)
            hard_neg_dist = torch.tensor(float("inf"), dtype=torch.float)

            # find idxs positive/negative
            pos_idxs = (y == y_tilde).nonzero(as_tuple=True)[0]
            neg_idxs = (y != y_tilde).nonzero(as_tuple=True)[0]

            if (
                len(pos_idxs) > 1 and len(neg_idxs) > 1
            ):  # only there are positivie samples in the mini-batch

                for idx in pos_idxs:
                    pos = hook[idx]
                    dist = torch.norm(sample - pos, p=2)  # l2-norm

                    if dist >= hard_pos_dist:
                        hard_pos_dist = dist
                # print("hpd", hard_pos_dist)

                for idx in neg_idxs:
                    neg = hook[idx]
                    dist = torch.norm(sample - neg, p=2)  # l2-norm
                    if dist <= hard_neg_dist:
                        hard_neg_dist = dist

                # calculate
                pred = hard_neg_dist.exp() / (hard_pos_dist.exp() + hard_neg_dist.exp())
                preds.append(pred)

        return preds

    def _step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _mean_parameters(self, model, mean_model, step: int, alpha: float = 0.999):
        self.alpha = min(1 - 1 / (step + 1), alpha)
        for mean_param, param in zip(mean_model.parameters(), model.parameters()):
            mean_param.data.mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)

    def _generate_pseudo_labels(self, dataloader, device):
        print("Encoding features for clustering.")
        full_features = []
        self.mean_model_1.eval()
        self.mean_model_2.eval()
        for imgs, _, key in dataloader:
            imgs = imgs.to(device)
            self.mean_model_1(imgs)
            self.mean_model_2(imgs)
            batch_features_1 = self.mean_model_1.hooks.numpy()
            batch_features_2 = self.mean_model_2.hooks.numpy()
            for f in zip(batch_features_1, batch_features_2):
                full_features.append((f[0] + f[1]) / 2)

            for k in key:
                self.pseudo_labels[k] = None

        full_features = np.array(full_features, dtype=object)
        pseudo_label = self.model_cluster.generate_pseudo_labels(full_features)
        for k, l in zip(self.pseudo_labels.keys(), pseudo_label):
            # self.pseudo_labels[k] = torch.tensor(l, dtype=torch.long)
            self.pseudo_labels[k] = l

    def __get_nan_idx(self, input):
        is_nan = torch.isnan(input)
        return is_nan.nonzero(as_tuple=True)[0]

    def save_model(self, model, name):
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), f"model/{name}.pt")
