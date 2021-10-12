import argparse

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data import ImageDataset
from utils.clustering import KMeansCluster
from model import ReidResNet

from mutual_teaching import MutualTeaching

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--pretrain", action="store_true")
opt = parser.parse_args()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.pretrain:
        num_classes = 1501
        model_1 = ReidResNet(num_classes=num_classes)
        model_1.to(device)
        optimizer = torch.optim.Adam(model_1.parameters())
    else:  # MMT
        num_classes = 500
        model_1 = ReidResNet(num_classes=num_classes)
        model_2 = ReidResNet(num_classes=num_classes)
        model_1.to(device)
        model_2.to(device)
        model_cluster = KMeansCluster(n_clusters=500)
        optimizer = torch.optim.Adam(
            [{"params": model_1.parameters()}, {"params": model_2.parameters()}]
        )

    # config
    height = 256
    width = 128
    train_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(padding=10),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.Pad(padding=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cluster_transform = val_transform

    train_dataset = ImageDataset(
        root_dir="datasets/market1501/Market-1501-v15.09.15/pytorch",
        data_name="train",
        transform=train_transform,
        mutual=True,
    )
    val_dataset = ImageDataset(
        root_dir="datasets/market1501/Market-1501-v15.09.15/pytorch",
        data_name="val",
        transform=val_transform,
        mutual=False,
    )
    cluster_dataset = ImageDataset(
        root_dir="datasets/market1501/Market-1501-v15.09.15/pytorch",
        data_name="train",
        transform=cluster_transform,
        mutual=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    cluster_loader = DataLoader(cluster_dataset, batch_size=32, shuffle=False)

    if opt.pretrain:
        print(f"Pretrain...")
        mt = MutualTeaching(model_1, None, None, optimizer, device, mmt=False)
        for e in range(20):
            mt.noraml_training_loop(train_loader, epoch=e)  # 지도학습을 통한 pretrain 모델 생성
    else:
        print(f"Mutual Mean Teaching...")
        mt = MutualTeaching(model_1, model_2, model_cluster, optimizer, device)
        for e in range(100):
            mt.training_loop(train_loader, cluster_loader, epoch=e)


if __name__ == "__main__":
    main()
