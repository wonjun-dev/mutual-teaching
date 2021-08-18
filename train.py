import torch
from torch.utils.data import DataLoader

from utils.data import Market1501
from utils.clustering import KMeansCluster
from model import ReidResNet

from mutual_teaching import MutualTeaching


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

    train_dataset = Market1501(
        "datasets/market1501/Market-1501-v15.09.15", data_name="bounding_box_train", mutual=True
    )
    train_loader = DataLoader(train_dataset, batch_size=32)

    mt = MutualTeaching(model_1, model_2, model_cluster, optimizer, device)

    for e in range(100):
        mt.training_loop(train_loader, epoch=e)


if __name__ == "__main__":
    main()
