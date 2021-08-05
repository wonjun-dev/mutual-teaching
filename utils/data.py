import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Market1501(Dataset):
    def __init__(self, root_dir, data_name="bounding_box_train"):

        data_dir = os.path.join(root_dir, data_name)
        file_names = os.listdir(data_dir)
        # TODO make label for supervised learning.

        self.files_dir = [os.path.join(data_dir, f) for f in file_names if ".jpg" in f]

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.641, 0.619, 0.612], std=[0.254, 0.262, 0.258]),
            ]
        )

    def __len__(self):
        return len(self.files_dir)

    def __getitem__(self, idx):
        image = Image.open(self.files_dir[idx])
        image = self.preprocess(image)
        return image


class DukeMTMC(Dataset):
    def __init__(self, root_dir):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    data = Market1501(root_dir="../datasets/market1501/Market-1501-v15.09.15")
    dataloader = DataLoader(data, batch_size=126, shuffle=True)

    for batch_idx, samples in enumerate(dataloader):
        print(batch_idx, samples.shape)
