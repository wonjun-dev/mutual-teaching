import os
from PIL import Image
from numpy.core.fromnumeric import mean
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ToTensor


class Market1501(Dataset):
    def __init__(
        self, root_dir, data_name="bounding_box_train", height=256, width=128, mutual=True
    ):

        data_dir = os.path.join(root_dir, data_name)
        file_names = os.listdir(data_dir)
        # TODO make label for supervised learning.

        self.files_dir = [os.path.join(data_dir, f) for f in file_names if ".jpg" in f]
        self.pseudo_labels = None

        self.transform = transforms.Compose(
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
        self.mutual = mutual

    def __len__(self):
        return len(self.files_dir)

    def __getitem__(self, idx):
        if self.mutual:
            img_1 = Image.open(self.files_dir[idx]).convert("RGB")
            img_2 = img_1.copy()

            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

            if self.pseudo_labels is not None:
                y_tilde = self.pseudo_labels[idx]
            else:
                y_tilde = []

            return img_1, img_2, y_tilde

        else:
            img = Image.open(self.files_dir[idx]).convert("RGB")
            img = self.transform(img)

            if self.pseudo_labels is not None:
                y_tilde = self.pseudo_labels[idx]
            else:
                y_tilde = []

            return img, y_tilde


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
