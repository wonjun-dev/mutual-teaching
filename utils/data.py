import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_dir, data_name, transform=None, mutual=True):

        data_dir = os.path.join(root_dir, data_name)
        classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {data_dir}.")

        instances = []
        for target_class in classes:
            class_index = int(target_class)
            target_dir = os.path.join(data_dir, target_class)

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    key = fname.split(".")[0]
                    item = path, class_index, key
                    instances.append(item)

        self.samples = instances
        self.transform = transform
        self.mutual = mutual

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mutual:
            path, target, key = self.samples[idx]
            with open(path, "rb") as f:
                img_1 = Image.open(f).convert("RGB")
                img_2 = img_1.copy()

            if self.transform is not None:
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)

            return img_1, img_2, target, key

        else:
            path, target, key = self.samples[idx]
            with open(path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")

            if self.transform is not None:
                img = self.transform(img)

            return img, target, key


if __name__ == "__main__":
    height = 256
    width = 128
    transform = transforms.Compose(
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
    data = ImageDataset(
        root_dir="../datasets/market1501/Market-1501-v15.09.15/pytorch",
        data_name="train",
        transform=transform,
        # mutual=False,
    )
    dataloader = DataLoader(data, batch_size=126, shuffle=True)

    for batch_idx, samples in enumerate(dataloader):
        print(batch_idx, samples[0][0].shape, samples[-1])
        print(samples[0][0] - samples[0][-1])
