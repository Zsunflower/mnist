import torch
from torch.utils.data import Dataset
import numpy as np


class MnistDataset(Dataset):
    def __init__(self, image_data_path, label_data_path, transform=None):
        with open(image_data_path, "rb") as f:
            image_data = f.read()
        with open(label_data_path, "rb") as f:
            label_data = f.read()
        images = np.frombuffer(image_data[16:], dtype=np.uint8)
        labels = np.frombuffer(label_data[8:], dtype=np.uint8)
        self.images = images.reshape((-1, 28, 28, 1))
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {"X": self.images[idx], "Y": self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(list_samples):
    images = [sample["X"] for sample in list_samples]
    labels = [sample["Y"] for sample in list_samples]
    return {"X": torch.stack(images, dim=0), "Y": torch.cat(labels, axis=0)}
