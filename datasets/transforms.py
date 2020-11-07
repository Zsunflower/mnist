import torch
import numpy as np


class ToTensor(object):
    def __call__(self, sample):
        image = sample["X"].astype(np.float32)
        label = sample["Y"]
        image = image.transpose(2, 1, 0)
        return {
            "X": torch.from_numpy(image),
            "Y": torch.tensor([label], dtype=torch.long),
        }
