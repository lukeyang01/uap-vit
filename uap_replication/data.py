# Utilities for the following:
# - Loading ImageNet training and validation data.
# - Loading and saving UAPs.


import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


def load_random_subset(path, size, transforms):
    dataset = ImageFolder(root=path, transform=transforms)
    lengths = (size, len(dataset) - size)
    return random_split(dataset, lengths)[0]


def load_uap(path):
    try:
        return torch.load(path)
    except:
        return torch.Tensor([np.load(path)])


def save_uap(v, path):
    torch.save(v, path)
