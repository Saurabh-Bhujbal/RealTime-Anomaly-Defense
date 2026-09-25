# -*- coding: utf-8 -*-
"""
cifar10_dataset.py
==================
Dataset class and loader for CIFAR-10.

CIFAR10Dataset
--------------
Accepts pre-loaded numpy uint8 arrays of shape (N, 32, 32, 3) and
yields (image_tensor, label) pairs where image_tensor is float32 in
[0, 1] with shape (3, 32, 32)  — channel-first, ready for CIFAR10CNN.

load_cifar10
------------
Loads the dataset using torchvision (which caches into data/cifar10/)
and returns four numpy arrays with the same signature as load_mnist():

    (train_images, train_labels, test_images, test_labels)

    train_images : (50000, 32, 32, 3)  uint8
    train_labels : (50000,)            int
    test_images  : (10000, 32, 32, 3)  uint8
    test_labels  : (10000,)            int

Note
----
The raw tar.gz bundled at data/cifar10/cifar-10-python.tar.gz is
incomplete (EOFError), so we fall back to torchvision's auto-downloader,
which stores the extracted batches in data/cifar10/cifar-10-batches-py/.
torchvision.datasets.CIFAR10 already returns .data as (N, 32, 32, 3)
uint8, so no reshape is needed beyond converting targets to numpy.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Dataset class
# ============================================================

class CIFAR10Dataset(Dataset):
    """
    Lightweight wrapper around pre-loaded CIFAR-10 numpy arrays.

    Parameters
    ----------
    images : np.ndarray, shape (N, 32, 32, 3), dtype uint8
    labels : array-like, length N
    """

    def __init__(self, images: np.ndarray, labels):
        self.images = images   # numpy uint8, shape (N, 32, 32, 3)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        # Normalise to [0, 1] and convert HWC -> CHW: (32,32,3) -> (3,32,32)
        image = (
            torch.tensor(self.images[index], dtype=torch.float32) / 255.0
        )
        image = image.permute(2, 0, 1)   # (H, W, C) -> (C, H, W)

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, label


# ============================================================
# Loader
# ============================================================

def load_cifar10(data_dir):
    """
    Load CIFAR-10 train and test splits.

    Parameters
    ----------
    data_dir : str or Path
        Directory where the dataset is cached (e.g. ``data/cifar10``).
        torchvision will download the dataset here on the first call if
        the extracted batches are not already present.

    Returns
    -------
    train_images : np.ndarray  (50000, 32, 32, 3)  uint8
    train_labels : np.ndarray  (50000,)             int64
    test_images  : np.ndarray  (10000, 32, 32, 3)  uint8
    test_labels  : np.ndarray  (10000,)             int64
    """

    # torchvision import is intentionally local so that the rest of the
    # module stays importable even without it installed.
    from torchvision import datasets as tv_datasets

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_ds = tv_datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
    )

    test_ds = tv_datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
    )

    # .data is already (N, 32, 32, 3) uint8 for torchvision CIFAR10
    train_images = train_ds.data                          # (50000, 32, 32, 3)
    train_labels = np.array(train_ds.targets, dtype=np.int64)

    test_images  = test_ds.data                           # (10000, 32, 32, 3)
    test_labels  = np.array(test_ds.targets, dtype=np.int64)

    return train_images, train_labels, test_images, test_labels
