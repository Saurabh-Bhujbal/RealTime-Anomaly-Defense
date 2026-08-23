import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """
    PyTorch Dataset for MNIST NumPy arrays.
    """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Convert uint8 [0,255] -> float32 [0,1]
        image = torch.tensor(
            image,
            dtype=torch.float32
        ) / 255.0

        # Add channel dimension: 28x28 -> 1x28x28
        image = image.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label
        