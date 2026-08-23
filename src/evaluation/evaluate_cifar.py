import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.cifar10_cnn import CIFAR10CNN


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cifar10_cnn.pth"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cifar10"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 70)
    print("CIFAR-10 MODEL EVALUATION")
    print("=" * 70)

    print("Device:", DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    model = CIFAR10CNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total

    print()
    print("=" * 70)
    print("CIFAR-10 RESULTS")
    print("=" * 70)

    print(f"Samples          : {total}")
    print(f"Clean Accuracy   : {accuracy:.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()