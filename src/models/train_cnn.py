from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist
from src.data.preprocessing import MNISTDataset
from src.models.cnn import MNISTCNN

import config


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


def main():

    set_seed(config.RANDOM_SEED)

    device = config.DEVICE

    print("=" * 60)
    print("BASELINE CNN TRAINING")
    print("=" * 60)

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Load MNIST
    # --------------------------------------------------------

    train_images, train_labels, test_images, test_labels = load_mnist(
        config.DATA_DIR
    )

    train_dataset = MNISTDataset(
        train_images,
        train_labels
    )

    test_dataset = MNISTDataset(
        test_images,
        test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MNISTCNN(
        num_classes=config.NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(config.EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_accuracy = evaluate(
            model,
            train_loader,
            device
        )

        test_accuracy = evaluate(
            model,
            test_loader,
            device
        )

        average_loss = (
            running_loss / len(train_loader)
        )

        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"Loss: {average_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Test Acc: {test_accuracy:.2f}%"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    config.MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        config.MODEL_DIR /
        "baseline_cnn.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    print(
        f"Model saved to: {model_path}"
    )


if __name__ == "__main__":
    main()