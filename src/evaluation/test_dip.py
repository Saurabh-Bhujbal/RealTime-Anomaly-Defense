import sys
import os
import struct

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

import torch

from src.purification.dip_purifier import (
    DIPPurifier
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_first_mnist_image():

    path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "t10k-images-idx3-ubyte"
    )

    with open(path, "rb") as f:

        magic, count, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        data = f.read(
            rows * cols
        )

    image = torch.tensor(
        list(data),
        dtype=torch.float32
    )

    image = image.reshape(
        1,
        1,
        rows,
        cols
    )

    image = image / 255.0

    return image


def main():

    print("=" * 70)
    print("DIP PURIFIER TEST")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    image = load_first_mnist_image()

    image = image.to(
        DEVICE
    )

    print(
        f"Input shape: "
        f"{tuple(image.shape)}"
    )

    purifier = DIPPurifier(
        iterations=100,
        learning_rate=0.01,
        device=DEVICE
    )

    print(
        "✓ DIP purifier created"
    )

    print(
        "Running DIP optimization..."
    )

    purified = purifier.purify(
        image
    )

    print(
        f"✓ Purified shape: "
        f"{tuple(purified.shape)}"
    )

    print(
        f"Input range: "
        f"{image.min().item():.4f} - "
        f"{image.max().item():.4f}"
    )

    print(
        f"Output range: "
        f"{purified.min().item():.4f} - "
        f"{purified.max().item():.4f}"
    )

    difference = torch.mean(
        torch.abs(
            purified - image
        )
    )

    print(
        f"Mean absolute difference: "
        f"{difference.item():.6f}"
    )

    print("=" * 70)
    print("DIP TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()