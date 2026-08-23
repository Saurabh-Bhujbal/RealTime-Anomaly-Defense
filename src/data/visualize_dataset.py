from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist


def main():

    data_dir = PROJECT_ROOT / "data" / "raw"

    train_images, train_labels, _, _ = load_mnist(data_dir)

    plt.figure(figsize=(10, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        plt.imshow(
            train_images[i],
            cmap="gray"
        )

        plt.title(f"Label: {train_labels[i]}")
        plt.axis("off")

    plt.tight_layout()

    output_dir = PROJECT_ROOT / "experiments" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "mnist_samples.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()