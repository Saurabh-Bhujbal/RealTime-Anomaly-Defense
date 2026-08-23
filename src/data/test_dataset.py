from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist


def main():

    data_dir = PROJECT_ROOT / "data" / "raw"

    print("=" * 60)
    print("MNIST DATASET VERIFICATION")
    print("=" * 60)

    train_images, train_labels, test_images, test_labels = load_mnist(
        data_dir
    )

    print(f"Train images shape : {train_images.shape}")
    print(f"Train labels shape : {train_labels.shape}")
    print(f"Test images shape  : {test_images.shape}")
    print(f"Test labels shape  : {test_labels.shape}")

    print()

    print(f"Train image dtype  : {train_images.dtype}")
    print(f"Train label dtype  : {train_labels.dtype}")

    print()

    print(f"Pixel minimum      : {train_images.min()}")
    print(f"Pixel maximum      : {train_images.max()}")

    print()

    print(f"Number of classes  : {len(set(train_labels.tolist()))}")
    print(f"Classes            : {sorted(set(train_labels.tolist()))}")

    print()

    print("First 10 labels:")
    print(train_labels[:10])

    print()

    print("First image shape:")
    print(train_images[0].shape)

    print("=" * 60)
    print("MNIST DATASET VERIFIED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()