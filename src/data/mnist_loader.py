import struct
import numpy as np
from pathlib import Path


def load_mnist_images(file_path):
    """
    Load MNIST images from the original IDX binary format.
    Returns a NumPy array of shape:
    (number_of_images, 28, 28)
    """

    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic_number != 2051:
            raise ValueError(
                f"Invalid MNIST image file. Expected magic number 2051, "
                f"got {magic_number}"
            )

        image_data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    images = image_data.reshape(num_images, rows, cols)

    return images


def load_mnist_labels(file_path):
    """
    Load MNIST labels from the original IDX binary format.
    Returns a NumPy array of shape:
    (number_of_labels,)
    """

    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        magic_number, num_labels = struct.unpack(">II", f.read(8))

        if magic_number != 2049:
            raise ValueError(
                f"Invalid MNIST label file. Expected magic number 2049, "
                f"got {magic_number}"
            )

        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    if len(labels) != num_labels:
        raise ValueError(
            f"Label count mismatch. Expected {num_labels}, "
            f"got {len(labels)}"
        )

    return labels


def load_mnist(data_dir):
    """
    Load the complete MNIST dataset.

    Expected directory structure:

    data/
        raw/
            train-images-idx3-ubyte
            train-labels-idx1-ubyte
            t10k-images-idx3-ubyte
            t10k-labels-idx1-ubyte
    """

    data_dir = Path(data_dir)

    train_images_path = data_dir / "train-images-idx3-ubyte"
    train_labels_path = data_dir / "train-labels-idx1-ubyte"

    test_images_path = data_dir / "t10k-images-idx3-ubyte"
    test_labels_path = data_dir / "t10k-labels-idx1-ubyte"

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)

    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    return (
        train_images,
        train_labels,
        test_images,
        test_labels
    )