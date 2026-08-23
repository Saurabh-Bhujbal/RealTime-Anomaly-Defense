import sys
import os
import json
import struct

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import MNISTCNN
from src.attacks.cw import carlini_wagner_attack


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "baseline_cnn.pth"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

TEST_IMAGES_PATH = os.path.join(
    DATA_PATH,
    "t10k-images-idx3-ubyte"
)

TEST_LABELS_PATH = os.path.join(
    DATA_PATH,
    "t10k-labels-idx1-ubyte"
)

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "results",
    "cw_results.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD RAW IDX IMAGES
# ============================================================

def load_idx_images(path):

    print(
        f"Loading images from:\n{path}"
    )

    with open(path, "rb") as f:

        magic, count, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        data = f.read()

    if magic != 2051:
        raise ValueError(
            f"Invalid MNIST image file: {path}"
        )

    images = torch.frombuffer(
        data,
        dtype=torch.uint8
    ).clone()

    images = images.reshape(
        count,
        rows,
        cols
    )

    # Convert [0,255] → [0,1]
    images = images.float() / 255.0

    # CNN expects:
    # [N, 1, 28, 28]

    images = images.unsqueeze(1)

    return images


# ============================================================
# LOAD RAW IDX LABELS
# ============================================================

def load_idx_labels(path):

    print(
        f"Loading labels from:\n{path}"
    )

    with open(path, "rb") as f:

        magic, count = struct.unpack(
            ">II",
            f.read(8)
        )

        data = f.read()

    if magic != 2049:
        raise ValueError(
            f"Invalid MNIST label file: {path}"
        )

    labels = torch.frombuffer(
        data,
        dtype=torch.uint8
    ).clone()

    return labels.long()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading baseline CNN..."
    )

    model = MNISTCNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "✓ Baseline CNN loaded"
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CARLINI-WAGNER ATTACK EVALUATION")
    print("=" * 80)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # CHECK DATASET
    # ========================================================

    print("\nChecking existing MNIST dataset...")

    required_files = [
        TEST_IMAGES_PATH,
        TEST_LABELS_PATH
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"\nMNIST file not found:\n{file_path}"
            )

    print(
        "✓ Existing MNIST IDX files found"
    )

    # ========================================================
    # LOAD EXISTING DATASET
    # ========================================================

    images = load_idx_images(
        TEST_IMAGES_PATH
    )

    labels = load_idx_labels(
        TEST_LABELS_PATH
    )

    if len(images) != len(labels):

        raise ValueError(
            "Number of images and labels do not match."
        )

    print(
        f"✓ Test images loaded: {len(images)}"
    )

    print(
        f"✓ Test labels loaded: {len(labels)}"
    )

    print(
        f"✓ Image shape: {tuple(images.shape)}"
    )

    # ========================================================
    # CREATE DATASET
    # ========================================================

    dataset = TensorDataset(
        images,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # EVALUATION VARIABLES
    # ========================================================

    total = 0

    clean_correct = 0

    adversarial_correct = 0

    successful_attacks = 0

    total_l2 = 0.0

    # ========================================================
    # FIRST EXPERIMENT
    # ========================================================

    # 20 batches × 32 images = 640 images

    max_batches = 20

    print()
    print("=" * 80)
    print("RUNNING C&W ATTACK")
    print("=" * 80)

    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch_index, (
        batch_images,
        batch_labels
    ) in enumerate(loader):

        if batch_index >= max_batches:

            break

        batch_images = batch_images.to(
            DEVICE
        )

        batch_labels = batch_labels.to(
            DEVICE
        )

        # ====================================================
        # CLEAN PREDICTION
        # ====================================================

        with torch.no_grad():

            clean_logits = model(
                batch_images
            )

            clean_predictions = (
                clean_logits.argmax(
                    dim=1
                )
            )

        clean_correct += (
            clean_predictions ==
            batch_labels
        ).sum().item()

        # ====================================================
        # C&W ATTACK
        # ====================================================

        adversarial_images = (
            carlini_wagner_attack(
                model=model,
                images=batch_images,
                labels=batch_labels,
                targeted=False,
                c=1.0,
                kappa=0.0,
                lr=0.01,
                steps=100,
                device=DEVICE
            )
        )

        # ====================================================
        # ADVERSARIAL PREDICTION
        # ====================================================

        with torch.no_grad():

            adversarial_logits = model(
                adversarial_images
            )

            adversarial_predictions = (
                adversarial_logits.argmax(
                    dim=1
                )
            )

        adversarial_correct += (
            adversarial_predictions ==
            batch_labels
        ).sum().item()

        successful_attacks += (
            adversarial_predictions !=
            batch_labels
        ).sum().item()

        # ====================================================
        # L2 PERTURBATION
        # ====================================================

        batch_l2 = (
            (
                adversarial_images -
                batch_images
            ) ** 2
        ).view(
            batch_images.size(0),
            -1
        ).sum(dim=1)

        total_l2 += (
            batch_l2.sum().item()
        )

        total += (
            batch_images.size(0)
        )

        print(
            f"Batch "
            f"{batch_index + 1:02d}/{max_batches} "
            f"completed"
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    clean_accuracy = (
        clean_correct /
        total
    ) * 100

    adversarial_accuracy = (
        adversarial_correct /
        total
    ) * 100

    attack_success_rate = (
        successful_attacks /
        total
    ) * 100

    average_l2 = (
        total_l2 /
        total
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "attack": "Carlini-Wagner",

        "dataset": "MNIST",

        "dataset_source": (
            "Existing project IDX files"
        ),

        "device": DEVICE,

        "samples_evaluated": total,

        "clean_accuracy":
            clean_accuracy,

        "adversarial_accuracy":
            adversarial_accuracy,

        "attack_success_rate":
            attack_success_rate,

        "average_l2":
            average_l2,

        "parameters": {

            "c": 1.0,

            "kappa": 0.0,

            "learning_rate": 0.01,

            "steps": 100,

            "batch_size": 32
        }
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        os.path.dirname(
            RESULT_PATH
        ),
        exist_ok=True
    )

    with open(
        RESULT_PATH,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("C&W RESULTS")
    print("=" * 80)

    print(
        f"Samples Evaluated      : "
        f"{total}"
    )

    print(
        f"Clean Accuracy         : "
        f"{clean_accuracy:.2f}%"
    )

    print(
        f"C&W Accuracy           : "
        f"{adversarial_accuracy:.2f}%"
    )

    print(
        f"Attack Success Rate    : "
        f"{attack_success_rate:.2f}%"
    )

    print(
        f"Average L2 Perturbation: "
        f"{average_l2:.6f}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print("=" * 80)
    print("C&W EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()