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
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

try:
    from src.attacks.eot_pgd import eot_pgd_attack
    EOT_AVAILABLE = True
except ImportError:
    EOT_AVAILABLE = False

try:
    from src.attacks.cw import carlini_wagner_attack
    CW_AVAILABLE = True
except ImportError:
    CW_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "graddiv_cnn.pth"
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
    "graddiv_evaluation_results.json"
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MNIST IDX IMAGES
# ============================================================

def load_idx_images(path):

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

    images = torch.tensor(
        list(data),
        dtype=torch.float32
    )

    images = images.reshape(
        count,
        rows,
        cols
    )

    images = images / 255.0

    return images.unsqueeze(1)


# ============================================================
# LOAD MNIST IDX LABELS
# ============================================================

def load_idx_labels(path):

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

    labels = torch.tensor(
        list(data),
        dtype=torch.long
    )

    return labels


# ============================================================
# LOAD GRADDIV MODEL
# ============================================================

def load_model():

    model = MNISTCNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
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

    return model


# ============================================================
# ACCURACY
# ============================================================

def evaluate_clean(
    model,
    loader
):

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        100.0 *
        correct /
        total
    )


# ============================================================
# ATTACK EVALUATION
# ============================================================

def evaluate_attack(
    model,
    loader,
    attack_name,
    attack_function,
    attack_kwargs,
    max_batches=20
):

    correct = 0
    total = 0

    for batch_index, (
        images,
        labels
    ) in enumerate(loader):

        if batch_index >= max_batches:
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        adversarial_images = attack_function(
            model,
            images,
            labels,
            **attack_kwargs
        )

        with torch.no_grad():

            logits = model(
                adversarial_images
            )

            predictions = logits.argmax(
                dim=1
            )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        print(
            f"{attack_name}: "
            f"batch "
            f"{batch_index + 1}/{max_batches}"
        )

    accuracy = (
        100.0 *
        correct /
        total
    )

    return accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GRADDIV MODEL EVALUATION")
    print("=" * 80)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"GradDiv model not found:\n{MODEL_PATH}"
        )

    print(
        "✓ GradDiv model found"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading existing MNIST test dataset..."
    )

    images = load_idx_images(
        TEST_IMAGES_PATH
    )

    labels = load_idx_labels(
        TEST_LABELS_PATH
    )

    dataset = TensorDataset(
        images,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    print(
        f"✓ Test samples: {len(dataset)}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    print(
        "✓ GradDiv CNN loaded"
    )

    # ========================================================
    # CLEAN
    # ========================================================

    print()
    print("-" * 80)
    print("CLEAN ACCURACY")
    print("-" * 80)

    clean_accuracy = evaluate_clean(
        model,
        loader
    )

    print(
        f"Clean Accuracy: "
        f"{clean_accuracy:.2f}%"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "model": "GradDiv CNN",

        "dataset": "MNIST",

        "device": DEVICE,

        "clean_accuracy":
            clean_accuracy
    }

    # ========================================================
    # FGSM
    # ========================================================

    print()
    print("-" * 80)
    print("FGSM")
    print("-" * 80)

    fgsm_results = {}

    for epsilon in [0.10, 0.20]:

        print(
            f"\nFGSM epsilon={epsilon:.2f}"
        )

        accuracy = evaluate_attack(
            model=model,
            loader=loader,
            attack_name="FGSM",
            attack_function=fgsm_attack,
            attack_kwargs={
                "epsilon": epsilon
            }
        )

        fgsm_results[
            str(epsilon)
        ] = accuracy

        print(
            f"Accuracy: {accuracy:.2f}%"
        )

    results["fgsm"] = fgsm_results

    # ========================================================
    # PGD
    # ========================================================

    print()
    print("-" * 80)
    print("PGD")
    print("-" * 80)

    pgd_results = {}

    for epsilon in [0.10, 0.20]:

        print(
            f"\nPGD epsilon={epsilon:.2f}"
        )

        accuracy = evaluate_attack(
            model=model,
            loader=loader,
            attack_name="PGD",
            attack_function=pgd_attack,
            attack_kwargs={
                "epsilon": epsilon,
                "alpha": 0.01,
                "steps": 10
            }
        )

        pgd_results[
            str(epsilon)
        ] = accuracy

        print(
            f"Accuracy: {accuracy:.2f}%"
        )

    results["pgd"] = pgd_results

    # ========================================================
    # C&W
    # ========================================================

    if CW_AVAILABLE:

        print()
        print("-" * 80)
        print("C&W")
        print("-" * 80)

        print(
            "\nC&W evaluation"
        )

        accuracy = evaluate_attack(
            model=model,
            loader=loader,
            attack_name="C&W",
            attack_function=carlini_wagner_attack,
            attack_kwargs={
                "targeted": False,
                "c": 1.0,
                "kappa": 0.0,
                "lr": 0.01,
                "steps": 100,
                "device": DEVICE
            }
        )

        results["cw"] = accuracy

        print(
            f"C&W Accuracy: "
            f"{accuracy:.2f}%"
        )

    else:

        print(
            "\n⚠ C&W module not found"
        )

    # ========================================================
    # EOT-PGD
    # ========================================================

    if EOT_AVAILABLE:

        print()
        print("-" * 80)
        print("EOT-PGD")
        print("-" * 80)

        print(
            "\nEOT-PGD evaluation"
        )

        try:

            accuracy = evaluate_attack(
                model=model,
                loader=loader,
                attack_name="EOT-PGD",
                attack_function=eot_pgd_attack,
                attack_kwargs={
                    "epsilon": 0.20,
                    "alpha": 0.01,
                    "steps": 10
                }
            )

            results["eot_pgd"] = accuracy

            print(
                f"EOT-PGD Accuracy: "
                f"{accuracy:.2f}%"
            )

        except TypeError as error:

            print(
                "⚠ Existing EOT-PGD function "
                "uses different arguments."
            )

            print(
                f"Details: {error}"
            )

    else:

        print(
            "\n⚠ EOT-PGD module not found"
        )

    # ========================================================
    # SAVE
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
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("GRADDIV EVALUATION COMPLETE")
    print("=" * 80)

    print(
        json.dumps(
            results,
            indent=4
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print("=" * 80)


if __name__ == "__main__":
    main()