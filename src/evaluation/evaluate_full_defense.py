from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist
from src.data.preprocessing import MNISTDataset
from src.models.cnn import MNISTCNN

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

from src.detection.anomaly_detector import (
    DynamicAnomalyDetector
)

from src.purification.purifier import (
    SelfPurifier
)

from src.defenses.randomized_defense import (
    RandomizedDefense
)

from src.defenses.gradient_diversity import (
    GradientDiversityDefense
)

from src.defenses.combined_defense import (
    CombinedDefense
)

import config


def evaluate_attack(
    model,
    defense,
    loader,
    device,
    attack_name,
    epsilon
):

    total = 0

    attack_correct = 0
    defense_correct = 0
    detected = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Generate adversarial samples
        if attack_name == "FGSM":

            adversarial_images = fgsm_attack(
                model,
                images,
                labels,
                epsilon=epsilon
            )

        elif attack_name == "PGD":

            adversarial_images = pgd_attack(
                model,
                images,
                labels,
                epsilon=epsilon,
                alpha=0.01,
                steps=10
            )

        else:

            raise ValueError(
                "Unknown attack"
            )

        # Accuracy before defense
        with torch.no_grad():

            logits = model(
                adversarial_images
            )

            predictions = logits.argmax(
                dim=1
            )

        attack_correct += (
            predictions == labels
        ).sum().item()

        # Full defense
        result = defense.predict(
            adversarial_images
        )

        final_predictions = (
            result["predictions"]
        )

        anomalous = (
            result["anomalous"]
        )

        defense_correct += (
            final_predictions == labels
        ).sum().item()

        detected += (
            anomalous.sum().item()
        )

        total += labels.size(0)

    return {
        "attack_accuracy":
            100.0 * attack_correct / total,

        "defended_accuracy":
            100.0 * defense_correct / total,

        "detection_rate":
            100.0 * detected / total
    }


def main():

    print("=" * 80)
    print("FULL ADVERSARIAL DEFENSE EVALUATION")
    print("=" * 80)

    device = config.DEVICE

    print(f"Device: {device}")

    # ======================================================
    # DATASET
    # ======================================================

    (
        train_images,
        train_labels,
        test_images,
        test_labels
    ) = load_mnist(
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
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # ======================================================
    # MODEL
    # ======================================================

    model = MNISTCNN(
        num_classes=config.NUM_CLASSES
    ).to(device)

    model.load_state_dict(
        torch.load(
            config.MODEL_DIR /
            "baseline_cnn.pth",
            map_location=device
        )
    )

    model.eval()

    print("✓ Baseline CNN loaded")

    # ======================================================
    # ANOMALY DETECTOR
    # ======================================================

    detector = DynamicAnomalyDetector(
        model,
        threshold_percentile=95.0
    )

    threshold = detector.calibrate(
        train_loader,
        device
    )

    print(
        f"✓ Anomaly threshold: "
        f"{threshold:.6f}"
    )

    # ======================================================
    # PURIFIER
    # ======================================================

    purifier = SelfPurifier(
        model
    )

    print("✓ Self-purifier ready")

    # ======================================================
    # RANDOMIZED DEFENSE
    # ======================================================

    randomized = RandomizedDefense(
        model,
        num_samples=4,
        noise_std=0.03
    )

    print("✓ Randomized defense ready")

    # ======================================================
    # GRADIENT DIVERSITY
    # ======================================================

    gradient_defense = (
        GradientDiversityDefense(
            model,
            consistency_threshold=0.75
        )
    )

    print("✓ Gradient diversity defense ready")

    # ======================================================
    # COMBINED DEFENSE
    # ======================================================

    combined = CombinedDefense(
        model=model,
        detector=detector,
        purifier=purifier,
        randomized_defense=randomized,
        gradient_defense=gradient_defense
    )

    print("✓ Combined defense ready")

    # ======================================================
    # FGSM
    # ======================================================

    print()
    print("-" * 80)
    print("FGSM + FULL DEFENSE")
    print("-" * 80)

    for epsilon in [
        0.05,
        0.10,
        0.20
    ]:

        results = evaluate_attack(
            model,
            combined,
            test_loader,
            device,
            "FGSM",
            epsilon
        )

        print(
            f"FGSM ε={epsilon:.2f} | "
            f"Attack Accuracy: "
            f"{results['attack_accuracy']:.2f}% | "
            f"Defended Accuracy: "
            f"{results['defended_accuracy']:.2f}% | "
            f"Detection Rate: "
            f"{results['detection_rate']:.2f}%"
        )

    # ======================================================
    # PGD
    # ======================================================

    print()
    print("-" * 80)
    print("PGD + FULL DEFENSE")
    print("-" * 80)

    for epsilon in [
        0.05,
        0.10,
        0.20
    ]:

        results = evaluate_attack(
            model,
            combined,
            test_loader,
            device,
            "PGD",
            epsilon
        )

        print(
            f"PGD ε={epsilon:.2f} | "
            f"Attack Accuracy: "
            f"{results['attack_accuracy']:.2f}% | "
            f"Defended Accuracy: "
            f"{results['defended_accuracy']:.2f}% | "
            f"Detection Rate: "
            f"{results['detection_rate']:.2f}%"
        )

    print()
    print("=" * 80)
    print("FULL DEFENSE EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()