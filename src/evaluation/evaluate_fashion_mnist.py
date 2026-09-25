# -*- coding: utf-8 -*-
"""
evaluate_fashion_mnist.py
=========================
Full adversarial defense evaluation for the Fashion-MNIST dataset.

Covers:
  - Clean accuracy
  - FGSM  (ε = 0.05, 0.10, 0.20)
  - PGD   (ε = 0.05, 0.10, 0.20)
  - EOT-PGD (ε = 0.20, 10 steps, 4 EOT samples)

Results are saved to:
  experiments/results/fashion_mnist_full_defense.json
"""

from pathlib import Path
import sys
import json

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist
from src.data.preprocessing import MNISTDataset
from src.models.cnn import MNISTCNN

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.eot_pgd import eot_pgd_attack

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


# ============================================================
# Helper: clean accuracy
# ============================================================

def evaluate_clean(model, loader, device):
    """Return clean accuracy (%) with no defense applied."""

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

    return 100.0 * correct / total


# ============================================================
# Helper: FGSM / PGD attack + defense evaluation
# ============================================================

def evaluate_attack(
    model,
    defense,
    loader,
    device,
    attack_name,
    epsilon
):
    """
    Generate adversarial examples with FGSM or PGD,
    measure per-sample attack success and full-defense recovery.

    Returns a dict with keys:
      attack_accuracy   – model accuracy on adversarial inputs (no defense)
      defended_accuracy – model accuracy after full combined defense
      detection_rate    – fraction of adversarial samples flagged by detector
    """

    total = 0

    attack_correct = 0
    defense_correct = 0
    detected = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # --------------------------------------------------
        # Generate adversarial samples
        # --------------------------------------------------

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
                f"Unknown attack: {attack_name}"
            )

        # --------------------------------------------------
        # Accuracy before defense (no-defense baseline)
        # --------------------------------------------------

        with torch.no_grad():

            logits = model(adversarial_images)
            predictions = logits.argmax(dim=1)

        attack_correct += (
            predictions == labels
        ).sum().item()

        # --------------------------------------------------
        # Full combined defense
        # --------------------------------------------------

        result = defense.predict(adversarial_images)

        final_predictions = result["predictions"]
        anomalous = result["anomalous"]

        defense_correct += (
            final_predictions == labels
        ).sum().item()

        detected += anomalous.sum().item()

        total += labels.size(0)

    return {
        "attack_accuracy":
            100.0 * attack_correct / total,

        "defended_accuracy":
            100.0 * defense_correct / total,

        "detection_rate":
            100.0 * detected / total
    }


# ============================================================
# Helper: EOT-PGD evaluation
# ============================================================

def evaluate_eot(
    model,
    defense,
    loader,
    device,
    epsilon=0.20,
    attack_steps=10,
    eot_samples=4
):
    """
    Adaptive EOT-PGD attack that averages gradients over randomized
    transformations applied by the defense.

    Returns the same dict schema as evaluate_attack().
    """

    total = 0

    attack_correct = 0
    defended_correct = 0
    detected = 0

    for batch_index, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        # Adaptive EOT-PGD attack
        adversarial_images = eot_pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=epsilon,
            alpha=0.01,
            steps=attack_steps,
            eot_samples=eot_samples
        )

        # Attack accuracy (no defense)
        with torch.no_grad():

            logits = model(adversarial_images)
            predictions = logits.argmax(dim=1)

        attack_correct += (
            predictions == labels
        ).sum().item()

        # Full defense
        result = defense.predict(adversarial_images)

        defended_predictions = result["predictions"]
        anomalous = result["anomalous"]

        defended_correct += (
            defended_predictions == labels
        ).sum().item()

        detected += anomalous.sum().item()

        total += labels.size(0)

        # Progress print every 20 batches
        if (batch_index + 1) % 20 == 0:

            print(
                f"  EOT-PGD: processed "
                f"{batch_index + 1} batches..."
            )

    return {
        "attack_accuracy":
            100.0 * attack_correct / total,

        "defended_accuracy":
            100.0 * defended_correct / total,

        "detection_rate":
            100.0 * detected / total
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("FASHION-MNIST - FULL ADVERSARIAL DEFENSE EVALUATION")
    print("=" * 80)

    device = config.DEVICE

    fashion_data_dir = config.DATA_DIR / "fashion_mnist"

    print(f"Device   : {device}")
    print(f"Data dir : {fashion_data_dir}")

    # ======================================================
    # Dataset
    # ======================================================

    (
        train_images,
        train_labels,
        test_images,
        test_labels
    ) = load_mnist(fashion_data_dir)

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
    # Model
    # ======================================================

    model = MNISTCNN(
        num_classes=config.NUM_CLASSES
    ).to(device)

    model_path = (
        config.MODEL_DIR /
        "fashion_mnist_cnn.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    print(f"[OK] Fashion-MNIST CNN loaded from {model_path}")

    # ======================================================
    # Clean accuracy
    # ======================================================

    clean_accuracy = evaluate_clean(
        model,
        test_loader,
        device
    )

    print(f"[OK] Clean accuracy: {clean_accuracy:.2f}%")

    # ======================================================
    # Anomaly Detector (calibrated on Fashion-MNIST train set)
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
        f"[OK] Anomaly threshold: {threshold:.6f}"
    )

    # ======================================================
    # Defense components
    # ======================================================

    purifier = SelfPurifier(model)

    randomized = RandomizedDefense(
        model,
        num_samples=4,
        noise_std=0.03
    )

    gradient_defense = GradientDiversityDefense(
        model,
        consistency_threshold=0.75
    )

    combined = CombinedDefense(
        model=model,
        detector=detector,
        purifier=purifier,
        randomized_defense=randomized,
        gradient_defense=gradient_defense
    )

    print("[OK] Full combined defense initialized")

    # ======================================================
    # Collect all results
    # ======================================================

    all_results = {
        "dataset": "Fashion-MNIST",
        "device": str(device),
        "anomaly_threshold": threshold,
        "clean_accuracy": clean_accuracy,
        "attacks": {}
    }

    # --------------------------------------------------------
    # FGSM
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("FGSM + FULL DEFENSE")
    print("-" * 80)

    fgsm_epsilons = [0.05, 0.10, 0.20]

    for epsilon in fgsm_epsilons:

        print(f"  Running FGSM eps={epsilon:.2f} ...")

        results = evaluate_attack(
            model,
            combined,
            test_loader,
            device,
            "FGSM",
            epsilon
        )

        key = f"FGSM_eps{epsilon:.2f}"
        all_results["attacks"][key] = {
            "attack": "FGSM",
            "epsilon": epsilon,
            **results
        }

        print(
            f"  FGSM eps={epsilon:.2f} | "
            f"Attack Acc: {results['attack_accuracy']:.2f}% | "
            f"Defended Acc: {results['defended_accuracy']:.2f}% | "
            f"Detection Rate: {results['detection_rate']:.2f}%"
        )

    # --------------------------------------------------------
    # PGD
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("PGD + FULL DEFENSE")
    print("-" * 80)

    pgd_epsilons = [0.05, 0.10, 0.20]

    for epsilon in pgd_epsilons:

        print(f"  Running PGD eps={epsilon:.2f} ...")

        results = evaluate_attack(
            model,
            combined,
            test_loader,
            device,
            "PGD",
            epsilon
        )

        key = f"PGD_eps{epsilon:.2f}"
        all_results["attacks"][key] = {
            "attack": "PGD",
            "epsilon": epsilon,
            **results
        }

        print(
            f"  PGD eps={epsilon:.2f} | "
            f"Attack Acc: {results['attack_accuracy']:.2f}% | "
            f"Defended Acc: {results['defended_accuracy']:.2f}% | "
            f"Detection Rate: {results['detection_rate']:.2f}%"
        )

    # --------------------------------------------------------
    # EOT-PGD
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("EOT-PGD (eps=0.20, steps=10, eot_samples=4) + FULL DEFENSE")
    print("-" * 80)

    eot_epsilon = 0.20

    eot_results = evaluate_eot(
        model=model,
        defense=combined,
        loader=test_loader,
        device=device,
        epsilon=eot_epsilon,
        attack_steps=10,
        eot_samples=4
    )

    all_results["attacks"]["EOT_PGD_eps0.20"] = {
        "attack": "EOT-PGD",
        "epsilon": eot_epsilon,
        "attack_steps": 10,
        "eot_samples": 4,
        **eot_results
    }

    print(
        f"  EOT-PGD eps={eot_epsilon:.2f} | "
        f"Attack Acc: {eot_results['attack_accuracy']:.2f}% | "
        f"Defended Acc: {eot_results['defended_accuracy']:.2f}% | "
        f"Detection Rate: {eot_results['detection_rate']:.2f}%"
    )

    # ======================================================
    # Save results
    # ======================================================

    results_dir = (
        PROJECT_ROOT /
        "experiments" /
        "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        results_dir /
        "fashion_mnist_full_defense.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print()
    print("=" * 80)
    print("FASHION-MNIST EVALUATION COMPLETE")
    print("=" * 80)

    print(f"[OK] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
