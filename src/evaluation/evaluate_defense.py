from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT)
)

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

from src.defenses.defense_pipeline import (
    AnomalyDefensePipeline
)

import config


def calculate_accuracy(
    predictions,
    labels
):

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.size(0)

    return 100.0 * correct / total


def get_clean_accuracy(
    model,
    loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


def calibrate_detector(
    detector,
    loader,
    device
):

    threshold = detector.calibrate(
        loader,
        device
    )

    print(
        f"Calibrated anomaly threshold: "
        f"{threshold:.6f}"
    )

    return threshold


def evaluate_defense(
    model,
    detector,
    purifier,
    loader,
    device,
    attack_name,
    epsilon
):

    model.eval()

    total = 0

    clean_attack_correct = 0
    defended_correct = 0

    detected_anomalies = 0

    true_adversarial = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # --------------------------------------------------
        # Generate adversarial images
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
                "Unknown attack."
            )

        # --------------------------------------------------
        # Accuracy before defense
        # --------------------------------------------------

        with torch.no_grad():

            attack_outputs = model(
                adversarial_images
            )

            attack_predictions = (
                attack_outputs.argmax(
                    dim=1
                )
            )

        clean_attack_correct += (
            attack_predictions == labels
        ).sum().item()

        # --------------------------------------------------
        # Detection
        # --------------------------------------------------

        detection = detector.detect(
            adversarial_images
        )

        anomaly_flags = detection[
            "is_anomalous"
        ]

        detected_anomalies += (
            anomaly_flags.sum().item()
        )

        true_adversarial += (
            labels.size(0)
        )

        # --------------------------------------------------
        # Purification
        # --------------------------------------------------

        defended_images = (
            adversarial_images.clone()
        )

        if anomaly_flags.any():

            purified = purifier.purify(
                adversarial_images[
                    anomaly_flags
                ]
            )

            defended_images[
                anomaly_flags
            ] = purified

        # --------------------------------------------------
        # Final prediction
        # --------------------------------------------------

        with torch.no_grad():

            defended_outputs = model(
                defended_images
            )

            defended_predictions = (
                defended_outputs.argmax(
                    dim=1
                )
            )

        defended_correct += (
            defended_predictions == labels
        ).sum().item()

        total += labels.size(0)

    attack_accuracy = (
        100.0 *
        clean_attack_correct /
        total
    )

    defended_accuracy = (
        100.0 *
        defended_correct /
        total
    )

    detection_rate = (
        100.0 *
        detected_anomalies /
        true_adversarial
    )

    return {
        "attack_accuracy":
            attack_accuracy,

        "defended_accuracy":
            defended_accuracy,

        "detection_rate":
            detection_rate
    }


def main():

    print("=" * 75)
    print("ANOMALY DETECTION + SELF-PURIFICATION EVALUATION")
    print("=" * 75)

    device = config.DEVICE

    print(
        f"Device: {device}"
    )

    # ------------------------------------------------------
    # Load MNIST
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Load CNN
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Baseline
    # ------------------------------------------------------

    clean_accuracy = get_clean_accuracy(
        model,
        test_loader,
        device
    )

    print()
    print(
        f"Clean Accuracy: "
        f"{clean_accuracy:.2f}%"
    )

    # ------------------------------------------------------
    # Detector
    # ------------------------------------------------------

    detector = DynamicAnomalyDetector(
        model,
        threshold_percentile=95.0
    )

    calibrate_detector(
        detector,
        train_loader,
        device
    )

    # ------------------------------------------------------
    # Purifier
    # ------------------------------------------------------

    purifier = SelfPurifier(
        model
    )

    # ------------------------------------------------------
    # FGSM evaluation
    # ------------------------------------------------------

    print()
    print("-" * 75)
    print("FGSM DEFENSE RESULTS")
    print("-" * 75)

    for epsilon in [
        0.05,
        0.10,
        0.20
    ]:

        results = evaluate_defense(
            model,
            detector,
            purifier,
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

    # ------------------------------------------------------
    # PGD evaluation
    # ------------------------------------------------------

    print()
    print("-" * 75)
    print("PGD DEFENSE RESULTS")
    print("-" * 75)

    for epsilon in [
        0.05,
        0.10,
        0.20
    ]:

        results = evaluate_defense(
            model,
            detector,
            purifier,
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
    print("=" * 75)
    print("DEFENSE EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()