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


def evaluate_eot(
    model,
    defense,
    loader,
    device,
    epsilon=0.20,
    attack_steps=10,
    eot_samples=4
):

    total = 0

    attack_correct = 0
    defended_correct = 0
    detected = 0

    for batch_index, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        # --------------------------------------------------
        # Adaptive EOT-PGD attack
        # --------------------------------------------------

        adversarial_images = eot_pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=epsilon,
            alpha=0.01,
            steps=attack_steps,
            eot_samples=eot_samples
        )

        # --------------------------------------------------
        # Attack accuracy
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Full defense
        # --------------------------------------------------

        result = defense.predict(
            adversarial_images
        )

        defended_predictions = (
            result["predictions"]
        )

        anomalous = (
            result["anomalous"]
        )

        defended_correct += (
            defended_predictions == labels
        ).sum().item()

        detected += (
            anomalous.sum().item()
        )

        total += labels.size(0)

        # Progress
        if (batch_index + 1) % 20 == 0:

            print(
                f"Processed "
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


def main():

    print("=" * 80)
    print("ADAPTIVE EOT-PGD DEFENSE EVALUATION")
    print("=" * 80)

    device = config.DEVICE

    print(
        f"Device: {device}"
    )

    # ------------------------------------------------------
    # Dataset
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
    # Model
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

    print(
        "✓ Baseline CNN loaded"
    )

    # ------------------------------------------------------
    # Detector
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Defense components
    # ------------------------------------------------------

    purifier = SelfPurifier(
        model
    )

    randomized = RandomizedDefense(
        model,
        num_samples=4,
        noise_std=0.03
    )

    gradient_defense = (
        GradientDiversityDefense(
            model,
            consistency_threshold=0.75
        )
    )

    combined = CombinedDefense(
        model=model,
        detector=detector,
        purifier=purifier,
        randomized_defense=randomized,
        gradient_defense=gradient_defense
    )

    print(
        "✓ Full defense initialized"
    )

    # ------------------------------------------------------
    # EOT evaluation
    # ------------------------------------------------------

    epsilon = 0.20

    print()
    print("-" * 80)
    print("EOT-PGD RESULTS")
    print("-" * 80)

    results = evaluate_eot(
        model=model,
        defense=combined,
        loader=test_loader,
        device=device,
        epsilon=epsilon,
        attack_steps=10,
        eot_samples=4
    )

    print()
    print(
        f"EOT-PGD ε={epsilon:.2f}"
    )

    print(
        f"Attack Accuracy: "
        f"{results['attack_accuracy']:.2f}%"
    )

    print(
        f"Defended Accuracy: "
        f"{results['defended_accuracy']:.2f}%"
    )

    print(
        f"Detection Rate: "
        f"{results['detection_rate']:.2f}%"
    )

    # ------------------------------------------------------
    # Save result
    # ------------------------------------------------------

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
        "eot_results.json"
    )

    output = {
        "attack": "EOT-PGD",
        "epsilon": epsilon,
        "attack_steps": 10,
        "eot_samples": 4,
        "anomaly_threshold": threshold,
        **results
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print()
    print(
        f"✓ Results saved to:\n"
        f"{output_file}"
    )

    print()
    print("=" * 80)
    print("EOT EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()