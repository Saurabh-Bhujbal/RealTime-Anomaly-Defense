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

from src.detection.anomaly_detector import DynamicAnomalyDetector
from src.purification.purifier import SelfPurifier
from src.defenses.randomized_defense import RandomizedDefense
from src.defenses.gradient_diversity import GradientDiversityDefense
from src.defenses.combined_defense import CombinedDefense

import config


def get_predictions(model, images):
    with torch.no_grad():
        return model(images).argmax(dim=1)


def evaluate_method(
    model,
    method,
    attack_name,
    epsilon,
    loader,
    device,
    purifier,
    randomized,
    gradient,
    combined
):
    """
    Evaluate one defense method against one attack.
    """

    total = 0
    correct = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # --------------------------------------------------
        # Generate adversarial samples
        # --------------------------------------------------

        if attack_name == "FGSM":

            adv = fgsm_attack(
                model,
                images,
                labels,
                epsilon=epsilon
            )

        elif attack_name == "PGD":

            adv = pgd_attack(
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
        # BASELINE
        # --------------------------------------------------

        if method == "baseline":

            predictions = get_predictions(
                model,
                adv
            )

        # --------------------------------------------------
        # PURIFICATION ONLY
        # --------------------------------------------------

        elif method == "purification":

            purified = purifier.purify(
                adv
            )

            predictions = get_predictions(
                model,
                purified
            )

        # --------------------------------------------------
        # RANDOMIZED DEFENSE ONLY
        # --------------------------------------------------

        elif method == "randomized":

            result = randomized.predict(
                adv
            )

            predictions = result[
                "predictions"
            ]

        # --------------------------------------------------
        # GRADIENT DIVERSITY ONLY
        # --------------------------------------------------

        elif method == "gradient":

            result = gradient.predict(
                adv
            )

            predictions = result[
                "final_predictions"
            ]

        # --------------------------------------------------
        # COMBINED DEFENSE
        # --------------------------------------------------

        elif method == "combined":

            result = combined.predict(
                adv
            )

            predictions = result[
                "predictions"
            ]

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

        # --------------------------------------------------
        # Accuracy
        # --------------------------------------------------

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return 100.0 * correct / total


def main():

    print("=" * 80)
    print("DEFENSE ABLATION STUDY")
    print("=" * 80)

    device = config.DEVICE

    print(
        f"Device: {device}"
    )

    # ======================================================
    # DATA
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

    print(
        "✓ Baseline CNN loaded"
    )

    # ======================================================
    # DETECTOR
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
    # DEFENSE COMPONENTS
    # ======================================================

    purifier = SelfPurifier(
        model
    )

    randomized = RandomizedDefense(
        model,
        num_samples=4,
        noise_std=0.03
    )

    gradient = GradientDiversityDefense(
        model,
        consistency_threshold=0.75
    )

    combined = CombinedDefense(
        model=model,
        detector=detector,
        purifier=purifier,
        randomized_defense=randomized,
        gradient_defense=gradient
    )

    print(
        "✓ Purifier ready"
    )

    print(
        "✓ Randomized defense ready"
    )

    print(
        "✓ Gradient diversity ready"
    )

    print(
        "✓ Combined defense ready"
    )

    # ======================================================
    # METHODS
    # ======================================================

    methods = [
        "baseline",
        "purification",
        "randomized",
        "gradient",
        "combined"
    ]

    method_labels = {
        "baseline": "Baseline",
        "purification": "Purification",
        "randomized": "Randomized",
        "gradient": "Gradient Diversity",
        "combined": "Combined"
    }

    results = {}

    # ======================================================
    # ABLATION
    # ======================================================

    for attack_name in [
        "FGSM",
        "PGD"
    ]:

        results[attack_name] = {}

        print()
        print("-" * 80)
        print(
            f"{attack_name} ABLATION"
        )
        print("-" * 80)

        for epsilon in [
            0.10,
            0.20
        ]:

            epsilon_key = (
                f"{epsilon:.2f}"
            )

            results[attack_name][
                epsilon_key
            ] = {}

            print()
            print(
                f"{attack_name} "
                f"epsilon={epsilon:.2f}"
            )

            for method in methods:

                accuracy = evaluate_method(
                    model=model,
                    method=method,
                    attack_name=attack_name,
                    epsilon=epsilon,
                    loader=test_loader,
                    device=device,
                    purifier=purifier,
                    randomized=randomized,
                    gradient=gradient,
                    combined=combined
                )

                results[
                    attack_name
                ][
                    epsilon_key
                ][method] = round(
                    accuracy,
                    2
                )

                print(
                    f"{method_labels[method]:20s}: "
                    f"{accuracy:.2f}%"
                )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    results[
        "anomaly_threshold"
    ] = round(
        threshold,
        6
    )

    results_dir = (
        PROJECT_ROOT /
        "experiments" /
        "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        results_dir /
        "ablation_results.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)

    print(
        f"Results saved to:\n{output}"
    )


if __name__ == "__main__":
    main()