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

from src.detection.anomaly_detector import DynamicAnomalyDetector
from src.purification.purifier import SelfPurifier

from src.defenses.randomized_defense import RandomizedDefense
from src.defenses.gradient_diversity import GradientDiversityDefense
from src.defenses.combined_defense import CombinedDefense


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
    "cw_defense_results.json"
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
# LOAD IDX IMAGES
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
        dtype=torch.uint8
    )

    images = images.reshape(
        count,
        rows,
        cols
    )

    images = images.float() / 255.0

    return images.unsqueeze(1)


# ============================================================
# LOAD IDX LABELS
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
# LOAD MODEL
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
# CALIBRATE ANOMALY DETECTOR
# ============================================================

def calibrate_detector(model, clean_loader):

    print()
    print("Calibrating anomaly detector...")

    detector = DynamicAnomalyDetector(
        model=model,
        threshold_percentile=95.0
    )

    threshold = detector.calibrate(
        clean_loader=clean_loader,
        device=DEVICE
    )

    print(
        f"✓ Anomaly threshold: {threshold:.6f}"
    )

    return detector, threshold


# ============================================================
# CREATE DEFENSES
# ============================================================

def create_defenses(model, detector):

    print()
    print("Creating defense modules...")

    purifier = SelfPurifier(
        model=model
    )

    randomized_defense = RandomizedDefense(
        model=model
    )

    gradient_defense = GradientDiversityDefense(
        model=model
    )

    combined_defense = CombinedDefense(
        model=model,
        detector=detector,
        purifier=purifier,
        randomized_defense=randomized_defense,
        gradient_defense=gradient_defense
    )

    print("✓ Purifier ready")
    print("✓ Randomized defense ready")
    print("✓ Gradient diversity ready")
    print("✓ Combined defense ready")

    return (
        purifier,
        randomized_defense,
        gradient_defense,
        combined_defense
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("C&W DEFENSE EVALUATION")
    print("=" * 80)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("Loading existing MNIST test data...")

    images = load_idx_images(
        TEST_IMAGES_PATH
    )

    labels = load_idx_labels(
        TEST_LABELS_PATH
    )

    print(
        f"✓ Test images: {len(images)}"
    )

    print(
        f"✓ Test labels: {len(labels)}"
    )

    dataset = TensorDataset(
        images,
        labels
    )

    # --------------------------------------------------------
    # Calibration loader
    # --------------------------------------------------------

    calibration_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    # --------------------------------------------------------
    # Evaluation loader
    # --------------------------------------------------------

    evaluation_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading baseline CNN...")

    model = load_model()

    print(
        "✓ Baseline CNN loaded"
    )

    # ========================================================
    # CALIBRATE DETECTOR
    # ========================================================

    detector, threshold = calibrate_detector(
        model,
        calibration_loader
    )

    # ========================================================
    # CREATE DEFENSES
    # ========================================================

    (
        purifier,
        randomized_defense,
        gradient_defense,
        combined_defense
    ) = create_defenses(
        model,
        detector
    )

    # ========================================================
    # METRIC STORAGE
    # ========================================================

    total = 0

    baseline_correct = 0
    purification_correct = 0
    randomized_correct = 0
    gradient_correct = 0
    combined_correct = 0

    detected_count = 0

    # ========================================================
    # LIMIT INITIAL EXPERIMENT
    # ========================================================

    max_batches = 20

    print()
    print("=" * 80)
    print("GENERATING C&W EXAMPLES AND TESTING DEFENSES")
    print("=" * 80)

    # ========================================================
    # LOOP
    # ========================================================

    for batch_index, (
        batch_images,
        batch_labels
    ) in enumerate(evaluation_loader):

        if batch_index >= max_batches:
            break

        batch_images = batch_images.to(
            DEVICE
        )

        batch_labels = batch_labels.to(
            DEVICE
        )

        # ====================================================
        # C&W
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
        # BASELINE
        # ====================================================

        with torch.no_grad():

            baseline_logits = model(
                adversarial_images
            )

            baseline_predictions = (
                baseline_logits.argmax(
                    dim=1
                )
            )

        baseline_correct += (
            baseline_predictions ==
            batch_labels
        ).sum().item()

        # ====================================================
        # ANOMALY DETECTION
        # ====================================================

        detection = detector.detect(
            adversarial_images
        )

        anomalous = detection[
            "is_anomalous"
        ]

        detected_count += (
            anomalous.sum().item()
        )

        # ====================================================
        # PURIFICATION
        # ====================================================

        purified_images = (
            adversarial_images.clone()
        )

        if anomalous.any():

            purified = purifier.purify(
                adversarial_images[
                    anomalous
                ]
            )

            purified_images[
                anomalous
            ] = purified

        with torch.no_grad():

            purification_logits = model(
                purified_images
            )

            purification_predictions = (
                purification_logits.argmax(
                    dim=1
                )
            )

        purification_correct += (
            purification_predictions ==
            batch_labels
        ).sum().item()

        # ====================================================
        # RANDOMIZED DEFENSE
        # ====================================================

        randomized_result = (
            randomized_defense.predict(
                purified_images
            )
        )

        randomized_predictions = (
            randomized_result[
                "predictions"
            ]
        )

        randomized_correct += (
            randomized_predictions ==
            batch_labels
        ).sum().item()

        # ====================================================
        # GRADIENT DIVERSITY
        # ====================================================

        gradient_result = (
            gradient_defense.predict(
                purified_images
            )
        )

        gradient_predictions = (
            gradient_result[
                "final_predictions"
            ]
        )

        gradient_correct += (
            gradient_predictions ==
            batch_labels
        ).sum().item()

        # ====================================================
        # COMBINED DEFENSE
        # ====================================================

        combined_result = (
            combined_defense.predict(
                adversarial_images
            )
        )

        combined_predictions = (
            combined_result[
                "predictions"
            ]
        )

        combined_correct += (
            combined_predictions ==
            batch_labels
        ).sum().item()

        # ====================================================
        # COUNT
        # ====================================================

        total += batch_images.size(0)

        print(
            f"Batch "
            f"{batch_index + 1:02d}/{max_batches} "
            f"completed"
        )

    # ========================================================
    # METRICS
    # ========================================================

    baseline_accuracy = (
        baseline_correct /
        total
    ) * 100

    purification_accuracy = (
        purification_correct /
        total
    ) * 100

    randomized_accuracy = (
        randomized_correct /
        total
    ) * 100

    gradient_accuracy = (
        gradient_correct /
        total
    ) * 100

    combined_accuracy = (
        combined_correct /
        total
    ) * 100

    detection_rate = (
        detected_count /
        total
    ) * 100

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "attack": "Carlini-Wagner",

        "dataset": "MNIST",

        "device": DEVICE,

        "samples_evaluated": total,

        "anomaly_threshold": threshold,

        "baseline_accuracy":
            baseline_accuracy,

        "purification_accuracy":
            purification_accuracy,

        "randomized_accuracy":
            randomized_accuracy,

        "gradient_diversity_accuracy":
            gradient_accuracy,

        "combined_defense_accuracy":
            combined_accuracy,

        "detection_rate":
            detection_rate,

        "parameters": {

            "c": 1.0,

            "kappa": 0.0,

            "learning_rate": 0.01,

            "steps": 100,

            "batch_size": 32
        }
    }

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
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print("C&W DEFENSE RESULTS")
    print("=" * 80)

    print(
        f"Samples Evaluated       : "
        f"{total}"
    )

    print(
        f"Anomaly Threshold       : "
        f"{threshold:.6f}"
    )

    print()

    print(
        f"Baseline                : "
        f"{baseline_accuracy:.2f}%"
    )

    print(
        f"Purification            : "
        f"{purification_accuracy:.2f}%"
    )

    print(
        f"Randomized              : "
        f"{randomized_accuracy:.2f}%"
    )

    print(
        f"Gradient Diversity      : "
        f"{gradient_accuracy:.2f}%"
    )

    print(
        f"Combined                : "
        f"{combined_accuracy:.2f}%"
    )

    print()

    print(
        f"Detection Rate          : "
        f"{detection_rate:.2f}%"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print("=" * 80)
    print("C&W DEFENSE EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()