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
from src.attacks.cw import carlini_wagner_attack

try:
    from src.attacks.eot_pgd import eot_pgd_attack
    EOT_AVAILABLE = True
except ImportError:
    EOT_AVAILABLE = False

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
    "graddiv_defense_results.json"
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
# LOAD MODEL
# ============================================================

def load_model():

    model = MNISTCNN().to(
        DEVICE
    )

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

    return model


# ============================================================
# CREATE DEFENSE PIPELINE
# ============================================================

def create_defenses(model, calibration_loader):

    print()
    print("Calibrating anomaly detector...")

    detector = DynamicAnomalyDetector(
        model=model,
        threshold_percentile=95.0
    )

    threshold = detector.calibrate(
        clean_loader=calibration_loader,
        device=DEVICE
    )

    print(
        f"✓ Anomaly threshold: {threshold:.6f}"
    )

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
        detector,
        purifier,
        randomized_defense,
        gradient_defense,
        combined_defense,
        threshold
    )


# ============================================================
# APPLY DEFENSES
# ============================================================

def evaluate_defenses(
    model,
    detector,
    purifier,
    randomized_defense,
    gradient_defense,
    combined_defense,
    images,
    labels
):

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            images
        )

        baseline_predictions = (
            logits.argmax(
                dim=1
            )
        )

    baseline_accuracy = (
        baseline_predictions ==
        labels
    ).float().mean().item() * 100

    # --------------------------------------------------------
    # Anomaly Detection
    # --------------------------------------------------------

    detection = detector.detect(
        images
    )

    anomalous = detection[
        "is_anomalous"
    ]

    detection_rate = (
        anomalous.float().mean().item()
        * 100
    )

    # --------------------------------------------------------
    # Purification
    # --------------------------------------------------------

    purified_images = images.clone()

    if anomalous.any():

        purified = purifier.purify(
            images[anomalous]
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

    purification_accuracy = (
        purification_predictions ==
        labels
    ).float().mean().item() * 100

    # --------------------------------------------------------
    # Randomized
    # --------------------------------------------------------

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

    randomized_accuracy = (
        randomized_predictions ==
        labels
    ).float().mean().item() * 100

    # --------------------------------------------------------
    # Gradient Diversity
    # --------------------------------------------------------

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

    gradient_accuracy = (
        gradient_predictions ==
        labels
    ).float().mean().item() * 100

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    combined_result = (
        combined_defense.predict(
            images
        )
    )

    combined_predictions = (
        combined_result[
            "predictions"
        ]
    )

    combined_accuracy = (
        combined_predictions ==
        labels
    ).float().mean().item() * 100

    return {

        "baseline": baseline_accuracy,

        "purification": purification_accuracy,

        "randomized": randomized_accuracy,

        "gradient_diversity": gradient_accuracy,

        "combined": combined_accuracy,

        "detection_rate": detection_rate
    }


# ============================================================
# ATTACK EVALUATION
# ============================================================

def run_attack(
    model,
    loader,
    attack_name,
    attack_function,
    attack_kwargs,
    defenses,
    max_batches=20
):

    (
        detector,
        purifier,
        randomized_defense,
        gradient_defense,
        combined_defense,
        threshold
    ) = defenses

    totals = {

        "baseline": 0.0,

        "purification": 0.0,

        "randomized": 0.0,

        "gradient_diversity": 0.0,

        "combined": 0.0,

        "detection_rate": 0.0
    }

    total_batches = 0

    total_samples = 0

    print()
    print(
        "-" * 80
    )

    print(
        attack_name
    )

    print(
        "-" * 80
    )

    for batch_index, (
        images,
        labels
    ) in enumerate(loader):

        if batch_index >= max_batches:

            break

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Generate attack
        # ----------------------------------------------------

        adversarial_images = attack_function(
            model,
            images,
            labels,
            **attack_kwargs
        )

        # ----------------------------------------------------
        # Evaluate defenses
        # ----------------------------------------------------

        metrics = evaluate_defenses(
            model=model,
            detector=detector,
            purifier=purifier,
            randomized_defense=randomized_defense,
            gradient_defense=gradient_defense,
            combined_defense=combined_defense,
            images=adversarial_images,
            labels=labels
        )

        batch_size = labels.size(0)

        for key in totals:

            totals[key] += (
                metrics[key] *
                batch_size
            )

        total_samples += batch_size

        total_batches += 1

        print(
            f"{attack_name}: "
            f"batch "
            f"{batch_index + 1}/{max_batches}"
        )

    # --------------------------------------------------------
    # Average
    # --------------------------------------------------------

    final_metrics = {}

    for key in totals:

        final_metrics[key] = (
            totals[key] /
            total_samples
        )

    print()

    print(
        f"Baseline           : "
        f"{final_metrics['baseline']:.2f}%"
    )

    print(
        f"Purification       : "
        f"{final_metrics['purification']:.2f}%"
    )

    print(
        f"Randomized         : "
        f"{final_metrics['randomized']:.2f}%"
    )

    print(
        f"Gradient Diversity : "
        f"{final_metrics['gradient_diversity']:.2f}%"
    )

    print(
        f"Combined           : "
        f"{final_metrics['combined']:.2f}%"
    )

    print(
        f"Detection Rate     : "
        f"{final_metrics['detection_rate']:.2f}%"
    )

    return final_metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GRADDIV MODEL + DEFENSE EVALUATION")
    print("=" * 80)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"GradDiv model not found:\n{MODEL_PATH}"
        )

    # ========================================================
    # DATA
    # ========================================================

    print()
    print(
        "Loading existing MNIST test data..."
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

    calibration_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    evaluation_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    print(
        f"✓ Test samples: {len(dataset)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    print(
        "✓ GradDiv CNN loaded"
    )

    # ========================================================
    # DEFENSES
    # ========================================================

    defenses = create_defenses(
        model,
        calibration_loader
    )

    (
        detector,
        purifier,
        randomized_defense,
        gradient_defense,
        combined_defense,
        threshold
    ) = defenses

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "model": "GradDiv CNN",

        "dataset": "MNIST",

        "device": DEVICE,

        "anomaly_threshold": threshold,

        "fgsm": {},

        "pgd": {},

        "cw": {},

        "eot_pgd": {}
    }

    # ========================================================
    # FGSM
    # ========================================================

    for epsilon in [0.10, 0.20]:

        metrics = run_attack(
            model=model,
            loader=evaluation_loader,
            attack_name=f"FGSM epsilon={epsilon:.2f}",
            attack_function=fgsm_attack,
            attack_kwargs={
                "epsilon": epsilon
            },
            defenses=defenses
        )

        results["fgsm"][
            str(epsilon)
        ] = metrics

    # ========================================================
    # PGD
    # ========================================================

    for epsilon in [0.10, 0.20]:

        metrics = run_attack(
            model=model,
            loader=evaluation_loader,
            attack_name=f"PGD epsilon={epsilon:.2f}",
            attack_function=pgd_attack,
            attack_kwargs={
                "epsilon": epsilon,
                "alpha": 0.01,
                "steps": 10
            },
            defenses=defenses
        )

        results["pgd"][
            str(epsilon)
        ] = metrics

    # ========================================================
    # C&W
    # ========================================================

    metrics = run_attack(
        model=model,
        loader=evaluation_loader,
        attack_name="C&W",
        attack_function=carlini_wagner_attack,
        attack_kwargs={
            "targeted": False,
            "c": 1.0,
            "kappa": 0.0,
            "lr": 0.01,
            "steps": 100,
            "device": DEVICE
        },
        defenses=defenses
    )

    results["cw"] = metrics

    # ========================================================
    # EOT-PGD
    # ========================================================

    if EOT_AVAILABLE:

        try:

            metrics = run_attack(
                model=model,
                loader=evaluation_loader,
                attack_name="EOT-PGD",
                attack_function=eot_pgd_attack,
                attack_kwargs={
                    "epsilon": 0.20,
                    "alpha": 0.01,
                    "steps": 10
                },
                defenses=defenses
            )

            results["eot_pgd"] = metrics

        except TypeError as error:

            print()
            print(
                "⚠ EOT-PGD skipped because "
                "its current function signature differs."
            )

            print(
                error
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
    # COMPLETE
    # ========================================================

    print()
    print("=" * 80)
    print("GRADDIV DEFENSE EVALUATION COMPLETE")
    print("=" * 80)

    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print("=" * 80)


if __name__ == "__main__":
    main()