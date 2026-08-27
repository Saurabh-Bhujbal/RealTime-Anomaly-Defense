from pathlib import Path
import sys
import asyncio

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist
from src.data.preprocessing import MNISTDataset
from src.models.cnn import MNISTCNN
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

import config

# ─────────────────────────────────────────────────────────────────────────────
# DB seeding: writes results into the backend's benchmark_results table
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    # insert (not append): PROJECT_ROOT is already on sys.path and contains
    # a root-level app.py (the legacy Streamlit entry point). If BACKEND_DIR
    # were appended instead, Python would find that app.py first when
    # resolving `import app`, since PROJECT_ROOT was added to sys.path
    # earlier. Inserting at index 0 makes backend/app/ (the real package
    # with database.py, models.py, etc.) win the name lookup instead.
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import AsyncSessionLocal
from app.models import BenchmarkResult

# ─────────────────────────────────────────────────────────────────────────────
# CombinedDefense (Part 2 — defended-vs-baseline comparison)
# ─────────────────────────────────────────────────────────────────────────────

from src.detection.anomaly_detector import DynamicAnomalyDetector
from src.purification.purifier import SelfPurifier
from src.defenses.randomized_defense import RandomizedDefense
from src.defenses.gradient_diversity import GradientDiversityDefense
from src.defenses.combined_defense import CombinedDefense

# Limit how many test batches CombinedDefense evaluation runs over.
# CombinedDefense does ~10+ forward passes per batch (detector + purifier +
# randomized + gradient), so a full 10,000-image pass across 6 epsilon/attack
# combos is slow on CPU. Start small, then raise/remove once confirmed working.
COMBINED_DEFENSE_MAX_BATCHES = 20


def evaluate_clean(model, loader, device):

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


def evaluate_attack(
    model,
    loader,
    device,
    attack_function,
    **attack_parameters
):

    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        adversarial_images = attack_function(
            model,
            images,
            labels,
            **attack_parameters
        )

        with torch.no_grad():

            outputs = model(
                adversarial_images
            )

            predictions = outputs.argmax(
                dim=1
            )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# CombinedDefense builder + evaluator (Part 2)
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_defense(model):
    """
    Builds a CombinedDefense instance using the same configuration as
    backend/app/services/defense_service.py, so benchmark numbers match
    what the live API actually serves.
    """

    detector = DynamicAnomalyDetector(model, threshold_percentile=95.0)
    detector.threshold = 0.037860  # pre-calibrated value, matches defense_service.py

    purifier = SelfPurifier(model)

    randomized = RandomizedDefense(model, num_samples=4, noise_std=0.03)

    gradient = GradientDiversityDefense(model, consistency_threshold=0.75)

    return CombinedDefense(
        model=model,
        detector=detector,
        purifier=purifier,
        randomized_defense=randomized,
        gradient_defense=gradient,
    )


def evaluate_combined_defense(
    defense,
    loader,
    device,
    attack_function=None,
    max_batches=None,
    **attack_parameters
):
    """
    Runs CombinedDefense.predict() on clean images (attack_function=None) or
    attacked images (attack_function=fgsm_attack / pgd_attack).

    Returns (accuracy, detection_rate) as percentages.
    detection_rate = % of images the anomaly detector flagged as suspicious.
    """

    correct = 0
    total = 0
    anomalous_count = 0

    for batch_idx, (images, labels) in enumerate(loader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        if attack_function is not None:
            images = attack_function(
                defense.model,
                images,
                labels,
                **attack_parameters
            )

        result = defense.predict(images)

        correct += (
            result["predictions"] == labels
        ).sum().item()

        anomalous_count += result["anomalous"].sum().item()

        total += labels.size(0)

    accuracy = 100.0 * correct / total
    detection_rate = 100.0 * anomalous_count / total

    return accuracy, detection_rate


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark result persistence
# ─────────────────────────────────────────────────────────────────────────────

# ── Pending rows collected during the evaluation run ─────────────────────────
_pending_rows: list[dict] = []


def save_benchmark_result(
    model_name,
    defense_type,
    attack_type,
    epsilon,
    accuracy,
    detection_rate=None
):
    """
    Queue one benchmark result for bulk insertion at the end of main().
    Using a single async session (flush_all_to_db) avoids the asyncpg
    connection-pool crash that occurs when asyncio.run() is called
    repeatedly in the same process.
    """
    _pending_rows.append(dict(
        model_name=model_name,
        defense_type=defense_type,
        attack_type=attack_type,
        epsilon=epsilon,
        accuracy=accuracy,
        detection_rate=detection_rate,
    ))
    print(
        f"  -> queued: "
        f"defense={defense_type} attack={attack_type} "
        f"eps={epsilon:.2f} acc={accuracy:.2f}%"
    )


async def _flush_all_to_db():
    """Insert all queued rows in one async session, then dispose the engine."""
    from app.database import engine
    async with AsyncSessionLocal() as session:
        for row in _pending_rows:
            session.add(BenchmarkResult(**row))
        await session.commit()
    await engine.dispose()
    print(f"\n  -> {len(_pending_rows)} rows saved to benchmark_results.")


def generate_visualization(
    model,
    images,
    labels,
    device
):

    model.eval()

    images = images.to(device)
    labels = labels.to(device)

    # FGSM
    fgsm_images = fgsm_attack(
        model,
        images,
        labels,
        epsilon=0.2
    )

    # PGD
    pgd_images = pgd_attack(
        model,
        images,
        labels,
        epsilon=0.2,
        alpha=0.01,
        steps=10
    )

    with torch.no_grad():

        clean_pred = model(
            images
        ).argmax(dim=1)

        fgsm_pred = model(
            fgsm_images
        ).argmax(dim=1)

        pgd_pred = model(
            pgd_images
        ).argmax(dim=1)

    output_dir = (
        config.FIGURES_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 8)
    )

    for i in range(5):

        # Clean
        plt.subplot(3, 5, i + 1)

        plt.imshow(
            images[i].cpu().squeeze(),
            cmap="gray"
        )

        plt.title(
            f"Clean\nTrue:{labels[i].item()}\n"
            f"Pred:{clean_pred[i].item()}"
        )

        plt.axis("off")

        # FGSM
        plt.subplot(
            3,
            5,
            i + 6
        )

        plt.imshow(
            fgsm_images[i].cpu().squeeze(),
            cmap="gray"
        )

        plt.title(
            f"FGSM\nPred:{fgsm_pred[i].item()}"
        )

        plt.axis("off")

        # PGD
        plt.subplot(
            3,
            5,
            i + 11
        )

        plt.imshow(
            pgd_images[i].cpu().squeeze(),
            cmap="gray"
        )

        plt.title(
            f"PGD\nPred:{pgd_pred[i].item()}"
        )

        plt.axis("off")

    plt.tight_layout()

    output_path = (
        output_dir /
        "clean_fgsm_pgd_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Visualization saved to:\n"
        f"{output_path}"
    )


def main():

    device = config.DEVICE

    print("=" * 70)
    print("ADVERSARIAL ATTACK EVALUATION")
    print("=" * 70)

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    _, _, test_images, test_labels = load_mnist(
        config.DATA_DIR
    )

    test_dataset = MNISTDataset(
        test_images,
        test_labels
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # ---------------------------------------------------------
    # Load trained CNN
    # ---------------------------------------------------------

    model = MNISTCNN(
        num_classes=config.NUM_CLASSES
    ).to(device)

    model.load_state_dict(
        torch.load(
            config.MODEL_DIR / "baseline_cnn.pth",
            map_location=device
        )
    )

    model.eval()

    # Build CombinedDefense once, reused across all defended evaluations below
    combined_defense = build_combined_defense(model)

    # ---------------------------------------------------------
    # Clean accuracy — baseline (undefended)
    # ---------------------------------------------------------

    clean_accuracy = evaluate_clean(
        model,
        test_loader,
        device
    )

    print()
    print(
        f"Clean Accuracy: "
        f"{clean_accuracy:.2f}%"
    )

    save_benchmark_result(
        model_name="baseline_cnn",
        defense_type="none",
        attack_type="clean",
        epsilon=0.0,
        accuracy=clean_accuracy,
    )

    # ---------------------------------------------------------
    # Clean accuracy — defended (CombinedDefense)
    # ---------------------------------------------------------

    print()
    print("Running CombinedDefense on clean images...")

    defended_clean_accuracy, clean_detection_rate = evaluate_combined_defense(
        combined_defense,
        test_loader,
        device,
        attack_function=None,
        max_batches=COMBINED_DEFENSE_MAX_BATCHES,
    )

    print(
        f"Defended Clean Accuracy: {defended_clean_accuracy:.2f}% "
        f"| Detection Rate: {clean_detection_rate:.2f}%"
    )

    save_benchmark_result(
        model_name="baseline_cnn",
        defense_type="combined",
        attack_type="clean",
        epsilon=0.0,
        accuracy=defended_clean_accuracy,
        detection_rate=clean_detection_rate,
    )

    # ---------------------------------------------------------
    # FGSM
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("FGSM RESULTS")
    print("-" * 70)

    fgsm_results = {}

    for epsilon in [
        0.05,
        0.10,
        0.20
    ]:

        accuracy = evaluate_attack(
            model,
            test_loader,
            device,
            fgsm_attack,
            epsilon=epsilon
        )

        fgsm_results[epsilon] = accuracy

        print(
            f"Epsilon = {epsilon:.2f} "
            f"| Accuracy = {accuracy:.2f}%"
        )

        save_benchmark_result(
            model_name="baseline_cnn",
            defense_type="none",
            attack_type="fgsm",
            epsilon=epsilon,
            accuracy=accuracy,
        )

        # Defended pass — same attacked distribution, run through CombinedDefense
        defended_accuracy, detection_rate = evaluate_combined_defense(
            combined_defense,
            test_loader,
            device,
            fgsm_attack,
            max_batches=COMBINED_DEFENSE_MAX_BATCHES,
            epsilon=epsilon,
        )

        print(
            f"  [defended] Epsilon = {epsilon:.2f} "
            f"| Accuracy = {defended_accuracy:.2f}% "
            f"| Detection Rate = {detection_rate:.2f}%"
        )

        save_benchmark_result(
            model_name="baseline_cnn",
            defense_type="combined",
            attack_type="fgsm",
            epsilon=epsilon,
            accuracy=defended_accuracy,
            detection_rate=detection_rate,
        )

    # ---------------------------------------------------------
    # PGD
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("PGD RESULTS")
    print("-" * 70)

    pgd_results = {}

    for epsilon in [
        0.05,
        0.10,
        0.20
    ]:

        accuracy = evaluate_attack(
            model,
            test_loader,
            device,
            pgd_attack,
            epsilon=epsilon,
            alpha=0.01,
            steps=10
        )

        pgd_results[epsilon] = accuracy

        print(
            f"Epsilon = {epsilon:.2f} "
            f"| Accuracy = {accuracy:.2f}%"
        )

        save_benchmark_result(
            model_name="baseline_cnn",
            defense_type="none",
            attack_type="pgd",
            epsilon=epsilon,
            accuracy=accuracy,
        )

        # Defended pass
        defended_accuracy, detection_rate = evaluate_combined_defense(
            combined_defense,
            test_loader,
            device,
            pgd_attack,
            max_batches=COMBINED_DEFENSE_MAX_BATCHES,
            epsilon=epsilon,
            alpha=0.01,
            steps=10,
        )

        print(
            f"  [defended] Epsilon = {epsilon:.2f} "
            f"| Accuracy = {defended_accuracy:.2f}% "
            f"| Detection Rate = {detection_rate:.2f}%"
        )

        save_benchmark_result(
            model_name="baseline_cnn",
            defense_type="combined",
            attack_type="pgd",
            epsilon=epsilon,
            accuracy=defended_accuracy,
            detection_rate=detection_rate,
        )

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    images, labels = next(
        iter(test_loader)
    )

    generate_visualization(
        model,
        images[:5],
        labels[:5],
        device
    )

    # ── Bulk-save all queued benchmark rows in one async session ────────────
    print()
    print("Saving all benchmark results to database...")
    asyncio.run(_flush_all_to_db())

    print()
    print("=" * 70)
    print("ATTACK EVALUATION COMPLETE")
    print("="  * 70)


if __name__ == "__main__":
    main()
