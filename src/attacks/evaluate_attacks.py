from pathlib import Path
import sys

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

    # ---------------------------------------------------------
    # Clean accuracy
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

    print()
    print("=" * 70)
    print("ATTACK EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()