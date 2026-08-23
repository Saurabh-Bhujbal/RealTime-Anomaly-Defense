import os
import sys
import json

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.cifar10_cnn import CIFAR10CNN


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cifar10_cnn.pth"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cifar10"
)

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "results",
    "cifar10_attack_results.json"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(1, 3, 1, 1).to(DEVICE)

STD = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(1, 3, 1, 1).to(DEVICE)


def clamp_normalized(x):

    lower = (0.0 - MEAN) / STD
    upper = (1.0 - MEAN) / STD

    return torch.max(
        torch.min(x, upper),
        lower
    )


def fgsm(model, images, labels, epsilon):

    x = images.clone().detach()
    x.requires_grad = True

    outputs = model(x)

    loss = F.cross_entropy(
        outputs,
        labels
    )

    model.zero_grad()

    loss.backward()

    perturbation = epsilon * x.grad.sign()

    adversarial = x + perturbation

    adversarial = clamp_normalized(
        adversarial
    )

    return adversarial.detach()


def pgd(
    model,
    images,
    labels,
    epsilon,
    alpha=None,
    steps=10
):

    if alpha is None:
        alpha = epsilon / 4

    original = images.clone().detach()

    noise = torch.empty_like(
        original
    ).uniform_(
        -epsilon,
        epsilon
    )

    adversarial = (
        original + noise
    )

    adversarial = clamp_normalized(
        adversarial
    )

    for _ in range(steps):

        adversarial.requires_grad = True

        outputs = model(adversarial)

        loss = F.cross_entropy(
            outputs,
            labels
        )

        model.zero_grad()

        loss.backward()

        gradient = adversarial.grad.sign()

        adversarial = (
            adversarial.detach()
            +
            alpha * gradient
        )

        delta = adversarial - original

        delta = torch.clamp(
            delta,
            -epsilon,
            epsilon
        )

        adversarial = (
            original + delta
        )

        adversarial = clamp_normalized(
            adversarial
        )

    return adversarial.detach()


def evaluate_attack(
    model,
    loader,
    attack,
    epsilon=None,
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

        if attack == "FGSM":

            adversarial = fgsm(
                model,
                images,
                labels,
                epsilon
            )

        elif attack == "PGD":

            adversarial = pgd(
                model,
                images,
                labels,
                epsilon
            )

        with torch.no_grad():

            outputs = model(
                adversarial
            )

            predictions = outputs.argmax(
                dim=1
            )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        print(
            f"{attack}: "
            f"batch {batch_index + 1}/{max_batches}"
        )

    return 100 * correct / total


def main():

    print("=" * 70)
    print("CIFAR-10 ADVERSARIAL ATTACK EVALUATION")
    print("=" * 70)

    print("Device:", DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    model = CIFAR10CNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.eval()

    results = {
        "model": "CIFAR-10 CNN",
        "clean_accuracy": None,
        "fgsm": {},
        "pgd": {}
    }

    # --------------------------------------------------
    # CLEAN
    # --------------------------------------------------

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(
                images
            ).argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    clean_accuracy = (
        100 * correct / total
    )

    results["clean_accuracy"] = clean_accuracy

    print()
    print(
        f"Clean Accuracy: "
        f"{clean_accuracy:.2f}%"
    )

    # --------------------------------------------------
    # FGSM
    # --------------------------------------------------

    for epsilon in [0.10, 0.20]:

        print()
        print(
            f"FGSM epsilon={epsilon:.2f}"
        )

        accuracy = evaluate_attack(
            model,
            loader,
            "FGSM",
            epsilon
        )

        results["fgsm"][
            str(epsilon)
        ] = accuracy

        print(
            f"Accuracy: {accuracy:.2f}%"
        )

    # --------------------------------------------------
    # PGD
    # --------------------------------------------------

    for epsilon in [0.10, 0.20]:

        print()
        print(
            f"PGD epsilon={epsilon:.2f}"
        )

        accuracy = evaluate_attack(
            model,
            loader,
            "PGD",
            epsilon
        )

        results["pgd"][
            str(epsilon)
        ] = accuracy

        print(
            f"Accuracy: {accuracy:.2f}%"
        )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

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

    print()
    print("=" * 70)
    print("CIFAR-10 ATTACK EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":
    main()