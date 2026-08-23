import torch
import torch.nn.functional as F


def pgd_attack(
    model,
    images,
    labels,
    epsilon=0.2,
    alpha=0.01,
    steps=10
):
    """
    Projected Gradient Descent (PGD) attack.

    Args:
        model: Trained PyTorch classifier.
        images: Clean images in [0, 1].
        labels: Ground-truth labels.
        epsilon: Maximum perturbation.
        alpha: Step size.
        steps: Number of attack iterations.

    Returns:
        PGD adversarial images.
    """

    original_images = images.clone().detach()

    # Random initialization inside epsilon-ball
    adversarial_images = (
        original_images +
        torch.empty_like(original_images).uniform_(
            -epsilon,
            epsilon
        )
    )

    adversarial_images = torch.clamp(
        adversarial_images,
        0.0,
        1.0
    )

    for _ in range(steps):

        adversarial_images.requires_grad = True

        outputs = model(adversarial_images)

        loss = F.cross_entropy(
            outputs,
            labels
        )

        model.zero_grad()

        loss.backward()

        gradient_sign = (
            adversarial_images.grad.sign()
        )

        adversarial_images = (
            adversarial_images.detach()
            + alpha * gradient_sign
        )

        # Project into epsilon-ball around original image
        perturbation = (
            adversarial_images -
            original_images
        )

        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon
        )

        adversarial_images = (
            original_images + perturbation
        )

        adversarial_images = torch.clamp(
            adversarial_images,
            0.0,
            1.0
        )

    return adversarial_images.detach()