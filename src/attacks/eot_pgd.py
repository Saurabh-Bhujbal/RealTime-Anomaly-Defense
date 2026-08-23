import torch
import torch.nn.functional as F


def eot_transform(images, noise_std=0.03):
    """
    Randomized transformation used by the adaptive attack.
    """

    noise = torch.randn_like(images) * noise_std

    transformed = images + noise

    return torch.clamp(
        transformed,
        0.0,
        1.0
    )


def eot_pgd_attack(
    model,
    images,
    labels,
    epsilon=0.20,
    alpha=0.01,
    steps=10,
    eot_samples=4
):
    """
    Expectation Over Transformation (EOT) PGD.

    The attack estimates the gradient over multiple
    randomized transformations before updating the image.
    """

    original_images = images.detach().clone()

    # Random initialization
    adversarial_images = (
        original_images
        + torch.empty_like(original_images).uniform_(
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

        gradient_sum = torch.zeros_like(
            adversarial_images
        )

        for _ in range(eot_samples):

            transformed = eot_transform(
                adversarial_images
            )

            transformed.requires_grad_(True)

            logits = model(transformed)

            loss = F.cross_entropy(
                logits,
                labels
            )

            gradient = torch.autograd.grad(
                loss,
                transformed,
                retain_graph=False,
                create_graph=False
            )[0]

            gradient_sum += gradient.detach()

        # Expected gradient
        expected_gradient = (
            gradient_sum / eot_samples
        )

        # PGD update
        adversarial_images = (
            adversarial_images.detach()
            + alpha * expected_gradient.sign()
        )

        # Project into epsilon ball
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
            original_images +
            perturbation
        )

        adversarial_images = torch.clamp(
            adversarial_images,
            0.0,
            1.0
        )

    return adversarial_images.detach()