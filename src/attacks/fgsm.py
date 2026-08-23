import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon):
    """
    Fast Gradient Sign Method (FGSM).

    Args:
        model: Trained PyTorch classifier.
        images: Clean images in [0, 1].
        labels: Ground-truth labels.
        epsilon: Perturbation strength.

    Returns:
        Adversarial images in [0, 1].
    """

    images = images.clone().detach()
    images.requires_grad = True

    outputs = model(images)

    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()

    loss.backward()

    gradient_sign = images.grad.sign()

    adversarial_images = (
        images + epsilon * gradient_sign
    )

    adversarial_images = torch.clamp(
        adversarial_images,
        min=0.0,
        max=1.0
    )

    return adversarial_images.detach()