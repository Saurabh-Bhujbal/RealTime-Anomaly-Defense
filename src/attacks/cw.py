import torch
import torch.nn.functional as F


def carlini_wagner_attack(
    model,
    images,
    labels,
    targeted=False,
    target_labels=None,
    c=1.0,
    kappa=0.0,
    lr=0.01,
    steps=100,
    device="cpu",
):
    """
    Simplified L2 Carlini-Wagner adversarial attack.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.

    images : torch.Tensor
        Input images in [0, 1].

    labels : torch.Tensor
        Correct labels.

    targeted : bool
        Whether to perform a targeted attack.

    target_labels : torch.Tensor
        Target labels for targeted attack.

    c : float
        Trade-off between perturbation size and attack success.

    kappa : float
        Confidence margin.

    lr : float
        Adam learning rate.

    steps : int
        Number of optimization iterations.

    device : str
        "cpu" or "cuda".

    Returns
    -------
    torch.Tensor
        Generated adversarial images.
    """

    model.eval()

    images = images.to(device)
    labels = labels.to(device)

    if targeted:
        if target_labels is None:
            raise ValueError(
                "target_labels must be provided for targeted C&W attack."
            )

        target_labels = target_labels.to(device)

    # Keep images inside valid range.
    images = images.clamp(0.0, 1.0)

    # Convert image from [0, 1] to tanh-space.
    scaled_images = images * 2.0 - 1.0

    scaled_images = scaled_images.clamp(
        -0.999999,
        0.999999
    )

    w = torch.atanh(scaled_images)

    w = w.detach().clone()
    w.requires_grad = True

    optimizer = torch.optim.Adam(
        [w],
        lr=lr
    )

    best_adv = images.detach().clone()

    best_l2 = torch.full(
        (images.size(0),),
        float("inf"),
        device=device
    )

    attack_labels = (
        target_labels
        if targeted
        else labels
    )

    for step in range(steps):

        # Convert back from tanh-space.
        adv_images = torch.tanh(w)

        adv_images = (
            adv_images + 1.0
        ) / 2.0

        logits = model(adv_images)

        # Logit corresponding to correct/target class.
        real = logits.gather(
            1,
            attack_labels.unsqueeze(1)
        ).squeeze(1)

        # Remove the selected class.
        other_logits = logits.clone()

        other_logits.scatter_(
            1,
            attack_labels.unsqueeze(1),
            float("-inf")
        )

        other = other_logits.max(
            dim=1
        ).values

        if targeted:

            # Target class should be stronger.
            attack_loss = torch.clamp(
                other - real + kappa,
                min=0.0
            )

        else:

            # Original class should become weaker.
            attack_loss = torch.clamp(
                real - other + kappa,
                min=0.0
            )

        # L2 perturbation.
        l2 = (
            (adv_images - images) ** 2
        ).view(
            images.size(0),
            -1
        ).sum(dim=1)

        total_loss = (
            l2 +
            c * attack_loss
        )

        optimizer.zero_grad()

        total_loss.sum().backward()

        optimizer.step()

        # Track successful attacks with smallest perturbation.
        with torch.no_grad():

            predictions = logits.argmax(
                dim=1
            )

            if targeted:

                success = (
                    predictions ==
                    target_labels
                )

            else:

                success = (
                    predictions !=
                    labels
                )

            improved = (
                success &
                (l2 < best_l2)
            )

            best_l2[improved] = (
                l2[improved]
            )

            best_adv[improved] = (
                adv_images[improved]
            )

    return best_adv.detach()