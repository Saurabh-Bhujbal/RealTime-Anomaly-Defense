import torch
import torch.nn as nn
import torch.nn.functional as F


class DIPNetwork(nn.Module):
    """
    Small Deep Image Prior network for MNIST images.

    The network is randomly initialized for every image.
    No pretrained weights are used.
    """

    def __init__(self, channels=32):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                1,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.net(x)


class DIPPurifier:
    """
    DIP-style image purification.

    A fresh randomly initialized network is optimized for
    each input image.

    The optimization uses a fixed random input and a noisy
    reconstruction target to encourage a stable structured
    reconstruction.
    """

    def __init__(
        self,
        iterations=300,
        learning_rate=0.005,
        noise_std=0.01,
        device="cpu"
    ):

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.device = device

    def purify_single(
        self,
        image
    ):

        image = image.detach().to(
            self.device
        )

        # --------------------------------------------------
        # Ensure shape is [1, 1, H, W]
        # --------------------------------------------------

        if image.dim() == 3:

            target = image.unsqueeze(0)

        elif image.dim() == 4:

            target = image

        else:

            raise ValueError(
                "Expected image shape [1,H,W] "
                "or [1,1,H,W]."
            )

        target = target.float().clamp(
            0.0,
            1.0
        )

        # --------------------------------------------------
        # New network for every image
        # --------------------------------------------------

        network = DIPNetwork(
            channels=32
        ).to(
            self.device
        )

        network.train()

        # --------------------------------------------------
        # Fixed random input
        # --------------------------------------------------

        z = torch.rand(
            1,
            1,
            target.shape[-2],
            target.shape[-1],
            device=self.device
        )

        z = z.detach()

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.learning_rate
        )

        # --------------------------------------------------
        # Best reconstruction
        # --------------------------------------------------

        best_output = target.detach().clone()

        best_loss = float("inf")

        # --------------------------------------------------
        # Optimization
        # --------------------------------------------------

        for iteration in range(
            self.iterations
        ):

            # Add very small noise to DIP input.
            noisy_z = (
                z +
                torch.randn_like(z) *
                self.noise_std
            )

            noisy_z = noisy_z.clamp(
                0.0,
                1.0
            )

            output = network(
                noisy_z
            )

            # Reconstruction loss.
            reconstruction_loss = F.mse_loss(
                output,
                target
            )

            # Small smoothness regularization.
            tv_h = torch.mean(
                torch.abs(
                    output[:, :, 1:, :] -
                    output[:, :, :-1, :]
                )
            )

            tv_w = torch.mean(
                torch.abs(
                    output[:, :, :, 1:] -
                    output[:, :, :, :-1]
                )
            )

            total_variation = (
                tv_h + tv_w
            )

            loss = (
                reconstruction_loss +
                0.0001 * total_variation
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # --------------------------------------------------
            # Track best reconstruction
            # --------------------------------------------------

            if (
                reconstruction_loss.item()
                < best_loss
            ):

                best_loss = (
                    reconstruction_loss.item()
                )

                best_output = (
                    output.detach().clone()
                )

        network.eval()

        return best_output.squeeze(0).clamp(
            0.0,
            1.0
        )

    def purify(
        self,
        images,
        verbose=False
    ):

        purified_images = []

        for index in range(
            images.size(0)
        ):

            if verbose:

                print(
                    f"DIP purification "
                    f"{index + 1}/"
                    f"{images.size(0)}"
                )

            purified = self.purify_single(
                images[index]
            )

            purified_images.append(
                purified
            )

        return torch.stack(
            purified_images
        ).detach()