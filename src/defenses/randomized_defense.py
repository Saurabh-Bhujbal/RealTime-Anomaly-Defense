import torch
import torch.nn.functional as F


class RandomizedDefense:
    """
    Randomized input defense.

    Multiple noisy versions of an image are classified.
    The final prediction is obtained from aggregated
    predictions across the randomized samples.
    """

    def __init__(
        self,
        model,
        num_samples=8,
        noise_std=0.03
    ):
        self.model = model
        self.num_samples = num_samples
        self.noise_std = noise_std

    def _add_noise(self, images):

        noise = torch.randn_like(images)

        noisy_images = (
            images +
            self.noise_std * noise
        )

        return torch.clamp(
            noisy_images,
            0.0,
            1.0
        )

    def predict(self, images):

        self.model.eval()

        all_probabilities = []

        with torch.no_grad():

            for _ in range(self.num_samples):

                noisy_images = self._add_noise(
                    images
                )

                logits = self.model(
                    noisy_images
                )

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

                all_probabilities.append(
                    probabilities
                )

        probability_tensor = torch.stack(
            all_probabilities,
            dim=0
        )

        # Average prediction probabilities
        mean_probabilities = (
            probability_tensor.mean(dim=0)
        )

        confidence, predictions = (
            mean_probabilities.max(dim=1)
        )

        return {
            "predictions": predictions,
            "confidence": confidence,
            "probabilities": mean_probabilities
        }

    def predict_with_variance(self, images):

        self.model.eval()

        all_probabilities = []

        with torch.no_grad():

            for _ in range(self.num_samples):

                noisy_images = self._add_noise(
                    images
                )

                logits = self.model(
                    noisy_images
                )

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

                all_probabilities.append(
                    probabilities
                )

        probability_tensor = torch.stack(
            all_probabilities,
            dim=0
        )

        mean_probabilities = (
            probability_tensor.mean(dim=0)
        )

        probability_variance = (
            probability_tensor.var(dim=0)
        )

        confidence, predictions = (
            mean_probabilities.max(dim=1)
        )

        return {
            "predictions": predictions,
            "confidence": confidence,
            "probabilities": mean_probabilities,
            "variance": probability_variance
        }