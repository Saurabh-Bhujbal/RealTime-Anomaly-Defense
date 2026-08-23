import torch
import torch.nn.functional as F


class SelfPurifier:
    """
    Lightweight self-purification module.

    Multiple transformed versions of an input are generated.
    The candidate producing the highest classifier confidence
    is selected as the purified image.
    """

    def __init__(self, model):

        self.model = model

    def _smooth_3x3(self, images):

        return F.avg_pool2d(
            images,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def _smooth_5x5(self, images):

        return F.avg_pool2d(
            images,
            kernel_size=5,
            stride=1,
            padding=2
        )

    def _denoise(self, images):

        smoothed = self._smooth_3x3(
            images
        )

        purified = (
            0.7 * smoothed
            +
            0.3 * images
        )

        return torch.clamp(
            purified,
            0.0,
            1.0
        )

    def generate_candidates(self, images):

        return [
            images,
            self._smooth_3x3(images),
            self._smooth_5x5(images),
            self._denoise(images)
        ]

    def purify(self, images):

        self.model.eval()

        candidates = self.generate_candidates(
            images
        )

        confidence_scores = []

        with torch.no_grad():

            for candidate in candidates:

                logits = self.model(
                    candidate
                )

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

                confidence, _ = probabilities.max(
                    dim=1
                )

                confidence_scores.append(
                    confidence
                )

        confidence_matrix = torch.stack(
            confidence_scores,
            dim=1
        )

        best_indices = confidence_matrix.argmax(
            dim=1
        )

        candidate_tensor = torch.stack(
            candidates,
            dim=1
        )

        batch_indices = torch.arange(
            images.size(0),
            device=images.device
        )

        purified_images = candidate_tensor[
            batch_indices,
            best_indices
        ]

        return purified_images.detach()

    def purify_with_details(self, images):

        self.model.eval()

        candidates = self.generate_candidates(
            images
        )

        confidence_scores = []

        predictions = []

        with torch.no_grad():

            for candidate in candidates:

                logits = self.model(
                    candidate
                )

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

                confidence, prediction = (
                    probabilities.max(dim=1)
                )

                confidence_scores.append(
                    confidence
                )

                predictions.append(
                    prediction
                )

        confidence_matrix = torch.stack(
            confidence_scores,
            dim=1
        )

        best_indices = confidence_matrix.argmax(
            dim=1
        )

        candidate_tensor = torch.stack(
            candidates,
            dim=1
        )

        batch_indices = torch.arange(
            images.size(0),
            device=images.device
        )

        purified_images = candidate_tensor[
            batch_indices,
            best_indices
        ]

        best_confidence = confidence_matrix[
            batch_indices,
            best_indices
        ]

        prediction_tensor = torch.stack(
            predictions,
            dim=1
        )

        purified_predictions = prediction_tensor[
            batch_indices,
            best_indices
        ]

        return {
            "purified_images": purified_images.detach(),
            "predictions": purified_predictions.detach(),
            "confidence": best_confidence.detach(),
            "selected_candidate": best_indices.detach()
        }