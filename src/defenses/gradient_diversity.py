import torch
import torch.nn.functional as F


class GradientDiversityDefense:
    """
    Gradient-diversity-inspired defense.

    The input is evaluated through several slightly different
    differentiable transformations. The consistency of the
    resulting predictions is used as a stability signal.

    This implementation is intended as a lightweight defense
    component for the MNIST experimental pipeline.
    """

    def __init__(
        self,
        model,
        consistency_threshold=0.75
    ):

        self.model = model
        self.consistency_threshold = (
            consistency_threshold
        )

    def _transformations(self, images):

        transformed = []

        # Original
        transformed.append(
            images
        )

        # 3x3 smoothing
        transformed.append(
            F.avg_pool2d(
                images,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # 5x5 smoothing
        transformed.append(
            F.avg_pool2d(
                images,
                kernel_size=5,
                stride=1,
                padding=2
            )
        )

        # Slight sharpening
        blurred = F.avg_pool2d(
            images,
            kernel_size=3,
            stride=1,
            padding=1
        )

        sharpened = torch.clamp(
            images + 0.2 * (images - blurred),
            0.0,
            1.0
        )

        transformed.append(
            sharpened
        )

        return transformed

    def analyze(self, images):

        self.model.eval()

        transformations = self._transformations(
            images
        )

        predictions = []
        confidences = []

        with torch.no_grad():

            for transformed in transformations:

                logits = self.model(
                    transformed
                )

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

                confidence, prediction = (
                    probabilities.max(dim=1)
                )

                predictions.append(
                    prediction
                )

                confidences.append(
                    confidence
                )

        prediction_tensor = torch.stack(
            predictions,
            dim=1
        )

        confidence_tensor = torch.stack(
            confidences,
            dim=1
        )

        # Agreement with the first/original prediction
        reference_prediction = (
            prediction_tensor[:, 0]
        )

        agreement = (
            prediction_tensor ==
            reference_prediction.unsqueeze(1)
        ).float().mean(dim=1)

        mean_confidence = (
            confidence_tensor.mean(dim=1)
        )

        diversity_score = (
            1.0 - agreement
        )

        unstable = (
            agreement <
            self.consistency_threshold
        )

        return {
            "predictions": prediction_tensor,
            "confidences": confidence_tensor,
            "agreement": agreement,
            "mean_confidence": mean_confidence,
            "diversity_score": diversity_score,
            "unstable": unstable
        }

    def predict(self, images):

        results = self.analyze(
            images
        )

        prediction_tensor = (
            results["predictions"]
        )

        batch_size = images.size(0)

        final_predictions = []

        for i in range(batch_size):

            values = prediction_tensor[i]

            counts = torch.bincount(
                values,
                minlength=10
            )

            final_predictions.append(
                counts.argmax()
            )

        final_predictions = torch.stack(
            final_predictions
        )

        results["final_predictions"] = (
            final_predictions
        )

        return results