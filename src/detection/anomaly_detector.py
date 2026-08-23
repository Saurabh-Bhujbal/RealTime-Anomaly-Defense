import torch
import torch.nn.functional as F


class DynamicAnomalyDetector:
    """
    Dynamic adversarial anomaly detector.

    The detector uses:
    1. Model confidence
    2. Prediction instability after smoothing
    3. Confidence variation after smoothing

    The threshold is calibrated using clean validation data.
    """

    def __init__(
        self,
        model,
        threshold_percentile=95.0
    ):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def _get_statistics(self, images):

        self.model.eval()

        with torch.no_grad():

            # Original prediction
            original_logits = self.model(images)

            original_probs = F.softmax(
                original_logits,
                dim=1
            )

            original_confidence, original_prediction = (
                original_probs.max(dim=1)
            )

            # Smoothed image
            smoothed_images = F.avg_pool2d(
                images,
                kernel_size=3,
                stride=1,
                padding=1
            )

            smoothed_logits = self.model(
                smoothed_images
            )

            smoothed_probs = F.softmax(
                smoothed_logits,
                dim=1
            )

            smoothed_confidence, smoothed_prediction = (
                smoothed_probs.max(dim=1)
            )

        # Prediction changes after smoothing
        prediction_change = (
            original_prediction != smoothed_prediction
        ).float()

        # Confidence changes after smoothing
        confidence_change = torch.abs(
            original_confidence -
            smoothed_confidence
        )

        # Normalize confidence risk
        low_confidence = (
            1.0 - original_confidence
        )

        # Combined anomaly score
        anomaly_score = (
            0.50 * low_confidence
            +
            0.30 * prediction_change
            +
            0.20 * confidence_change
        )

        return {
            "confidence": original_confidence,
            "prediction": original_prediction,
            "smoothed_confidence": smoothed_confidence,
            "smoothed_prediction": smoothed_prediction,
            "prediction_change": prediction_change,
            "confidence_change": confidence_change,
            "anomaly_score": anomaly_score
        }

    def calibrate(
        self,
        clean_loader,
        device
    ):
        """
        Calibrate threshold using clean validation data.
        """

        scores = []

        for images, _ in clean_loader:

            images = images.to(device)

            statistics = self._get_statistics(
                images
            )

            scores.extend(
                statistics["anomaly_score"]
                .detach()
                .cpu()
                .tolist()
            )

        scores = torch.tensor(scores)

        self.threshold = torch.quantile(
            scores,
            self.threshold_percentile / 100.0
        ).item()

        return self.threshold

    def detect(self, images):
        """
        Detect anomalous/adversarial samples.
        """

        if self.threshold is None:

            raise RuntimeError(
                "Detector has not been calibrated. "
                "Call calibrate() first."
            )

        statistics = self._get_statistics(
            images
        )

        statistics["is_anomalous"] = (
            statistics["anomaly_score"]
            > self.threshold
        )

        return statistics

    def get_threshold(self):
        """
        Return calibrated threshold.
        """

        if self.threshold is None:
            raise RuntimeError(
                "Detector has not been calibrated."
            )

        return self.threshold