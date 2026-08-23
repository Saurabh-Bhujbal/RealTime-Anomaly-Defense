import torch


class AnomalyDefensePipeline:
    """
    Complete anomaly-aware defense pipeline.

    Normal samples:
        Input -> CNN

    Anomalous samples:
        Input -> Purification -> CNN
    """

    def __init__(
        self,
        model,
        detector,
        purifier
    ):

        self.model = model
        self.detector = detector
        self.purifier = purifier

    def predict(self, images):

        self.model.eval()

        detection = self.detector.detect(
            images
        )

        anomalous = detection[
            "is_anomalous"
        ]

        defended_images = images.clone()

        # Purify only samples detected as anomalous
        if anomalous.any():

            anomalous_images = images[
                anomalous
            ]

            purified_images = self.purifier.purify(
                anomalous_images
            )

            defended_images[
                anomalous
            ] = purified_images

        with torch.no_grad():

            logits = self.model(
                defended_images
            )

            predictions = logits.argmax(
                dim=1
            )

        return {
            "predictions": predictions,
            "anomalous": anomalous,
            "anomaly_scores": detection[
                "anomaly_score"
            ],
            "confidence": detection[
                "confidence"
            ],
            "defended_images": defended_images
        }

    def predict_with_details(self, images):

        self.model.eval()

        detection = self.detector.detect(
            images
        )

        anomalous = detection[
            "is_anomalous"
        ]

        defended_images = images.clone()

        purification_details = None

        if anomalous.any():

            anomalous_images = images[
                anomalous
            ]

            purification_details = (
                self.purifier.purify_with_details(
                    anomalous_images
                )
            )

            defended_images[
                anomalous
            ] = purification_details[
                "purified_images"
            ]

        with torch.no_grad():

            logits = self.model(
                defended_images
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, predictions = (
                probabilities.max(dim=1)
            )

        return {
            "predictions": predictions,
            "confidence": confidence,
            "anomalous": anomalous,
            "anomaly_scores": detection[
                "anomaly_score"
            ],
            "defended_images": defended_images,
            "purification_details":
                purification_details
        }