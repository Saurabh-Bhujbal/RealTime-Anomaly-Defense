import torch


class CombinedDefense:

    """
    Combined defense:

        Input
          ↓
        Anomaly Detection
          ↓
        If suspicious:
            Purification
            +
            Randomized Prediction
            +
            Stability Analysis
          ↓
        Final Prediction
    """

    def __init__(
        self,
        model,
        detector,
        purifier,
        randomized_defense,
        gradient_defense
    ):

        self.model = model
        self.detector = detector
        self.purifier = purifier
        self.randomized_defense = (
            randomized_defense
        )
        self.gradient_defense = (
            gradient_defense
        )

    def predict(self, images):

        self.model.eval()

        detection = self.detector.detect(
            images
        )

        anomalous = detection[
            "is_anomalous"
        ]

        defended_images = images.clone()

        # --------------------------------------------------
        # Purification
        # --------------------------------------------------

        if anomalous.any():

            purified = self.purifier.purify(
                images[anomalous]
            )

            defended_images[
                anomalous
            ] = purified

        # --------------------------------------------------
        # Randomized prediction
        # --------------------------------------------------

        randomized_result = (
            self.randomized_defense.predict(
                defended_images
            )
        )

        # --------------------------------------------------
        # Stability / diversity analysis
        # --------------------------------------------------

        diversity_result = (
            self.gradient_defense.predict(
                defended_images
            )
        )

        randomized_predictions = (
            randomized_result[
                "predictions"
            ]
        )

        diversity_predictions = (
            diversity_result[
                "final_predictions"
            ]
        )

        # --------------------------------------------------
        # Prediction agreement
        # --------------------------------------------------

        agreement = (
            randomized_predictions ==
            diversity_predictions
        )

        # If both defenses agree, use that prediction.
        # Otherwise use the randomized prediction.
        final_predictions = torch.where(
            agreement,
            randomized_predictions,
            diversity_predictions
        )

        return {
            "predictions":
                final_predictions,

            "anomalous":
                anomalous,

            "anomaly_scores":
                detection["anomaly_score"],

            "randomized_predictions":
                randomized_predictions,

            "diversity_predictions":
                diversity_predictions,

            "agreement":
                agreement,

            "defended_images":
                defended_images
        }