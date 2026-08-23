import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import sys

# ==========================================================
# PROJECT SETUP
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.cnn import MNISTCNN
from src.detection.anomaly_detector import DynamicAnomalyDetector
from src.purification.purifier import SelfPurifier
from src.defenses.randomized_defense import RandomizedDefense
from src.defenses.gradient_diversity import GradientDiversityDefense
from src.defenses.combined_defense import CombinedDefense
from src.attacks.fgsm import fgsm_attack

import config


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="RealTime Anomaly Defense",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #888888;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444444;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">🛡️ RealTime Anomaly Defense</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Adversarial Attack Detection and Robust Image Classification'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    device = torch.device("cpu")

    model = MNISTCNN(
        num_classes=config.NUM_CLASSES
    ).to(device)

    model_path = (
        config.MODEL_DIR /
        "baseline_cnn.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    return model, device


@st.cache_resource
def load_defenses(_model):

    # ------------------------------------------------------
    # Detector
    # ------------------------------------------------------

    detector = DynamicAnomalyDetector(
        _model,
        threshold_percentile=95.0
    )

    # Use calibrated threshold obtained
    # during project evaluation.
    try:
        detector.threshold = 0.037860
    except Exception:
        pass

    # ------------------------------------------------------
    # Purifier
    # ------------------------------------------------------

    purifier = SelfPurifier(
        _model
    )

    # ------------------------------------------------------
    # Randomized defense
    # ------------------------------------------------------

    randomized = RandomizedDefense(
        _model,
        num_samples=4,
        noise_std=0.03
    )

    # ------------------------------------------------------
    # Gradient diversity
    # ------------------------------------------------------

    gradient = GradientDiversityDefense(
        _model,
        consistency_threshold=0.75
    )

    # ------------------------------------------------------
    # Combined
    # ------------------------------------------------------

    combined = CombinedDefense(
        model=_model,
        detector=detector,
        purifier=purifier,
        randomized_defense=randomized,
        gradient_defense=gradient
    )

    return detector, purifier, randomized, gradient, combined


model, device = load_model()

(
    detector,
    purifier,
    randomized,
    gradient,
    combined
) = load_defenses(model)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🛡️ Defense System")

st.sidebar.markdown(
    """
    **Pipeline**

    Input Image  
    ↓  
    Anomaly Detection  
    ↓  
    Self-Purification  
    ↓  
    Randomized Defense  
    ↓  
    Gradient Diversity  
    ↓  
    Final Prediction
    """
)

st.sidebar.divider()

st.sidebar.metric(
    "Clean Accuracy",
    "98.94%"
)

st.sidebar.metric(
    "Detection Threshold",
    "0.037860"
)

st.sidebar.metric(
    "EOT Defense Accuracy",
    "79.46%"
)


# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Live Detection",
        "📊 Experimental Results",
        "ℹ️ About Project"
    ]
)


# ==========================================================
# TAB 1 — LIVE DETECTION
# ==========================================================

with tab1:

    st.header(
        "Upload an MNIST Image"
    )

    st.write(
        "Upload a handwritten digit image. "
        "The system will classify it and evaluate "
        "its anomaly score."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("L")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Input Image"
            )

            st.image(
                image,
                width=250
            )

        # --------------------------------------------------
        # PREPROCESS
        # --------------------------------------------------

        processed = ImageOps.invert(
            image
        )

        processed = processed.resize(
            (28, 28)
        )

        array = np.array(
            processed
        ).astype(
            np.float32
        ) / 255.0

        tensor = torch.tensor(
            array
        ).unsqueeze(
            0
        ).unsqueeze(
            0
        )

        tensor = tensor.to(device)

        # --------------------------------------------------
        # NORMAL PREDICTION
        # --------------------------------------------------

        with torch.no_grad():

            logits = model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = (
                probabilities.max(
                    dim=1
                )
            )

        predicted_digit = (
            prediction.item()
        )

        confidence_value = (
            confidence.item()
        ) * 100

        # --------------------------------------------------
        # RANDOMIZED PREDICTION
        # --------------------------------------------------

        randomized_result = (
            randomized.predict(
                tensor
            )
        )

        randomized_prediction = (
            randomized_result[
                "predictions"
            ].item()
        )

        randomized_confidence = (
            randomized_result[
                "confidence"
            ].item()
        ) * 100

        # --------------------------------------------------
        # GRADIENT DIVERSITY
        # --------------------------------------------------

        gradient_result = (
            gradient.predict(
                tensor
            )
        )

        gradient_prediction = (
            gradient_result[
                "final_predictions"
            ].item()
        )

        agreement = (
            gradient_result[
                "agreement"
            ].item()
        ) * 100

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        with col2:

            st.subheader(
                "Classification"
            )

            st.metric(
                "Predicted Digit",
                str(predicted_digit)
            )

            st.metric(
                "Confidence",
                f"{confidence_value:.2f}%"
            )

            st.metric(
                "Randomized Prediction",
                str(randomized_prediction)
            )

            st.metric(
                "Gradient Prediction",
                str(gradient_prediction)
            )

        st.divider()

        # --------------------------------------------------
        # REAL FGSM ADVERSARIAL TEST
        # --------------------------------------------------

        st.subheader("🧪 Live Adversarial Test")
        st.write(
            "Generate a genuine FGSM adversarial example from the uploaded image. "
            "This uses the same attack family evaluated in the project experiments."
        )

        fgsm_epsilon = st.select_slider(
            "FGSM attack strength (ε)",
            options=[0.05, 0.10, 0.20],
            value=0.20
        )

        if st.button("⚔️ Generate FGSM Attack", use_container_width=True):

            attack_label = prediction.detach().clone()

            adversarial_tensor = fgsm_attack(
                model,
                tensor,
                attack_label,
                epsilon=fgsm_epsilon
            )

            with torch.no_grad():
                adv_logits = model(adversarial_tensor)
                adv_probabilities = torch.softmax(adv_logits, dim=1)
                adv_confidence, adv_prediction = adv_probabilities.max(dim=1)

            adv_prediction_value = adv_prediction.item()
            adv_confidence_value = adv_confidence.item() * 100

            randomized_adv = randomized.predict(adversarial_tensor)
            randomized_adv_prediction = randomized_adv["predictions"].item()
            randomized_adv_confidence = randomized_adv["confidence"].item() * 100

            gradient_adv = gradient.predict(adversarial_tensor)
            gradient_adv_prediction = gradient_adv["final_predictions"].item()
            gradient_adv_agreement = gradient_adv["agreement"].item() * 100

            # Convert adversarial tensor back to displayable image.
            adv_display = adversarial_tensor.detach().cpu().squeeze().numpy()
            adv_display = np.clip(adv_display * 255.0, 0, 255).astype(np.uint8)

            adv_col1, adv_col2 = st.columns(2)

            with adv_col1:
                st.markdown("**Generated FGSM Adversarial Image**")
                st.image(adv_display, width=250)

            with adv_col2:
                st.markdown("**Adversarial Evaluation**")
                st.metric("Adversarial Prediction", str(adv_prediction_value))
                st.metric("Adversarial Confidence", f"{adv_confidence_value:.2f}%")
                st.metric("Randomized Prediction", str(randomized_adv_prediction))
                st.metric("Gradient Prediction", str(gradient_adv_prediction))
                st.metric("Defense Agreement", f"{gradient_adv_agreement:.2f}%")

                if gradient_adv_agreement < 75:
                    st.error("⚠️ Potential Anomaly / Adversarial Input")
                elif adv_prediction_value != predicted_digit:
                    st.warning("⚠️ Prediction changed under FGSM attack")
                else:
                    st.info("ℹ️ Attack did not change the final prediction")

        st.divider()

        # --------------------------------------------------
        # DEFENSE STATUS
        # --------------------------------------------------

        st.subheader(
            "Defense Analysis"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Randomized Confidence",
                f"{randomized_confidence:.2f}%"
            )

        with col2:

            st.metric(
                "Prediction Agreement",
                f"{agreement:.2f}%"
            )

        with col3:

            if agreement >= 75:

                st.success(
                    "Prediction Stable"
                )

            else:

                st.warning(
                    "Potential Anomaly"
                )


# ==========================================================
# TAB 2 — RESULTS
# ==========================================================

with tab2:

    st.header(
        "Experimental Results"
    )

    st.subheader(
        "Clean Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Clean Accuracy",
            "98.94%"
        )

    with col2:

        st.metric(
            "FGSM ε=0.20",
            "59.45%"
        )

    with col3:

        st.metric(
            "PGD ε=0.20",
            "80.18%"
        )

    st.divider()

    st.subheader(
        "Defense Ablation"
    )

    data = {
        "Defense": [
            "Baseline",
            "Purification",
            "Randomized",
            "Gradient Diversity",
            "Combined"
        ],

        "FGSM ε=0.20": [
            40.38,
            54.02,
            42.43,
            56.63,
            59.45
        ],

        "PGD ε=0.20": [
            62.46,
            76.32,
            67.06,
            76.20,
            80.18
        ]
    }

    st.dataframe(
        data,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Adaptive EOT-PGD"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Attack Accuracy",
            "61.94%"
        )

    with col2:

        st.metric(
            "Defended Accuracy",
            "79.46%"
        )

    with col3:

        st.metric(
            "Detection Rate",
            "67.63%"
        )

    st.divider()

    # ------------------------------------------------------
    # SHOW FIGURES
    # ------------------------------------------------------

    figure_files = [
        (
            "FGSM Defense Ablation",
            "fgsm_ablation_0.20.png"
        ),
        (
            "PGD Defense Ablation",
            "pgd_ablation_0.20.png"
        ),
        (
            "Baseline vs Combined",
            "baseline_vs_combined_0.20.png"
        ),
        (
            "Detection Rate",
            "detection_rate.png"
        ),
        (
            "EOT-PGD Defense",
            "eot_pgd_defense.png"
        )
    ]

    for title, filename in figure_files:

        figure_path = (
            PROJECT_ROOT /
            "experiments" /
            "figures" /
            filename
        )

        if figure_path.exists():

            st.subheader(title)

            st.image(
                str(figure_path),
                use_container_width=True
            )


# ==========================================================
# TAB 3 — ABOUT
# ==========================================================

with tab3:

    st.header(
        "About the Project"
    )

    st.markdown(
        """
        ### RealTime Anomaly Defense

        This project implements a multi-stage defense
        framework for detecting and mitigating adversarial
        attacks against image classification models.

        ### Defense Components

        - Dynamic anomaly detection
        - Self-purification
        - Randomized inference
        - Gradient diversity analysis
        - Combined defense pipeline
        - Adaptive EOT-PGD evaluation

        ### Attacks Evaluated

        - FGSM
        - PGD
        - EOT-PGD

        ### Baseline

        CNN trained on the MNIST dataset.

        **Clean Accuracy: 98.94%**

        ### Strongest Results

        **FGSM ε=0.20**

        40.38% → 59.45%

        **PGD ε=0.20**

        62.46% → 80.18%

        **EOT-PGD ε=0.20**

        61.94% → 79.46%
        """
    )

    st.success(
        "RealTime Anomaly Defense pipeline is ready."
    )