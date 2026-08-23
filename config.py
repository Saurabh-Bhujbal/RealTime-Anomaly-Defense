from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"

LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"


# ============================================================
# DATASET
# ============================================================

NUM_CLASSES = 10

IMAGE_SIZE = 28

CHANNELS = 1


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 128

LEARNING_RATE = 0.001

EPOCHS = 5

RANDOM_SEED = 42


# ============================================================
# ADVERSARIAL ATTACKS
# ============================================================

FGSM_EPSILON = 0.2

PGD_EPSILON = 0.2

PGD_ALPHA = 0.01

PGD_STEPS = 10


# ============================================================
# ANOMALY DETECTION
# ============================================================

ANOMALY_THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
