"""
DefenseService — the single bridge between FastAPI and the src/ ML modules.

Responsibilities
────────────────
1. Load the MNISTCNN checkpoint once at startup (singleton pattern).
2. Build all defence components (DynamicAnomalyDetector, SelfPurifier,
   RandomizedDefense, GradientDiversityDefense, CombinedDefense).
3. Expose two public async methods that the routers call:
     - run_analysis(image_bytes)        → clean-image pipeline
     - run_attack_test(image_bytes, ...) → adversarial pipeline
4. Persist the result to PostgreSQL and return the Pydantic response schema.

No ML logic lives in the routers — they only call this service.
"""

import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps
import io

from sqlalchemy.ext.asyncio import AsyncSession

# ── Make src/ importable (repo root is two levels above backend/app/) ─────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.cnn import MNISTCNN
from src.detection.anomaly_detector import DynamicAnomalyDetector
from src.purification.purifier import SelfPurifier
from src.defenses.randomized_defense import RandomizedDefense
from src.defenses.gradient_diversity import GradientDiversityDefense
from src.defenses.combined_defense import CombinedDefense
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.eot_pgd import eot_pgd_attack

from app.config import settings
from app.models import AnomalyAnalysisLog
from app.schemas import AnalyzeResponse, AttackTestResponse


# ─────────────────────────────────────────────────────────────────────────────
# Singleton model + defense components
# ─────────────────────────────────────────────────────────────────────────────

_model: Optional[MNISTCNN] = None
_device: Optional[torch.device] = None
_detector: Optional[DynamicAnomalyDetector] = None
_purifier: Optional[SelfPurifier] = None
_randomized: Optional[RandomizedDefense] = None
_gradient: Optional[GradientDiversityDefense] = None
_combined: Optional[CombinedDefense] = None


def _load_once() -> None:
    """
    Load model and all defence components into module-level singletons.
    Called once from app lifespan; safe to call multiple times (idempotent).
    """
    global _model, _device, _detector, _purifier, _randomized, _gradient, _combined

    if _model is not None:
        return  # already loaded

    _device = torch.device("cpu")

    _model = MNISTCNN(num_classes=settings.NUM_CLASSES).to(_device)
    _model.load_state_dict(
        torch.load(str(settings.MODEL_PATH), map_location=_device)
    )
    _model.eval()

    # Detector — use calibrated threshold from project evaluation
    _detector = DynamicAnomalyDetector(_model, threshold_percentile=95.0)
    _detector.threshold = 0.037860  # pre-calibrated value

    _purifier = SelfPurifier(_model)

    _randomized = RandomizedDefense(_model, num_samples=4, noise_std=0.03)

    _gradient = GradientDiversityDefense(_model, consistency_threshold=0.75)

    _combined = CombinedDefense(
        model=_model,
        detector=_detector,
        purifier=_purifier,
        randomized_defense=_randomized,
        gradient_defense=_gradient,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing helper
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes) -> torch.Tensor:
    """
    Robust preprocessing pipeline:
      1. Open image and handle transparency
      2. Convert to grayscale
      3. Intelligently invert (MNIST uses white-on-black)
      4. Resize to 28x28
      5. Normalise to [0, 1]
      6. Return shape [1, 1, 28, 28]
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    # Handle transparency by pasting over a white background
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = image.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg.convert("RGB")
        
    image = image.convert("L")
    
    # Intelligently invert: if mostly light, assume it's black-on-white
    arr_check = np.array(image)
    if np.mean(arr_check) > 127:
        image = ImageOps.invert(image)
        
    # Resize with antialiasing
    if hasattr(Image, "Resampling"):
        resample_filter = Image.Resampling.LANCZOS
    else:
        resample_filter = Image.LANCZOS
    image = image.resize((28, 28), resample=resample_filter)
    
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    return tensor.to(_device)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def run_analysis(
    image_bytes: bytes,
    db: AsyncSession,
    user_id: Optional[uuid.UUID] = None,
) -> AnalyzeResponse:
    """
    Run the full defense pipeline on a clean image.
    Persists result to anomaly_analysis_logs and returns AnalyzeResponse.
    """
    _load_once()

    t0 = time.perf_counter()
    tensor = _preprocess(image_bytes)

    # ── Clean prediction ──────────────────────────────────────────────────────
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    clean_pred = pred.item()
    clean_conf = conf.item()

    # ── Anomaly detection ─────────────────────────────────────────────────────
    detection = _detector.detect(tensor)
    is_anomalous = bool(detection["is_anomalous"].item())
    anomaly_score = float(detection["anomaly_score"].item())

    # ── Randomized defense ────────────────────────────────────────────────────
    rand_result = _randomized.predict(tensor)
    rand_pred = int(rand_result["predictions"].item())
    rand_conf = float(rand_result["confidence"].item())

    # ── Gradient diversity ────────────────────────────────────────────────────
    grad_result = _gradient.predict(tensor)
    grad_pred = int(grad_result["final_predictions"].item())
    grad_agreement = float(grad_result["agreement"].item())

    # ── Combined ──────────────────────────────────────────────────────────────
    combined_result = _combined.predict(tensor)
    final_pred = int(combined_result["predictions"].item())
    final_agreement = float(combined_result["agreement"].item())

    exec_ms = (time.perf_counter() - t0) * 1000.0

    print("\n" + "="*50)
    print("🧠 LIVE PYTORCH ML PREDICTION LOG")
    print("="*50)
    print(f"  Clean Pred:       {clean_pred} (conf: {clean_conf*100:.1f}%)")
    print(f"  Anomaly Score:    {anomaly_score:.4f} (Anomalous: {is_anomalous})")
    print(f"  Randomized Pred:  {rand_pred}")
    print(f"  Gradient Pred:    {grad_pred}")
    print(f"  Final Combined:   {final_pred} (Agreement: {final_agreement*100:.1f}%)")
    print("="*50 + "\n")

    # ── Persist ───────────────────────────────────────────────────────────────
    log = AnomalyAnalysisLog(
        user_id=user_id,
        clean_prediction=clean_pred,
        clean_confidence=clean_conf,
        is_anomalous=is_anomalous,
        anomaly_score=anomaly_score,
        is_fgsm_attacked=False,
        randomized_prediction=rand_pred,
        gradient_prediction=grad_pred,
        final_defended_prediction=final_pred,
        defense_agreement_score=float(final_agreement),
        execution_time_ms=exec_ms,
    )
    db.add(log)
    await db.flush()  # get the generated UUID before commit

    return AnalyzeResponse(
        log_id=log.id,
        clean_prediction=clean_pred,
        clean_confidence=round(clean_conf, 6),
        is_anomalous=is_anomalous,
        anomaly_score=round(anomaly_score, 6),
        randomized_prediction=rand_pred,
        randomized_confidence=round(rand_conf, 6),
        gradient_prediction=grad_pred,
        gradient_agreement=round(grad_agreement, 6),
        final_prediction=final_pred,
        defense_agreement_score=round(float(final_agreement), 6),
        execution_time_ms=round(exec_ms, 2),
    )


async def run_attack_test(
    image_bytes: bytes,
    attack_type: str,
    epsilon: float,
    db: AsyncSession,
    user_id: Optional[uuid.UUID] = None,
) -> AttackTestResponse:
    """
    Generate an adversarial example, then run the full defense pipeline.
    Persists result to anomaly_analysis_logs and returns AttackTestResponse.
    """
    _load_once()

    t0 = time.perf_counter()
    tensor = _preprocess(image_bytes)

    # ── Clean baseline ────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    clean_pred = int(pred.item())
    clean_conf = float(conf.item())

    # ── Generate adversarial example ──────────────────────────────────────────
    label = pred.detach().clone()

    if attack_type == "fgsm":
        adv_tensor = fgsm_attack(_model, tensor, label, epsilon=epsilon)
    elif attack_type == "pgd":
        adv_tensor = pgd_attack(_model, tensor, label, epsilon=epsilon)
    elif attack_type == "eot_pgd":
        adv_tensor = eot_pgd_attack(_model, tensor, label, epsilon=epsilon)
    else:
        adv_tensor = fgsm_attack(_model, tensor, label, epsilon=epsilon)

    # ── Raw adversarial prediction (no defense) ───────────────────────────────
    with torch.no_grad():
        adv_logits = _model(adv_tensor)
        adv_probs = torch.softmax(adv_logits, dim=1)
        adv_conf, adv_pred = adv_probs.max(dim=1)

    adv_pred_val = int(adv_pred.item())
    adv_conf_val = float(adv_conf.item())

    # ── Anomaly detection on adversarial ─────────────────────────────────────
    detection = _detector.detect(adv_tensor)
    is_anomalous = bool(detection["is_anomalous"].item())
    anomaly_score = float(detection["anomaly_score"].item())

    # ── Defense heads on adversarial ──────────────────────────────────────────
    rand_result = _randomized.predict(adv_tensor)
    rand_pred = int(rand_result["predictions"].item())
    rand_conf = float(rand_result["confidence"].item())

    grad_result = _gradient.predict(adv_tensor)
    grad_pred = int(grad_result["final_predictions"].item())
    grad_agreement = float(grad_result["agreement"].item())

    combined_result = _combined.predict(adv_tensor)
    final_pred = int(combined_result["predictions"].item())
    final_agreement = float(combined_result["agreement"].item())

    exec_ms = (time.perf_counter() - t0) * 1000.0

    # ── Persist ───────────────────────────────────────────────────────────────
    log = AnomalyAnalysisLog(
        user_id=user_id,
        clean_prediction=clean_pred,
        clean_confidence=clean_conf,
        is_anomalous=is_anomalous,
        anomaly_score=anomaly_score,
        is_fgsm_attacked=True,
        attack_type=attack_type,
        attack_epsilon=epsilon,
        adversarial_prediction=adv_pred_val,
        randomized_prediction=rand_pred,
        gradient_prediction=grad_pred,
        final_defended_prediction=final_pred,
        defense_agreement_score=float(final_agreement),
        execution_time_ms=exec_ms,
    )
    db.add(log)
    await db.flush()

    return AttackTestResponse(
        log_id=log.id,
        clean_prediction=clean_pred,
        clean_confidence=round(clean_conf, 6),
        adversarial_prediction=adv_pred_val,
        adversarial_confidence=round(adv_conf_val, 6),
        is_anomalous=is_anomalous,
        anomaly_score=round(anomaly_score, 6),
        randomized_prediction=rand_pred,
        randomized_confidence=round(rand_conf, 6),
        gradient_prediction=grad_pred,
        gradient_agreement=round(grad_agreement, 6),
        final_defended_prediction=final_pred,
        defense_agreement_score=round(float(final_agreement), 6),
        attack_type=attack_type,
        epsilon=epsilon,
        execution_time_ms=round(exec_ms, 2),
    )
