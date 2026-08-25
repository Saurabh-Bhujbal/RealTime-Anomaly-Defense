"""
Pydantic schemas for request validation and response serialisation.
Keep these separate from SQLAlchemy models so the API surface stays clean.
"""

import uuid
from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, EmailStr, Field


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# DEFENSE — ANALYZE (POST /api/defense/analyze)
# Image is sent as multipart/form-data; response is JSON.
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    """Full pipeline result for a clean image upload."""

    log_id: uuid.UUID

    # Raw model output
    clean_prediction: int
    clean_confidence: float

    # Anomaly detection
    is_anomalous: bool
    anomaly_score: float

    # Defense heads
    randomized_prediction: int
    randomized_confidence: float
    gradient_prediction: int
    gradient_agreement: float

    # Final combined output
    final_prediction: int
    defense_agreement_score: float

    execution_time_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# DEFENSE — ATTACK TEST (POST /api/defense/attack-test)
# ─────────────────────────────────────────────────────────────────────────────

class AttackTestRequest(BaseModel):
    """
    Form fields sent alongside the image file.
    attack_type: "fgsm" | "pgd" | "eot_pgd"
    epsilon:     perturbation strength, e.g. 0.05 / 0.10 / 0.20
    """
    attack_type: Literal["fgsm", "pgd", "eot_pgd"] = "fgsm"
    epsilon: float = Field(default=0.20, ge=0.0, le=1.0)


class AttackTestResponse(BaseModel):
    """Full pipeline result including adversarial evaluation."""

    log_id: uuid.UUID

    # Clean baseline
    clean_prediction: int
    clean_confidence: float

    # Adversarial result (un-defended)
    adversarial_prediction: int
    adversarial_confidence: float

    # Anomaly detection on adversarial image
    is_anomalous: bool
    anomaly_score: float

    # Defense heads on adversarial image
    randomized_prediction: int
    randomized_confidence: float
    gradient_prediction: int
    gradient_agreement: float

    # Final
    final_defended_prediction: int
    defense_agreement_score: float

    # Attack metadata
    attack_type: str
    epsilon: float
    execution_time_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY  (GET /api/history)
# ─────────────────────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    id: uuid.UUID
    clean_prediction: int
    clean_confidence: float
    is_anomalous: bool
    anomaly_score: float
    is_fgsm_attacked: bool
    attack_type: Optional[str]
    attack_epsilon: Optional[float]
    final_defended_prediction: int
    defense_agreement_score: float
    execution_time_ms: float
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    total: int
    items: list[HistoryItem]


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS  (GET /api/experiments/results)
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkResultSchema(BaseModel):
    id: uuid.UUID
    model_name: str
    defense_type: str
    attack_type: str
    epsilon: float
    accuracy: float
    detection_rate: Optional[float]

    model_config = {"from_attributes": True}


class ExperimentsResponse(BaseModel):
    results: list[BenchmarkResultSchema]
