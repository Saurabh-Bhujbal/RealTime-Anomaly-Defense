"""
SQLAlchemy ORM table definitions.
Every class maps exactly to one PostgreSQL table.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


# ── Users ─────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # relationships
    analysis_logs: Mapped[list["AnomalyAnalysisLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


# ── Anomaly / Inference Logs ──────────────────────────────────────────────────
class AnomalyAnalysisLog(Base):
    """
    One row per image uploaded and processed through the defense pipeline.
    """
    __tablename__ = "anomaly_analysis_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Raw inference
    clean_prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    clean_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Anomaly detection
    is_anomalous: Mapped[bool] = mapped_column(Boolean, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Optional adversarial test
    is_fgsm_attacked: Mapped[bool] = mapped_column(Boolean, default=False)
    attack_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    attack_epsilon: Mapped[float | None] = mapped_column(Float, nullable=True)
    adversarial_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Defense outputs
    purified_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)
    randomized_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gradient_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_defended_prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    defense_agreement_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Performance
    execution_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # relationships
    user: Mapped["User | None"] = relationship(back_populates="analysis_logs")


# ── Benchmark Results ─────────────────────────────────────────────────────────
class BenchmarkResult(Base):
    """
    Static benchmark rows seeded from the experiments/ evaluation scripts.
    Read-only from the API; populated once via Alembic seed or evaluate_attacks.py.
    """
    __tablename__ = "benchmark_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    defense_type: Mapped[str] = mapped_column(String(100), nullable=False)
    attack_type: Mapped[str] = mapped_column(String(100), nullable=False)
    epsilon: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    detection_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
