"""
GET /api/experiments/results  — return all benchmark rows (public, read-only)

These rows are seeded by running the existing evaluate_attacks.py script
with a small adapter that writes to PostgreSQL instead of (or in addition to)
saving local JSON/CSV files.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import BenchmarkResult
from app.schemas import ExperimentsResponse, BenchmarkResultSchema

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])


@router.get("/results", response_model=ExperimentsResponse)
async def get_results(db: AsyncSession = Depends(get_db)):
    """
    Return all stored benchmark evaluation results.
    The React AnalyticsPage reads this to render charts.
    """
    result = await db.execute(
        select(BenchmarkResult).order_by(BenchmarkResult.created_at)
    )
    rows = result.scalars().all()
    return ExperimentsResponse(
        results=[BenchmarkResultSchema.model_validate(r) for r in rows]
    )
