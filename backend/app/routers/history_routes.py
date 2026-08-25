"""
GET /api/history          — paginated log of the authenticated user's analyses
GET /api/history/{log_id} — single log detail
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import get_current_user
from app.database import get_db
from app.models import AnomalyAnalysisLog, User
from app.schemas import HistoryItem, HistoryResponse

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=HistoryResponse)
async def get_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return the authenticated user's inference history, newest first.
    Supports cursor-style pagination via page / page_size query params.
    """
    offset = (page - 1) * page_size

    count_result = await db.execute(
        select(func.count(AnomalyAnalysisLog.id)).where(
            AnomalyAnalysisLog.user_id == current_user.id
        )
    )
    total = count_result.scalar_one()

    rows_result = await db.execute(
        select(AnomalyAnalysisLog)
        .where(AnomalyAnalysisLog.user_id == current_user.id)
        .order_by(AnomalyAnalysisLog.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    rows = rows_result.scalars().all()

    return HistoryResponse(
        total=total,
        items=[HistoryItem.model_validate(r) for r in rows],
    )


@router.get("/{log_id}", response_model=HistoryItem)
async def get_history_item(
    log_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return a single log entry, scoped to the current user."""
    result = await db.execute(
        select(AnomalyAnalysisLog).where(
            AnomalyAnalysisLog.id == log_id,
            AnomalyAnalysisLog.user_id == current_user.id,
        )
    )
    log = result.scalar_one_or_none()
    if log is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found.")
    return HistoryItem.model_validate(log)
