"""
POST /api/defense/analyze      — upload a clean image, run full pipeline
POST /api/defense/attack-test  — upload an image, generate adversarial, run pipeline

Both endpoints accept multipart/form-data and REQUIRE a valid Bearer JWT.
Every analysis is logged against the authenticated user's account.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import get_current_user
from app.database import get_db
from app.models import User
from app.schemas import AnalyzeResponse, AttackTestResponse
from app.services import defense_service

router = APIRouter(prefix="/api/defense", tags=["Defense"])

_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
_MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {file.content_type}. Use PNG or JPEG.",
        )
    data = await file.read()
    if len(data) > _MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 5 MB limit.",
        )
    return data


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Handwritten digit image (PNG/JPEG)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a grayscale digit image and run the full defense pipeline.

    Authentication is REQUIRED — the caller must present a valid Bearer JWT.
    The result is logged against the authenticated user's account.
    """
    image_bytes = await _read_image(file)
    return await defense_service.run_analysis(
        image_bytes=image_bytes,
        db=db,
        user_id=current_user.id,
    )


@router.post("/attack-test", response_model=AttackTestResponse)
async def attack_test(
    file: UploadFile = File(..., description="Handwritten digit image (PNG/JPEG)"),
    attack_type: str = Form(default="fgsm", description="fgsm | pgd | eot_pgd"),
    epsilon: float = Form(default=0.20, ge=0.0, le=1.0, description="Perturbation strength"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate an adversarial example from the uploaded image using the
    specified attack, then run the full defense pipeline on it.
    Requires authentication.
    """
    if attack_type not in ("fgsm", "pgd", "eot_pgd"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="attack_type must be one of: fgsm, pgd, eot_pgd",
        )
    image_bytes = await _read_image(file)
    return await defense_service.run_attack_test(
        image_bytes=image_bytes,
        attack_type=attack_type,
        epsilon=epsilon,
        db=db,
        user_id=current_user.id,
    )
