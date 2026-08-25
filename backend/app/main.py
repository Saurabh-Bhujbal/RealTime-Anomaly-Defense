"""
FastAPI application entrypoint.

Run locally:
    cd backend
    uvicorn app.main:app --reload --port 8000

API docs (auto-generated):
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services import defense_service

# ── Routers ───────────────────────────────────────────────────────────────────
from app.routers import (
    auth_routes,
    defense_routes,
    experiments_routes,
    history_routes,
)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Load the PyTorch model once at startup so the first request is not slow.
    All defence components (detector, purifier, etc.) are also initialised here.
    """
    defense_service._load_once()
    yield
    # Nothing to tear down for an in-process PyTorch model


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RealTime Anomaly Defense API",
    description=(
        "REST API exposing the adversarial defense pipeline "
        "(DynamicAnomalyDetector → SelfPurifier → RandomizedDefense "
        "→ GradientDiversity → CombinedDefense) as HTTP endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(auth_routes.router)
app.include_router(defense_routes.router)
app.include_router(experiments_routes.router)
app.include_router(history_routes.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model_loaded": defense_service._model is not None}
