from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from backend/.env.
    All fields map 1-to-1 with the keys in .env.example.
    """

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── App ───────────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    CORS_ORIGINS: str = "http://localhost:5173"

    # ── ML paths (relative to the repo root, two levels up from backend/app) ──
    MODEL_PATH: Path = (
        Path(__file__).resolve().parents[2] / "models" / "baseline_cnn.pth"
    )
    NUM_CLASSES: int = 10

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]


settings = Settings()
