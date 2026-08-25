# RealTime Anomaly Defense — Full-Stack Implementation Guide
### React (Frontend) + FastAPI (Backend) + PostgreSQL (Database)
> **Status as of 2026-08-25:** Backend and Frontend are both RUNNING. This guide has been updated to reflect the **actual implemented code**, not the original scaffold instructions.
>
> **Login is now required to analyze images.** The Dashboard still loads publicly as the landing page, but any analysis action — uploading an image, clicking **Analyze**, or running an **attack test** — requires a signed-in user and redirects to `/login` when signed out. The backend enforces this independently: `/api/defense/analyze` and `/api/defense/attack-test` both require a valid Bearer JWT.

---

## Table of Contents

1. [Actual Folder Structure](#1-actual-folder-structure)
2. [Prerequisites & Environment](#2-prerequisites--environment)
3. [Part A — Backend (FastAPI + PostgreSQL)](#3-part-a--backend-fastapi--postgresql)
4. [Part B — Frontend (React + Vite + Tailwind)](#4-part-b--frontend-react--vite--tailwind)
5. [Frontend ↔ Backend Connection](#5-frontend--backend-connection)
6. [End-to-End Test Checklist](#6-end-to-end-test-checklist)
7. [Remaining Work](#7-remaining-work)
8. [Docker & Deployment Notes](#8-docker--deployment-notes)

---

## 1. Actual Folder Structure

```text
RealTime-Anomaly-Defense/
├── src/                          ← UNCHANGED ML code
│   ├── models/                   cnn.py · cifar10_cnn.py · train_cnn.py
│   ├── detection/                anomaly_detector.py
│   ├── purification/             purifier.py · dip_purifier.py
│   ├── defenses/                 randomized_defense.py · gradient_diversity.py
│   │                             defense_pipeline.py · combined_defense.py
│   ├── attacks/                  fgsm.py · pgd.py · eot_pgd.py · cw.py
│   │                             evaluate_attacks.py
│   ├── data/                     mnist_loader.py · preprocessing.py
│   └── evaluation/               13 evaluation scripts
│
├── models/                       baseline_cnn.pth ← used by API
│                                 graddiv_cnn.pth · cifar10_cnn.pth
├── data/                         raw/ · processed/ · cifar10/
├── experiments/                  figures/ · logs/ · results/
├── app.py                        Legacy Streamlit app (still works)
├── config.py                     Legacy config
│
├── backend/                      ← IMPLEMENTED FastAPI stack
│   ├── requirements.txt
│   ├── .env.example
│   ├── alembic.ini
│   ├── alembic/env.py            Async-compatible migration runner
│   └── app/
│       ├── main.py               App factory + lifespan + CORS
│       ├── config.py             Pydantic Settings (reads backend/.env)
│       ├── database.py           AsyncEngine + get_db() dependency
│       ├── models.py             ORM: User, AnomalyAnalysisLog, BenchmarkResult
│       ├── schemas.py            All Pydantic v2 request/response schemas
│       ├── auth/
│       │   ├── security.py       bcrypt hashing
│       │   └── jwt_handler.py    JWT create/decode + get_current_user
│       ├── routers/
│       │   ├── auth_routes.py    register · login · /me
│       │   ├── defense_routes.py analyze · attack-test  (both require JWT)
│       │   ├── experiments_routes.py  /results (empty — needs seeding)
│       │   └── history_routes.py /history + /{log_id} with pagination
│       └── services/
│           └── defense_service.py  Async bridge: FastAPI ↔ src/ ML
│
└── frontend/                     ← IMPLEMENTED React stack
    ├── package.json
    ├── vite.config.js            Port 5173 + proxy /api → :8000
    ├── tailwind.config.js        Brand palette + Inter/JetBrains fonts
    └── src/
        ├── main.jsx              BrowserRouter + AuthProvider + Toaster
        ├── App.jsx               NavBar + Routes (RequireAuth guards /history)
        ├── index.css             Tailwind + .glass · .gradient-text utilities
        ├── api/client.js         Axios + JWT interceptor + 401 redirect
        ├── context/AuthContext.jsx  login · register · logout · /me rehydration
        ├── components/           RequireAuth · ImageUploader · MetricCard · DefenseStatusBadge
        │                         PipelineStepTracker · EpsilonSlider
        └── pages/                LoginPage · RegisterPage · DashboardPage
                                  AnalyticsPage · HistoryPage
```

---

## 2. Prerequisites & Environment

| Tool | Required Version | Check |
|------|-----------------|-------|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| PostgreSQL | 14+ | `psql --version` |

### Database — run once

```bash
psql -U postgres
CREATE DATABASE anomaly_defense;
CREATE USER anomaly_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE anomaly_defense TO anomaly_user;
\q
```

---

## 3. Part A — Backend (FastAPI + PostgreSQL)

### Step 1 — `backend/.env` (copy from `.env.example`)

```bash
DATABASE_URL=postgresql+asyncpg://anomaly_user:your_password@localhost:5432/anomaly_defense
JWT_SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
APP_ENV=development
CORS_ORIGINS=http://localhost:5173
```

> **Key difference from original guide:** Driver is `asyncpg` (async), env var is `JWT_SECRET_KEY` (not `SECRET_KEY`), and CORS is `CORS_ORIGINS` (not `FRONTEND_ORIGIN`).

### Step 2 — `backend/requirements.txt` (actual)

```text
fastapi==0.115.6
uvicorn[standard]==0.32.1
sqlalchemy==2.0.36
alembic==1.14.0
psycopg2-binary==2.9.12      # used by Alembic offline mode
asyncpg                       # async runtime driver
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9
pydantic-settings==2.6.1
pillow==10.4.0
torch>=2.0.0
numpy>=1.24.0
python-dotenv==1.0.1
torchvision
```

### Step 3 — `backend/app/config.py` (actual)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    APP_ENV: str = "development"
    CORS_ORIGINS: str = "http://localhost:5173"
    MODEL_PATH: Path = Path(__file__).resolve().parents[2] / "models" / "baseline_cnn.pth"
    NUM_CLASSES: int = 10

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

settings = Settings()
```

### Step 4 — `backend/app/database.py` (actual — fully async)

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.APP_ENV == "development", pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### Step 5 — `backend/app/models.py` (actual — extended schema)

```python
import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    analysis_logs: Mapped[list["AnomalyAnalysisLog"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class AnomalyAnalysisLog(Base):
    __tablename__ = "anomaly_analysis_logs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    clean_prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    clean_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomalous: Mapped[bool] = mapped_column(Boolean, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_fgsm_attacked: Mapped[bool] = mapped_column(Boolean, default=False)
    attack_type: Mapped[str | None] = mapped_column(String(20), nullable=True)   # ← added
    attack_epsilon: Mapped[float | None] = mapped_column(Float, nullable=True)
    adversarial_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)
    purified_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)
    randomized_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)  # ← added
    gradient_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)    # ← added
    final_defended_prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    defense_agreement_score: Mapped[float] = mapped_column(Float, nullable=False)
    execution_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped["User | None"] = relationship(back_populates="analysis_logs")

class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    defense_type: Mapped[str] = mapped_column(String(100), nullable=False)
    attack_type: Mapped[str] = mapped_column(String(100), nullable=False)
    epsilon: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    detection_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

### Step 6 — Alembic (async-compatible)

```bash
cd backend
# Already initialised. Just run migrations:
alembic upgrade head
```

`alembic/env.py` uses `run_async_migrations()` with `create_async_engine` — it reads `settings.DATABASE_URL` automatically.

### Step 7 — Auth (`auth/security.py` + `auth/jwt_handler.py`)

Uses `HTTPBearer` scheme (not `OAuth2PasswordBearer`). Login accepts JSON body `{email, password}` — not form-encoded.

```python
# jwt_handler.py key difference — HTTPBearer not OAuth2
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
_bearer = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer), db=Depends(get_db)) -> User:
    user_id_str = decode_access_token(credentials.credentials)
    ...
```

### Step 8 — `services/defense_service.py` (actual — fully wired)

All ML classes are real (no `# ⚠️ VERIFY` placeholders). Key facts:
- Pre-calibrated detector threshold: `0.037860`
- Preprocessing inverts image (`ImageOps.invert`) to match MNIST white-on-black format
- Supports 3 attack types: `fgsm`, `pgd`, `eot_pgd`
- Returns `randomized_prediction`, `gradient_prediction` in addition to guide's fields
- Both `run_analysis()` and `run_attack_test()` are **async**

```python
_combined = CombinedDefense(
    model=_model, detector=_detector, purifier=_purifier,
    randomized_defense=_randomized, gradient_defense=_gradient,
)
```

### Step 9 — Routers (actual endpoints)

| Router | Endpoints |
|--------|-----------|
| `auth_routes.py` | POST `/api/auth/register` · POST `/api/auth/login` · **GET `/api/auth/me`** |
| `defense_routes.py` | POST `/api/defense/analyze` (**JWT required**) · POST `/api/defense/attack-test` (**JWT required**; supports `fgsm`\|`pgd`\|`eot_pgd`) |
| `history_routes.py` | GET `/api/history?page=1&page_size=20` · **GET `/api/history/{log_id}`** |
| `experiments_routes.py` | GET `/api/experiments/results` |

### Step 10 — `app/main.py` (actual)

```python
@asynccontextmanager
async def lifespan(application: FastAPI):
    defense_service._load_once()   # preload model at startup
    yield

app = FastAPI(title="RealTime Anomaly Defense API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins_list, ...)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": defense_service._model is not None}
```

### Step 11 — Run the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## 4. Part B — Frontend (React + Vite + Tailwind)

### Key differences from original guide

| Item | Original Guide | Actual Implementation |
|------|---------------|----------------------|
| Extra packages | axios, recharts, lucide-react | + `react-dropzone`, `react-hot-toast` |
| Tailwind theme | Default extend | Custom `brand` palette (indigo), semantic colors, Inter + JetBrains Mono fonts |
| `index.css` | `@tailwind` only | + `.glass`, `.gradient-text`, `.glass-hover` utility classes |
| Vite config | Not configured | Dev proxy `/api → :8000` (avoids CORS preflight) |
| `client.js` | JWT attach only | + 401 response interceptor → clears token + redirects |
| `AuthContext` | login/logout | + `register()`, `isLoading`, `/me` rehydration on mount, stores full `user` object |
| `DashboardPage` | 2 attack types | 3 attack types (fgsm · pgd · **eot_pgd**); shows `randomized_prediction` + `gradient_prediction` |
| `HistoryPage` | Simple list | Paginated (`page`/`page_size`); fetches `HistoryResponse{total, items}` |
| `App.jsx` | Basic nav | Sticky navbar with lucide-react icons + user email display + loading spinner; `/history` wrapped in `RequireAuth` |
| Access control | None | `RequireAuth` guards `/history`; Dashboard loads publicly but its upload/Analyze/attack actions require login (redirect to `/login`) |
| `main.jsx` | BrowserRouter | + `Toaster` (react-hot-toast) with dark theme |

### Step 1 — `frontend/.env`

```bash
VITE_API_BASE_URL=   # Leave blank — Vite proxy handles /api/* routing
```

### Step 2 — `frontend/vite.config.js` (proxy — no CORS issues in dev)

```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } }
  }
})
```

### Step 3 — Run the frontend

```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

---

## 5. Frontend ↔ Backend Connection

| Setting | Value | Location |
|---------|-------|----------|
| API proxy | `/api` → `http://localhost:8000` | `vite.config.js` |
| JWT storage | `localStorage['access_token']` | `AuthContext.jsx` |
| JWT attach | `Authorization: Bearer <token>` | `api/client.js` interceptor |
| 401 handling | Clear token + redirect `/login` | `api/client.js` interceptor |
| Route guard | `RequireAuth` → redirect `/login` if signed out | `components/RequireAuth.jsx` (protects `/history`) |
| Action gate | Upload/Analyze/attack while signed out → toast + redirect `/login` | `pages/DashboardPage.jsx` (`requireLogin()`) |
| CORS backend | `CORS_ORIGINS=http://localhost:5173` | `backend/.env` |

---

## 6. End-to-End Test Checklist

- [x] `POST /api/auth/register` → creates user (check `/docs`)
- [x] `POST /api/auth/login` → returns JWT
- [x] `GET /api/auth/me` → returns user object
- [x] React `/register` + `/login` pages work
- [x] Dashboard loads while signed out (public landing page)
- [x] Signed out: clicking the upload zone / **Analyze** / **attack test** shows a toast and redirects to `/login`
- [x] Signed out: `/history` redirects to `/login` (RequireAuth)
- [x] `POST /api/defense/analyze` without a Bearer token → rejected (403/401)
- [x] Dashboard: upload digit image → clean analysis returns prediction, confidence, anomaly score, agreement
- [x] Dashboard: run FGSM/PGD/EOT_PGD attack test → adversarial + defended predictions shown
- [x] `GET /api/history` shows paginated logs for logged-in user
- [x] `GET /api/history/{id}` returns single log
- [x] Logout clears token and redirects to `/login`
- [x] `GET /health` returns `{"status":"ok","model_loaded":true}`
- [ ] Analytics page shows data (blocked — `benchmark_results` table empty)

---

## 7. Remaining Work

### ❌ Seed `benchmark_results` for Analytics page

The `AnalyticsPage.jsx` reads from `GET /api/experiments/results` which returns an empty list because `src/attacks/evaluate_attacks.py` only prints/saves to JSON — it does not write to PostgreSQL.

**Option A — Modify `evaluate_attacks.py`:** Add at the end:

```python
# After computing results, insert into DB
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import sys
sys.path.insert(0, "backend")
from app.config import settings
from app.models import BenchmarkResult

async def seed_results(rows):
    engine = create_async_engine(settings.DATABASE_URL)
    async with AsyncSession(engine) as session:
        for row in rows:
            session.add(BenchmarkResult(**row))
        await session.commit()
    await engine.dispose()

asyncio.run(seed_results(your_results_list))
```

**Option B — Standalone seed script:** Create `backend/seed_benchmarks.py` that reads from `experiments/results/*.json` and bulk-inserts.

---

## 8. Docker & Deployment Notes

```yaml
# docker-compose.yml (optional, for production)
version: "3.9"
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: anomaly_defense
      POSTGRES_USER: anomaly_user
      POSTGRES_PASSWORD: change_this_password
    ports: ["5432:5432"]
    volumes: [db_data:/var/lib/postgresql/data]

  backend:
    build: ./backend
    env_file: ./backend/.env
    ports: ["8000:8000"]
    depends_on: [db]

  frontend:
    build: ./frontend
    ports: ["5173:80"]
    depends_on: [backend]

volumes:
  db_data:
```

**Production hosting:**
- Backend + DB: **Render** or **Railway** (set `DATABASE_URL` env var in dashboard)
- Frontend: **Vercel** or **Netlify** (set `VITE_API_BASE_URL=https://your-backend-url.com`)
