# RealTime Anomaly Defense — Project Architecture & Full Handoff Guide

## 1. Project Overview & System Purpose

**RealTime Anomaly Defense** is a production-ready, full-stack ML security application that detects, purifies, and mitigates adversarial attack perturbations on computer vision models (specifically MNIST digit classification).

The system defends against adversarial attacks (**FGSM**, **PGD**, **EOT-PGD**, **C&W**) through a 5-stage pipeline: anomaly detection → self-purification → randomized defense → gradient diversity → consensus. This pipeline is now served via a **FastAPI** REST backend with **PostgreSQL** persistence and a **React + Vite** frontend.

---

## 2. Current Project Status

| Component | Status |
|-----------|--------|
| ML Pipeline (`src/`) | ✅ Complete — all 5 defense stages implemented |
| FastAPI Backend (`backend/`) | ✅ Running on port 8000 |
| React Frontend (`frontend/`) | ✅ Running on port 5173 |
| PostgreSQL + Alembic Migrations | ✅ Tables created |
| Authentication / Access control | ✅ Login required to analyze — JWT enforced on `/api/defense/*`; frontend gates upload/Analyze/attack actions + guards `/history` |
| Analytics DB Seeding | ❌ Pending — `benchmark_results` table empty |

---

## 3. Directory Structure

```text
RealTime-Anomaly-Defense/
│
├── app.py                          # Legacy Streamlit app (still functional for local demos)
├── config.py                       # Streamlit/legacy global config (DEVICE, paths, BATCH_SIZE)
├── requirements.txt                # Root-level legacy requirements (Streamlit stack)
│
├── src/                            # Core ML modules — imported directly by backend
│   ├── __init__.py
│   ├── models/
│   │   ├── cnn.py                  # MNISTCNN — 2 conv layers, dropout, ~98.94% clean accuracy
│   │   ├── cifar10_cnn.py          # CIFAR-10 variant (extended experiments)
│   │   └── train_cnn.py            # Training loop + checkpoint saving
│   ├── detection/
│   │   └── anomaly_detector.py     # DynamicAnomalyDetector — confidence + pooling variance
│   ├── purification/
│   │   ├── purifier.py             # SelfPurifier — 4-candidate best-confidence selection
│   │   └── dip_purifier.py         # DIP-inspired (Deep Image Prior) purification variant
│   ├── defenses/
│   │   ├── randomized_defense.py   # N=4 Monte Carlo Gaussian noise ensemble (σ=0.03)
│   │   ├── gradient_diversity.py   # Gradient consistency across structural transforms
│   │   ├── defense_pipeline.py     # Abstract sequential pipeline base class
│   │   └── combined_defense.py     # Orchestrator: integrates all 4 defense stages
│   ├── attacks/
│   │   ├── fgsm.py                 # Fast Gradient Sign Method
│   │   ├── pgd.py                  # Projected Gradient Descent (multi-step)
│   │   ├── eot_pgd.py              # Expectation Over Transformation PGD
│   │   ├── cw.py                   # Carlini-Wagner L2 attack
│   │   └── evaluate_attacks.py     # Benchmark runner (prints + saves JSON; no DB write yet)
│   ├── data/
│   │   ├── mnist_loader.py         # IDX-format MNIST loader
│   │   ├── preprocessing.py        # MNISTDataset torch.utils.data.Dataset wrapper
│   │   ├── test_dataset.py         # Dataset integrity tests
│   │   └── visualize_dataset.py    # Sample visualization helper
│   ├── evaluation/                 # 13 evaluation scripts
│   │   ├── ablation_study.py
│   │   ├── evaluate_cifar.py
│   │   ├── evaluate_cifar_attacks.py
│   │   ├── evaluate_cw.py
│   │   ├── evaluate_cw_defense.py
│   │   ├── evaluate_defense.py
│   │   ├── evaluate_eot.py
│   │   ├── evaluate_full_defense.py
│   │   ├── evaluate_graddiv.py
│   │   ├── evaluate_graddiv_defense.py
│   │   ├── generate_final_figures.py
│   │   ├── generate_results.py
│   │   └── test_dip.py
│   └── utils/
│       └── __init__.py             # General helpers placeholder
│
├── models/                         # Trained PyTorch checkpoints
│   ├── baseline_cnn.pth            # Primary model — used by the FastAPI backend
│   ├── cifar10_cnn.pth             # CIFAR-10 extended model
│   └── graddiv_cnn.pth             # Gradient-diversity regularized MNIST model
│
├── data/
│   ├── raw/                        # MNIST IDX files (auto-downloaded)
│   ├── processed/                  # Preprocessed tensors
│   └── cifar10/                    # CIFAR-10 dataset
│
├── experiments/
│   ├── figures/                    # Matplotlib benchmark charts (.png)
│   ├── logs/                       # Training and evaluation logs
│   └── results/                    # JSON / CSV benchmarking metrics
│
├── backend/                        # FastAPI + PostgreSQL — PRODUCTION STACK
│   ├── requirements.txt            # asyncpg, fastapi, sqlalchemy[asyncio], alembic, jose, passlib, torch
│   ├── .env.example                # Template for backend/.env
│   ├── alembic.ini
│   ├── alembic/
│   │   └── env.py                  # Async Alembic runner (run_async_migrations)
│   └── app/
│       ├── __init__.py
│       ├── main.py                 # App factory: lifespan + CORS + router registration
│       ├── config.py               # Pydantic Settings (DATABASE_URL, JWT_SECRET_KEY, MODEL_PATH…)
│       ├── database.py             # AsyncEngine + async_sessionmaker + get_db() dependency
│       ├── models.py               # ORM: User, AnomalyAnalysisLog, BenchmarkResult
│       ├── schemas.py              # Pydantic v2 schemas: all request/response contracts
│       ├── auth/
│       │   ├── __init__.py
│       │   ├── security.py         # passlib bcrypt hash_password / verify_password
│       │   └── jwt_handler.py      # create_access_token, decode_access_token, get_current_user
│       ├── routers/
│       │   ├── __init__.py
│       │   ├── auth_routes.py      # POST /api/auth/register|login  GET /api/auth/me
│       │   ├── defense_routes.py   # POST /api/defense/analyze|attack-test (both require JWT)
│       │   ├── experiments_routes.py # GET /api/experiments/results
│       │   └── history_routes.py   # GET /api/history  GET /api/history/{log_id}
│       └── services/
│           ├── __init__.py
│           └── defense_service.py  # Async bridge: FastAPI ↔ src/ ML pipeline
│
└── frontend/                       # React 18 + Vite 5 + Tailwind 3 — PRODUCTION STACK
    ├── package.json
    ├── vite.config.js              # Dev server port 5173; proxy /api → :8000
    ├── tailwind.config.js          # Brand palette (indigo-500/600/700), semantic colors, fonts
    ├── postcss.config.js
    ├── index.html
    ├── .env                        # VITE_API_BASE_URL (blank = use proxy)
    └── src/
        ├── main.jsx                # Root: BrowserRouter + AuthProvider + Toaster
        ├── App.jsx                 # Sticky NavBar + Routes; RequireAuth guards /history (Dashboard is a public landing page, gates analysis actions inline)
        ├── index.css               # @tailwind + .glass, .gradient-text, .glass-hover utilities
        ├── api/
        │   └── client.js           # Axios: JWT header injection + 401 auto-redirect
        ├── context/
        │   └── AuthContext.jsx     # {user, token, login, register, logout, isLoading} + /me rehydration
        ├── components/
        │   ├── RequireAuth.jsx     # Route guard: redirect to /login when signed out (protects /history)
        │   ├── ImageUploader.jsx   # react-dropzone with image preview
        │   ├── MetricCard.jsx      # Glass card: label + value + highlight variant
        │   ├── DefenseStatusBadge.jsx # Anomaly/clean status + agreement score
        │   ├── PipelineStepTracker.jsx # Animated 5-step pipeline progress
        │   └── EpsilonSlider.jsx   # Range input for ε with live numeric readout
        └── pages/
            ├── LoginPage.jsx
            ├── RegisterPage.jsx
            ├── DashboardPage.jsx   # Live Detection + Adversarial Attack Test panel
            ├── AnalyticsPage.jsx   # Benchmark bar chart (Recharts)
            └── HistoryPage.jsx     # Paginated inference history table
```

---

## 4. Core Technical Architecture & Data Flow

### 4.1 ML Defense Pipeline (5 Stages)

```mermaid
flowchart TD
    Upload[Upload Image bytes] --> Pre[Preprocess: grayscale→invert→28×28→0..1]
    Pre --> CNN[MNISTCNN forward pass]
    CNN --> Det[1. DynamicAnomalyDetector]

    Det -->|score > 0.037860| Anom{Anomalous?}

    Anom -->|Yes| Pur[2. SelfPurifier\n4 candidates → best confidence]
    Anom -->|No| Pass[Pass through unchanged]

    Pur --> Rand
    Pass --> Rand

    Rand[3. RandomizedDefense\nN=4 MC runs σ=0.03] --> Cons
    Rand --> Grad[4. GradientDiversityDefense\nstructural transform consistency]
    Grad --> Cons[5. CombinedDefense\nconsensus / agreement check]

    Cons --> Out[Final Prediction + Defense Metrics]
    Out --> DB[(PostgreSQL\nAnomalyAnalysisLog)]
    Out --> API[JSON Response → React UI]
```

### 4.2 Request Flow (FastAPI → src/ → DB)

```
Browser (React)
    │  multipart/form-data  POST /api/defense/analyze   (Authorization: Bearer <JWT> required)
    ▼
defense_routes.py
    │  await defense_service.run_analysis(image_bytes, db, user_id)
    ▼
defense_service.py  (async, singleton pattern)
    │  _preprocess() → _load_once() (model + all defense singletons)
    │  _detector.detect()
    │  _purifier.purify() [if anomalous]
    │  _randomized.predict()
    │  _gradient.predict()
    │  _combined.predict()
    ▼
AnomalyAnalysisLog row  →  db.flush()
    ▼
AnalyzeResponse (Pydantic)  →  JSON
    ▼
React: MetricCards + DefenseStatusBadge
```

### 4.3 Auth Flow

```
POST /api/auth/login  {email, password}
    → bcrypt verify → create_access_token (JWT, HS256, 60 min)
    → {access_token, token_type:"bearer"}

localStorage.setItem('access_token', token)
axios interceptor → Authorization: Bearer <token> on every request

GET /api/auth/me  → AuthContext rehydrates user on page refresh
GET /api/history  → get_current_user dependency decodes JWT → User ORM
POST /api/defense/analyze | attack-test → get_current_user REQUIRED (401/403 without a valid Bearer token); each analysis is logged to the caller's account
```

**Access model (client-side):** The Dashboard is a public landing page — it renders for anyone. Authentication is enforced at the *action* level, not by blocking the route: when a signed-out visitor tries to upload an image, click **Analyze**, or run an **attack test**, `DashboardPage` shows a toast and redirects to `/login` (`requireLogin()`). The `/history` route is fully guarded by the `RequireAuth` wrapper in `App.jsx`. The backend enforces the same rule independently, so the API is safe even if the UI is bypassed.

---

## 5. Key Module Details

### 5.1 MNISTCNN (`src/models/cnn.py`)

Architecture: Conv2d(1→32) → ReLU → MaxPool → Conv2d(32→64) → ReLU → MaxPool → Dropout(0.25) → Linear(64×7×7→128) → ReLU → Dropout(0.5) → Linear(128→10)

Clean accuracy: **~98.94%** on MNIST test set.

### 5.2 DynamicAnomalyDetector (`src/detection/anomaly_detector.py`)

Anomaly score = weighted combination:
- `low_confidence = 1.0 - max_softmax_prob` (weight: 0.50)
- `prediction_change = (original_pred ≠ avg_pool_pred)` (weight: 0.30)
- `confidence_change = |original_conf - avg_pool_conf|` (weight: 0.20)

Pre-calibrated threshold = **0.037860**.

### 5.3 SelfPurifier (`src/purification/purifier.py`)

Generates 4 candidate variants: Original · 3×3 AvgPool · 5×5 AvgPool · Denoised (0.7×smoothed + 0.3×original). Selects candidate with highest softmax confidence.

### 5.4 RandomizedDefense (`src/defenses/randomized_defense.py`)

N=4 Monte Carlo forward passes with Gaussian noise (σ=0.03). Averages softmax probabilities → most-voted prediction.

### 5.5 GradientDiversityDefense (`src/defenses/gradient_diversity.py`)

Tests prediction consistency across structural transforms (horizontal flip, small rotation, resize). Reports `agreement` ratio; flags low-agreement inputs as suspect.

### 5.6 CombinedDefense (`src/defenses/combined_defense.py`)

Orchestrates all 4 stages. Reaches consensus between `RandomizedDefense.predict()` and `GradientDiversityDefense.predict()`. Returns `{predictions, agreement}`.

---

## 6. PostgreSQL Schema (Implemented)

```sql
-- users
id UUID PK | email VARCHAR(255) UNIQUE | password_hash VARCHAR(255) | created_at TIMESTAMPTZ

-- anomaly_analysis_logs
id UUID PK | user_id UUID FK (SET NULL)
| clean_prediction INT | clean_confidence FLOAT
| is_anomalous BOOL | anomaly_score FLOAT
| is_fgsm_attacked BOOL | attack_type VARCHAR(20) | attack_epsilon FLOAT
| adversarial_prediction INT | purified_prediction INT
| randomized_prediction INT | gradient_prediction INT        -- ← added vs original schema
| final_defended_prediction INT | defense_agreement_score FLOAT
| execution_time_ms FLOAT | created_at TIMESTAMPTZ

-- benchmark_results  (currently empty — needs seeding)
id UUID PK | model_name VARCHAR(100) | defense_type VARCHAR(100)
| attack_type VARCHAR(100) | epsilon FLOAT | accuracy FLOAT
| detection_rate FLOAT | created_at TIMESTAMPTZ
```

---

## 7. API Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/register` | ❌ | `{email, password}` → `UserResponse` |
| POST | `/api/auth/login` | ❌ | `{email, password}` → `TokenResponse` |
| GET | `/api/auth/me` | ✅ | Current user info |
| POST | `/api/defense/analyze` | ✅ | `file` (multipart) → `AnalyzeResponse`; logged to user account |
| POST | `/api/defense/attack-test` | ✅ | `file + attack_type + epsilon` → `AttackTestResponse` |
| GET | `/api/history` | ✅ | `?page=1&page_size=20` → `HistoryResponse` |
| GET | `/api/history/{log_id}` | ✅ | `HistoryItem` |
| GET | `/api/experiments/results` | ❌ | `ExperimentsResponse` (empty until seeded) |
| GET | `/health` | ❌ | `{status: "ok", model_loaded: bool}` |

---

## 8. Environment Configuration

### `backend/.env` (required, copy from `.env.example`)

```bash
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/anomaly_defense
JWT_SECRET_KEY=<generate: python -c "import secrets; print(secrets.token_hex(32))">
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
APP_ENV=development
CORS_ORIGINS=http://localhost:5173
```

### `frontend/.env`

```bash
VITE_API_BASE_URL=   # Leave blank to use Vite proxy (recommended for local dev)
```

---

## 9. Running the Stack

```bash
# Terminal 1 — Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

- Backend API docs: http://localhost:8000/docs
- Frontend app:     http://localhost:5173

---

## 10. Pending Work

| Item | Priority | Details |
|------|----------|---------|
| Seed `benchmark_results` | High | Modify `src/attacks/evaluate_attacks.py` to insert rows into PostgreSQL at run-end, OR create `backend/seed_benchmarks.py` that reads `experiments/results/*.json` and bulk-inserts |
| `image_url` storage | Low | Original schema included image storage; current implementation omits it — add S3/local file storage if image audit trail is required |
| Docker Compose | Optional | Bundle Postgres + Backend + Frontend for one-command deployment |
| Production deployment | Optional | Render/Railway (backend + DB) + Vercel (frontend) |
