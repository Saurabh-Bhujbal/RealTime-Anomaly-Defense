# RealTime Anomaly Defense

**Real-Time Dynamic Anomaly Detection and Self-Purifying Defensive Layers for Autonomous Visual Systems**

A production-ready, full-stack ML security application that detects, purifies, and mitigates adversarial attacks on computer vision models — featuring a **FastAPI** async backend, **React + Vite + Tailwind** frontend, **PostgreSQL** database, and a 5-stage defense pipeline built on PyTorch.

---

## 🏗️ Architecture Overview

```
React Frontend (Vite + Tailwind)   ←→   FastAPI Backend (Async)   ←→   PostgreSQL
        :5173                                  :8000                   asyncpg
                                                 │
                                          src/ ML Pipeline
                              (MNISTCNN → Detector → Purifier →
                               Randomized → GradientDiversity → CombinedDefense)
```

---

## 📁 Project Structure

```text
RealTime-Anomaly-Defense/
├── app.py                          # Legacy Streamlit app (still functional)
├── config.py                       # Streamlit/legacy project config
├── requirements.txt                # Root-level legacy requirements
│
├── src/                            # Core ML modules (unchanged, reused by backend)
│   ├── models/
│   │   ├── cnn.py                  # MNISTCNN — 2 conv + FC + dropout (~98.94% clean accuracy)
│   │   ├── cifar10_cnn.py          # CIFAR-10 CNN variant
│   │   ├── gmdcn.py                # GMDCN — Deeper CNN parameterizable for gradient manipulation
│   │   ├── train_cnn.py            # Training script
│   │   ├── train_fashion_mnist.py  # Fashion-MNIST training script
│   │   └── train_gmdcn.py          # GMDCN training script with manipulation modes
│   ├── detection/
│   │   └── anomaly_detector.py     # DynamicAnomalyDetector (threshold = 0.037860)
│   ├── purification/
│   │   ├── purifier.py             # SelfPurifier — 4-candidate confidence-based selection
│   │   └── dip_purifier.py         # DIP-inspired purification variant
│   ├── defenses/
│   │   ├── randomized_defense.py   # Gaussian noise Monte Carlo ensemble (N=4, σ=0.03)
│   │   ├── gradient_diversity.py   # Gradient consistency across structural transforms
│   │   ├── defense_pipeline.py     # Abstract sequential pipeline base
│   │   └── combined_defense.py     # Orchestrator: Detector→Purifier→Rand→Grad→Consensus
│   ├── attacks/
│   │   ├── fgsm.py                 # Fast Gradient Sign Method
│   │   ├── pgd.py                  # Projected Gradient Descent
│   │   ├── eot_pgd.py              # Expectation Over Transformation PGD
│   │   ├── cw.py                   # Carlini-Wagner attack
│   │   └── evaluate_attacks.py     # Benchmark runner (prints results; DB seeding pending)
│   ├── data/
│   │   ├── mnist_loader.py         # MNIST IDX loader
│   │   ├── preprocessing.py        # MNISTDataset PyTorch Dataset
│   │   ├── cifar10_dataset.py      # CIFAR-10 Dataset wrapper
│   │   ├── test_dataset.py         # Dataset validation helper
│   │   └── visualize_dataset.py    # Sample visualization
│   ├── evaluation/                 # 13 evaluation scripts (ablation, CIFAR, CW, EOT, Fashion-MNIST, GMDCN, etc.)
│   ├── optimization/               # Swarm Optimization (Step Groups 4-5)
│   │   ├── ssa.py                  # Salp Swarm Algorithm (SSA)
│   │   ├── cuckoo_search.py        # Cuckoo Search Algorithm
│   │   ├── osprey.py               # Osprey Optimization Algorithm (OOA)
│   │   ├── ssa_tune_gmdcn.py       # Tune GMDCN hyperparams via SSA
│   │   └── compare_optimizers.py   # SSA vs Cuckoo vs Osprey benchmarking
│   └── utils/                      # General helpers
│
├── models/                         # Trained PyTorch checkpoints
│   ├── baseline_cnn.pth            # Main MNIST model (used by backend)
│   ├── cifar10_cnn.pth             # CIFAR-10 model
│   ├── fashion_mnist_cnn.pth       # Fashion-MNIST model
│   ├── graddiv_cnn.pth             # Gradient-diversity trained model
│   └── gmdcn_cnn_clip.pth          # GMDCN model trained with gradient clipping
│
├── data/                           # Dataset root
│   ├── raw/                        # MNIST IDX files
│   ├── processed/                  # Preprocessed tensors
│   └── cifar10/                    # CIFAR-10 dataset
│
├── experiments/                    # Benchmarking outputs
│   ├── figures/                    # Matplotlib charts (.png)
│   ├── logs/                       # Training/evaluation logs
│   └── results/                    # JSON/CSV benchmark metrics
│
├── backend/                        # FastAPI + PostgreSQL (PRODUCTION)
│   ├── requirements.txt
│   ├── .env.example
│   ├── alembic.ini
│   ├── alembic/
│   │   └── env.py                  # Async-compatible Alembic runner
│   └── app/
│       ├── main.py                 # FastAPI app — lifespan + routers + CORS
│       ├── config.py               # Pydantic Settings (reads .env)
│       ├── database.py             # Async SQLAlchemy engine + session factory
│       ├── models.py               # ORM: User, AnomalyAnalysisLog, BenchmarkResult
│       ├── schemas.py              # Pydantic schemas: request/response contracts
│       ├── auth/
│       │   ├── security.py         # bcrypt password hashing
│       │   └── jwt_handler.py      # JWT create/decode + get_current_user dependency
│       ├── routers/
│       │   ├── auth_routes.py      # POST /api/auth/register|login  GET /api/auth/me
│       │   ├── defense_routes.py   # POST /api/defense/analyze|attack-test (JWT required)
│       │   ├── experiments_routes.py # GET /api/experiments/results
│       │   └── history_routes.py   # GET /api/history  GET /api/history/{id}
│       └── services/
│           └── defense_service.py  # Bridge: FastAPI ↔ src/ ML pipeline (async singletons)
│
└── frontend/                       # React + Vite + Tailwind (PRODUCTION)
    ├── package.json
    ├── vite.config.js              # Dev proxy: /api → localhost:8000
    ├── tailwind.config.js          # Brand palette (indigo), Inter/JetBrains Mono fonts
    ├── postcss.config.js
    ├── index.html
    ├── .env                        # VITE_API_BASE_URL (empty = use proxy)
    └── src/
        ├── main.jsx                # ReactDOM root + BrowserRouter + AuthProvider + Toaster
        ├── App.jsx                 # Routes + sticky NavBar (RequireAuth guards /history)
        ├── index.css               # Tailwind base + glass/gradient-text utilities
        ├── api/
        │   └── client.js           # Axios instance — JWT interceptor + 401 redirect
        ├── context/
        │   └── AuthContext.jsx     # login/register/logout + /me rehydration on mount
        ├── components/
        │   ├── RequireAuth.jsx     # Route guard — redirects to /login when signed out
        │   ├── ImageUploader.jsx   # react-dropzone drag-and-drop + preview
        │   ├── MetricCard.jsx      # Glassmorphism stat card with highlight variants
        │   ├── DefenseStatusBadge.jsx # Anomaly status indicator
        │   ├── PipelineStepTracker.jsx # 5-step animated pipeline visualization
        │   └── EpsilonSlider.jsx   # ε range slider with live readout
        └── pages/
            ├── LoginPage.jsx
            ├── RegisterPage.jsx
            ├── DashboardPage.jsx   # Live Detection + Attack Test (main page)
            ├── AnalyticsPage.jsx   # Recharts benchmark bar chart
            └── HistoryPage.jsx     # Paginated inference history table
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend + ML |
| Node.js | 18+ | Frontend |
| PostgreSQL | 14+ | Database |

### 1. Database Setup

```bash
psql -U postgres
CREATE DATABASE anomaly_defense;
CREATE USER anomaly_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE anomaly_defense TO anomaly_user;
\q
```

### 2. Backend Setup

```bash
cd backend

# Copy and fill in environment variables
cp .env.example .env
# Edit .env: set DATABASE_URL, JWT_SECRET_KEY

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload --port 8000
```

> API docs auto-generated at **http://localhost:8000/docs**

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

> App runs at **http://localhost:5173** — Vite proxies `/api/*` to port 8000.

---

## 🔌 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/register` | ❌ | Create user account |
| POST | `/api/auth/login` | ❌ | Get JWT token |
| GET | `/api/auth/me` | ✅ | Current user info |
| POST | `/api/defense/analyze` | ✅ | Run defense pipeline on image (logged to your account) |
| POST | `/api/defense/attack-test` | ✅ | Generate adversarial + run defense |
| GET | `/api/history` | ✅ | Paginated inference history |
| GET | `/api/history/{id}` | ✅ | Single log entry |
| GET | `/api/experiments/results` | ❌ | Benchmark results for Analytics page |
| GET | `/health` | ❌ | Health check + model loaded status |

> 🔐 **Login is required to analyze images.** The `/api/defense/analyze` and `/api/defense/attack-test` endpoints reject unauthenticated requests (`get_current_user` JWT dependency). Client-side, the Dashboard loads for everyone (it's the landing page), but **uploading an image, clicking Analyze, or running an attack test while signed out redirects the visitor to `/login`**; the `/history` route is fully guarded by `RequireAuth`. Register or log in first, then analyze.

---

## 🛡️ Defense Pipeline

```
Upload Image
    ↓
1. Preprocess (grayscale → invert → 28×28 → [0,1])
    ↓
2. DynamicAnomalyDetector  (threshold = 0.037860)
    ↓ [if anomalous]
3. SelfPurifier  (4 candidates → best confidence wins)
    ↓
4a. RandomizedDefense  (N=4 MC runs, σ=0.03)
4b. GradientDiversityDefense  (structural transform consistency)
    ↓
5. CombinedDefense consensus → Final Prediction
```

---

## 🏋️ Supported Attacks

| Attack | Description |
|--------|-------------|
| `fgsm` | Fast Gradient Sign Method — single-step |
| `pgd` | Projected Gradient Descent — iterative |
| `eot_pgd` | Expectation Over Transformation PGD — defeats randomized defenses |

---

## 📊 Trained Models

| File | Dataset | Purpose |
|------|---------|---------|
| `models/baseline_cnn.pth` | MNIST | **Primary** — used by the API |
| `models/graddiv_cnn.pth` | MNIST | Gradient-diversity regularized variant |
| `models/gmdcn_cnn_clip.pth` | MNIST | GMDCN variant trained with gradient manipulation (clipping) |
| `models/cifar10_cnn.pth` | CIFAR-10 | Full CIFAR-10 defense evaluation |
| `models/fashion_mnist_cnn.pth`| Fashion-MNIST | Full Fashion-MNIST defense evaluation |

---

## 🗄️ Database Schema

Three tables managed via Alembic migrations:

- **`users`** — email + bcrypt password hash + UUID PK
- **`anomaly_analysis_logs`** — one row per API call; stores all pipeline outputs + execution time
- **`benchmark_results`** — static rows for the Analytics page (seed from `evaluate_attacks.py`)

---

## ⚠️ Known Pending Item

The `GET /api/experiments/results` endpoint returns data from `benchmark_results`. This table is currently **empty** — the existing `src/attacks/evaluate_attacks.py` writes results to local files only. A DB-insertion adapter needs to be added to seed this table.

---

## 📦 Tech Stack

**Backend:** Python 3.10 · FastAPI 0.115 · SQLAlchemy 2.0 (async) · asyncpg · Alembic · python-jose · passlib[bcrypt] · PyTorch 2.x · Pillow

**Frontend:** React 18 · Vite 5 · React Router 6 · Tailwind CSS 3 · Axios · Recharts · Lucide React · react-dropzone · react-hot-toast

**Database:** PostgreSQL 14+ (async via asyncpg driver)