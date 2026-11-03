# GNSS ErrorNet — Multi-Horizon Satellite Error Prediction

> **Multi-Horizon GNSS Clock and Ephemeris Error Prediction via Stacked Ensemble Learning**

A full-stack application for predicting GNSS satellite clock and ephemeris errors using a stacked ensemble of deep learning and gradient boosting models, with Gaussian Process residual correction for calibrated uncertainty estimates. Built with real ISRO satellite data.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![React](https://img.shields.io/badge/React-19-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Architecture

The prediction pipeline decomposes GNSS error forecasting into three subproblems — short-term drift, long-term periodicity, and covariate dependency — handled by specialized base models whose outputs are fused by a meta-learner:

```
Input (7-day error series, 15-min intervals)
    │
    ├─► LSTM-GRU        ── captures short-term temporal drift
    ├─► Transformer     ── captures periodic/attention patterns
    └─► XGBoost         ── captures tabular/covariate dependencies
          │
          ▼
    Ridge Stacker       ── per-horizon optimal blending (meta-learner)
          │
          ▼
    Gaussian Process    ── Matérn(2.5) + Periodic kernel residual correction
          │                with calibrated uncertainty estimates
          ▼
    Final Prediction ± Confidence Interval
```

### Prediction Horizons

| Horizon | Steps | Time       |
|---------|-------|------------|
| h=1     | 1     | 15 minutes |
| h=2     | 2     | 30 minutes |
| h=4     | 4     | 1 hour     |
| h=8     | 8     | 2 hours    |
| h=16    | 16    | 4 hours    |

### Satellite Types

- **GEO** (Geostationary) — 647 training rows (Sep 1–7), 95 test rows (Sep 8)
- **MEO** (Medium Earth Orbit) — 759 training rows (Sep 1–9), 205 test rows (Sep 8)

---

## Project Structure

```
gnss/
├── dataset/                    # ISRO satellite datasets
│   ├── train/                  #   Training CSVs (GEO + MEO)
│   └── test/                   #   Test CSVs
├── ml/                         # ML backend (Python)
│   ├── models/                 #   Model implementations
│   │   ├── lstm_gru.py         #     LSTM-GRU hybrid network
│   │   ├── transformer.py      #     Time-series Transformer
│   │   ├── xgboost_model.py    #     XGBoost regressor
│   │   ├── ridge_stacker.py    #     Ridge meta-learner
│   │   └── gaussian_process.py #     GP residual correction
│   ├── pipeline/               #   Training & inference pipeline
│   │   ├── train.py            #     Ensemble training orchestrator
│   │   ├── predict.py          #     Day-8 prediction generation
│   │   └── evaluate.py         #     Metrics & normality tests
│   ├── data/
│   │   └── data_loader.py      #     ISRO dataset loader & resampling
│   ├── features/
│   │   └── feature_engineering.py  # Lag, rolling, cyclic features
│   ├── config.py               #   All hyperparameters & paths
│   ├── api.py                  #   FastAPI REST server
│   ├── checkpoints/            #   Saved model weights (auto-generated)
│   └── results/                #   Prediction & evaluation JSONs
├── src/                        # React frontend
│   ├── pages/
│   │   ├── Dashboard.js        #   Main dashboard
│   │   ├── Predictions.js      #   Prediction visualizations
│   │   ├── ModelInsights.js     #   Model weights & architecture
│   │   ├── MetaLearner.js      #   Ridge stacker visualization
│   │   ├── DataUpload.js       #   Test data upload interface
│   │   ├── EarthVisualization.js # 3D globe satellite view
│   │   ├── ImpactBenefits.js   #   Impact analysis
│   │   └── Research.js         #   Research context
│   └── components/
│       ├── Layout.js           #   App layout
│       └── Sidebar.js          #   Navigation sidebar
├── docker-compose.yml          # Docker orchestration
├── Dockerfile.backend          # FastAPI container
├── Dockerfile.frontend         # React + Nginx container
├── nginx.conf                  # Nginx reverse proxy config
├── conference_101719.tex       # Research paper (Springer format)
└── requirements.txt            # Root Python dependencies
```

---

## Getting Started

### Prerequisites

- **Python** 3.11+
- **Node.js** 18+ (for frontend)
- **CUDA** (optional, for GPU-accelerated training)

### Backend Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r ml/requirements.txt

# Start the API server
cd ml
python api.py
```

On first launch, the server will:
1. Load ISRO training data from `dataset/`
2. Train all satellite models (~minutes) and save checkpoints
3. Subsequent restarts load checkpoints instantly (~seconds)

The API is available at `http://localhost:8000`.

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The frontend is available at `http://localhost:3000`.

### Docker (Full Stack)

```bash
docker-compose up --build
```

This starts both services:
- **Backend** → `http://localhost:8000`
- **Frontend** → `http://localhost:3000`

Model checkpoints are persisted via a Docker volume.

---

## API Reference

| Method | Endpoint                    | Description                         |
|--------|-----------------------------|-------------------------------------|
| GET    | `/api/status`               | Pipeline status & readiness         |
| POST   | `/api/upload`               | Upload test CSV (test data only)    |
| POST   | `/api/train`                | Retrain ensemble (background task)  |
| GET    | `/api/predict/all`          | Predictions for all horizons        |
| GET    | `/api/predict/{horizon}`    | Predictions for a specific horizon  |
| GET    | `/api/models`               | Model info & Ridge stacker weights  |
| GET    | `/api/evaluation`           | Evaluation metrics & normality      |
| GET    | `/api/evaluation/distribution` | Residual distribution (Q-Q, histogram) |
| GET    | `/api/data/summary`         | Dataset summary statistics          |
| POST   | `/api/clear`                | Clear test data & reset predictions |

### Expected CSV Schema

```
utc_time, x_error (m), y_error (m), z_error (m), satclockerror (m)
```

---

## Key Features

- **Expanding-window cross-validation** — 5-fold temporal CV that preserves time ordering and prevents data leakage
- **Difference-based modeling** — predicts Δe(t+h) instead of raw values to handle distribution shift between train/test
- **Test-time normalization** — scales test data to training statistics (critical for GEO where test variance is 41× training)
- **Data augmentation** — 10× augmented training sets with realistic noise injection
- **Comprehensive normality testing** — Shapiro-Wilk, D'Agostino-Pearson, Kolmogorov-Smirnov, Anderson-Darling
- **Calibrated uncertainty** — Gaussian Process posterior provides confidence intervals
- **3D Earth visualization** — Interactive globe showing satellite orbits

---

## Tech Stack

### Backend
- **PyTorch** — LSTM-GRU and Transformer models
- **XGBoost** — Gradient boosted trees
- **GPyTorch** — Gaussian Process regression
- **scikit-learn** — Ridge regression, preprocessing, metrics
- **FastAPI** — REST API server
- **pandas / NumPy / SciPy** — Data processing & statistical tests

### Frontend
- **React 19** — UI framework
- **Tailwind CSS** — Styling
- **Recharts / Nivo** — Data visualization
- **react-globe.gl / Three.js** — 3D Earth visualization
- **React Router** — Client-side routing

---

## Research Paper

This project accompanies the paper:

> **Multi-Horizon GNSS Clock and Ephemeris Error Prediction via Stacked Ensemble Learning**
>
> Manya Michelle, Tapas Guha, Pallavi A Kumbar, Sharat Patil, Harshavardhan Reddy A
>
> *CMR Institute of Technology, Bengaluru, India*

See [`conference_101719.tex`](conference_101719.tex) for the full manuscript (Springer Proceedings format).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

© 2026 Manya Michelle
