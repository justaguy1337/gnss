"""
Configuration for GNSS Error Prediction Pipeline
=================================================
Parameters tuned for the actual ISRO dataset provided:
  - GEO:  Train=142 rows (Sep 1-7), Test=69 rows (Sep 8)
  - MEO:  Train=334 rows (MEO_Train + MEO_Train2)
          Test =41  rows (MEO_Test  + MEO_Test2)
  - Irregular sampling (2-min to 6-hr gaps) → resampled to uniform grid
  - Target resample interval: 15 minutes

Paper: "Multi-Horizon GNSS Clock and Ephemeris Error Prediction
        via Stacked Ensemble Learning"
"""

import torch
import os

# =============================================================================
# DATA — adapted for real ISRO dataset
# =============================================================================
RESAMPLE_INTERVAL = "15min"       # Resample irregular data to this frequency
SEQUENCE_LENGTH   = 24            # Look-back window: 24 steps = 6 hours at 15min
                                  # (reduced from 96 — actual data is much sparser)
TRAIN_DAYS        = 7             # 7 days training (Sep 1-7)
TEST_STEPS        = 24            # ~6 hours of day-8 test data after resample

# Real dataset locations — both organised into subfolders
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")   # D:\github\gnss\dataset\train
TEST_DIR    = os.path.join(DATASET_DIR, "test")    # D:\github\gnss\dataset\test

# Dataset file mapping: satellite_id -> (train_files, test_files, type)
DATASET_FILES = {
    "GEO": {
        "train_dir": TRAIN_DIR,
        "test_dir":  TEST_DIR,
        "train": ["DATA_GEO_Train.csv"],
        "test":  ["DATA_GEO_Test.csv"],
        "type":  "GEO",
    },
    "MEO": {
        "train_dir": TRAIN_DIR,
        "test_dir":  TEST_DIR,
        "train": ["DATA_MEO_Train.csv", "DATA_MEO_Train2.csv"],
        "test":  ["DATA_MEO_Test.csv",  "DATA_MEO_Test2.csv"],
        "type":  "MEO",
    },
}

# =============================================================================
# HORIZONS — adapted to dataset's actual temporal resolution
# =============================================================================
# After 15-min resampling:
#   h=1  → 15 min
#   h=2  → 30 min
#   h=4  → 1 hr
#   h=8  → 2 hr
#   h=16 → 4 hr   (replaces h=96/24hr — not enough data for 24hr horizon)
HORIZONS = [1, 2, 4, 8, 16]

# =============================================================================
# FEATURE ENGINEERING (paper §III-B) — scaled for smaller dataset
# =============================================================================
LAG_STEPS     = [1, 2, 4, 8, 16]   # lag features matching horizons
ROLLING_WINDOW = 4                  # 1-hour rolling window (4 × 15min)
N_CV_FOLDS    = 3                   # reduced from 5 (less data available)
VAL_FRACTION  = 0.20                # 20% holdout — larger fraction for small dataset

# =============================================================================
# TRAINING
# =============================================================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 16               # smaller batch — less data
EPOCHS         = 100              # more epochs to compensate for less data
LEARNING_RATE  = 5e-4             # slightly lower LR for stability
GRAD_CLIP      = 1.0
PATIENCE       = 15               # more patience for small dataset
LR_PATIENCE    = 7

# =============================================================================
# LSTM-GRU (paper §III-C)
# =============================================================================
LSTM_HIDDEN   = 32               # smaller network for small dataset (was 64)
LSTM_LAYERS   = 1                # (was 2)
GRU_HIDDEN    = 32
GRU_LAYERS    = 1
LSTM_DROPOUT  = 0.1              # less dropout — less data
LSTM_HEAD_DIM = 16

# =============================================================================
# TRANSFORMER (paper §III-D)
# =============================================================================
TRANS_D_MODEL  = 32              # smaller model (was 64)
TRANS_NHEAD    = 4
TRANS_NUM_LAYERS = 1             # single layer (was 2)
TRANS_DIM_FF   = 64              # (was 128)
TRANS_DROPOUT  = 0.1
TRANS_HEAD_DIM = 16

# =============================================================================
# XGBOOST (paper §III-E)
# =============================================================================
XGB_N_ESTIMATORS  = 200          # fewer trees for small dataset (was 500)
XGB_MAX_DEPTH     = 4            # shallower (was 6) — prevent overfit
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE     = 0.8
XGB_COLSAMPLE     = 0.8
XGB_EARLY_STOPPING = 15          # (was 20)

# =============================================================================
# RIDGE STACKER (paper §III-F)
# =============================================================================
RIDGE_ALPHA = 0.5                # slightly less regularization (was 1.0)

# =============================================================================
# GAUSSIAN PROCESS (paper §III-G) — Matern(nu=2.5) + Periodic
# =============================================================================
GP_MATERN_NU    = 2.5
GP_TRAIN_ITERS  = 150            # more iterations for better fit (was 100)
GP_LEARNING_RATE = 0.05          # slower for stability (was 0.1)

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "output")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

# Create directories
for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Legacy alias (kept for backward compatibility)
SAMPLE_INTERVAL_MIN = 15
TRAIN_STEPS = TRAIN_DAYS * SEQUENCE_LENGTH  # used loosely; actual steps depend on resampled data
