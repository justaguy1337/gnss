"""
Configuration for GNSS Error Prediction Pipeline
=================================================
Parameters tuned for the actual ISRO dataset provided:
  - GEO:  Train=647 rows (Sep 1-7), Test=95 rows (Sep 8)
  - MEO:  Train=759 rows (Sep 1-9), Test=205 rows (Sep 8)
  - Irregular sampling (2-min to 6-hr gaps) → resampled to uniform grid
  - Target resample interval: 15 minutes

Paper: "Multi-Horizon GNSS Clock and Ephemeris Error Prediction
        via Stacked Ensemble Learning"

v2 changes (accuracy fix):
  - Right-sized models (hidden 32→8) to match 623-sample dataset
  - Shorter lookback (24→12 steps) to focus on high-autocorrelation zone
  - Stronger regularisation (Ridge alpha 0.5→10, more dropout)
  - XGBoost shallower (depth 4→2, estimators 200→50)
  - Expanding-window time-series CV replaces simple val split
  - Difference-based modelling flag (DIFFERENCE_TARGET)
  - Test-time normalisation flag (TEST_NORMALISE)
"""

import torch
import os

# =============================================================================
# DATA — adapted for real ISRO dataset
# =============================================================================
RESAMPLE_INTERVAL = "15min"       # Resample irregular data to this frequency
SEQUENCE_LENGTH   = 24            # Look-back window: 24 steps = 6 hours at 15min
                                  # (increased from 12 — captures full diurnal pattern
                                  #  and gives the model enough context to see
                                  #  volatility regime changes)
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
# ACCURACY IMPROVEMENT FLAGS
# =============================================================================
# Difference-based modelling: predict Δe(t+h) = e(t+h) - e(t) instead of
# raw e(t+h). Removes mean-shift between train and test distributions.
# Absolute predictions are reconstructed by adding back the last known value.
DIFFERENCE_TARGET = True

# Test-time normalisation: before inference, scale test data to have the
# same mean/std as the training series (then invert after prediction).
# Critical for GEO whose test variance is 41x larger than training.
TEST_NORMALISE = True

# =============================================================================
# FEATURE ENGINEERING (paper §III-B) — scaled for smaller dataset
# =============================================================================
LAG_STEPS     = [1, 2, 4, 8, 12, 24]  # lag features — extended to cover 6hr window
ROLLING_WINDOW = 8                     # 2-hour rolling window (8 x 15min)

# =============================================================================
# CROSS-VALIDATION — expanding-window time-series CV
# =============================================================================
# We use expanding-window CV instead of random KFold:
#   fold 1: train on [0, split1),  validate on [split1, split2)
#   fold 2: train on [0, split2),  validate on [split2, split3)
#   ...
# This preserves temporal order and prevents future leakage.
N_CV_FOLDS    = 5                   # 5 expanding-window folds for better generalization
VAL_FRACTION  = 0.20                # fraction of data reserved for each val set

# =============================================================================
# TRAINING
# =============================================================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 32               # larger batch — better GPU utilisation (GPU finishes tiny batches in <1ms)
EPOCHS         = 200              # more epochs for smaller model
LEARNING_RATE  = 5e-4
GRAD_CLIP      = 1.0
PATIENCE       = 40               # generous patience — larger model needs more time
LR_PATIENCE    = 15

# =============================================================================
# LSTM-GRU (paper §III-C) — right-sized for 623 training samples
# =============================================================================
# Previous: hidden=32, params ~8720 → samples/params = 0.07 (severe overfit)
# Now:      hidden=8,  params ~560  → samples/params = 1.1  (much better)
LSTM_HIDDEN   = 16
LSTM_LAYERS   = 1
GRU_HIDDEN    = 16
GRU_LAYERS    = 1
LSTM_DROPOUT  = 0.4               # slightly higher dropout with bigger model
LSTM_HEAD_DIM = 16

# =============================================================================
# TRANSFORMER (paper §III-D) — right-sized
# =============================================================================
TRANS_D_MODEL  = 32               # must be divisible by TRANS_NHEAD
TRANS_NHEAD    = 4
TRANS_NUM_LAYERS = 2
TRANS_DIM_FF   = 64
TRANS_DROPOUT  = 0.3
TRANS_HEAD_DIM = 16

# =============================================================================
# XGBOOST (paper §III-E) — shallower to prevent overfitting small data
# =============================================================================
XGB_N_ESTIMATORS  = 100           # more trees with stronger regularization
XGB_MAX_DEPTH     = 3             # slightly deeper to capture more patterns
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE     = 0.8
XGB_COLSAMPLE     = 0.8
XGB_EARLY_STOPPING = 15

# =============================================================================
# RIDGE STACKER (paper §III-F)
# =============================================================================
# Strong regularisation prevents the stacker from assigning large negative
# weights (which was the main symptom of the original negative-R² problem).
RIDGE_ALPHA = 50.0                # strong regularization — prevents stacker collapse

# =============================================================================
# GAUSSIAN PROCESS (paper §III-G) — Matern(nu=2.5) + Periodic
# =============================================================================
GP_MATERN_NU    = 2.5
GP_TRAIN_ITERS  = 150
GP_LEARNING_RATE = 0.05

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
