"""
Feature Engineering (Paper §III-B)
===================================
Constructs two complementary representations:
  1. X_seq — raw univariate sequence for LSTM-GRU / Transformer
  2. X_tab — derived tabular features for XGBoost

v2 changes:
  - DIFFERENCE_TARGET flag: model predicts Δe(t+h) = e(t+h) - e(t)
    instead of raw e(t+h). Removes mean-level distribution shift.
  - Lag steps capped to SEQUENCE_LENGTH (was using steps > seq_len silently)
  - Cyclic encoding uses modular time-of-day (i % 96) not global fraction
"""

import numpy as np
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LENGTH, LAG_STEPS, ROLLING_WINDOW, DIFFERENCE_TARGET


def build_features(
    series: np.ndarray,
    horizon: int,
    seq_len: int = SEQUENCE_LENGTH,
    difference_target: bool = DIFFERENCE_TARGET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X_seq, X_tab, y) for a single satellite error time-series.

    Parameters
    ----------
    series : np.ndarray, shape (N,)
        Full error time-series (normalised or raw).
    horizon : int
        Prediction horizon h (steps ahead). E.g., h=1 → 15min.
    seq_len : int
        Lookback window length.
    difference_target : bool
        If True, y = e(t+h) - e(t)  (difference from last known value).
        If False, y = e(t+h)        (absolute value, original behaviour).

    Returns
    -------
    X_seq : np.ndarray, shape (M, seq_len, 1) — raw sequence for LSTM/Transformer
    X_tab : np.ndarray, shape (M, n_features)  — tabular features for XGBoost
    y     : np.ndarray, shape (M,)             — target (diff or absolute)
    last_vals : np.ndarray, shape (M,)         — e(t) anchor for reconstruction
    """
    n = len(series)
    X_seq_list, X_tab_list, y_list = [], [], []

    for i in range(seq_len, n - horizon):
        window = series[i - seq_len:i]
        future_val = series[i + horizon - 1]
        last_val   = series[i - 1]           # most recent observed value e(t)

        # ── Sequential representation (paper §III-B, item 1) ──
        x_seq = window.reshape(-1, 1).astype(np.float32)

        # ── Derived feature representation (paper §III-B, item 2) ──
        tab_features = []

        # 1. Lag features — capped to seq_len
        for lag in LAG_STEPS:
            actual_lag = min(lag, seq_len)
            tab_features.append(window[-actual_lag])

        # 2. Rolling statistics over recent window
        recent = window[-ROLLING_WINDOW:]
        tab_features.append(recent.mean())           # moving average
        tab_features.append(recent.std() + 1e-8)     # spread (std)
        tab_features.append(recent.max())             # maximum
        tab_features.append(recent.min())             # minimum

        # 3. Cyclic time features — daily and half-daily (paper §III-B)
        time_in_day = (i % 96) / 96.0  # normalized [0, 1)
        tab_features.append(np.sin(2 * np.pi * time_in_day))
        tab_features.append(np.cos(2 * np.pi * time_in_day))
        tab_features.append(np.sin(4 * np.pi * time_in_day))
        tab_features.append(np.cos(4 * np.pi * time_in_day))

        # 4. First-order difference — rate of change
        diff1 = window[-1] - window[-2]
        tab_features.append(diff1)

        # 5. Second-order difference — acceleration
        if len(window) >= 3:
            diff2 = window[-2] - window[-3]
            accel = diff1 - diff2
        else:
            accel = 0.0
        tab_features.append(accel)

        # 6. Horizon h as input feature (paper §III-E)
        tab_features.append(float(horizon))

        # ── Target ──
        if difference_target:
            # Predict Δ = e(t+h) - e(t): removes mean-shift between train/test
            target = future_val - last_val
        else:
            target = future_val

        x_tab = np.array(tab_features, dtype=np.float32)
        X_seq_list.append(x_seq)
        X_tab_list.append(x_tab)
        y_list.append(target)

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(X_tab_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


def build_single_window(
    window: np.ndarray,
    horizon: int,
    time_offset: int = 0,
    seq_len: int = SEQUENCE_LENGTH,
) -> tuple:
    """
    Build features for ONE sliding window (no future value required).

    Used for rolling inference: at each test step we have exactly
    `seq_len` observed values and we want to predict `horizon` steps ahead.

    Parameters
    ----------
    window     : np.ndarray, shape (seq_len,) — the last seq_len observations
    horizon    : int — prediction horizon
    time_offset: int — absolute time index (for cyclic time features)
    seq_len    : int — lookback window length

    Returns
    -------
    X_seq : np.ndarray, shape (1, seq_len, 1)
    X_tab : np.ndarray, shape (1, n_features)
    anchor: float — last observed value (used to reconstruct from diff)
    """
    assert len(window) >= seq_len, (
        f"window length {len(window)} < seq_len {seq_len}"
    )
    w = np.array(window[-seq_len:], dtype=np.float32)

    x_seq = w.reshape(1, seq_len, 1)

    tab_features = []

    # 1. Lag features
    for lag in LAG_STEPS:
        actual_lag = min(lag, seq_len)
        tab_features.append(float(w[-actual_lag]))

    # 2. Rolling statistics
    recent = w[-ROLLING_WINDOW:]
    tab_features.append(float(recent.mean()))
    tab_features.append(float(recent.std() + 1e-8))
    tab_features.append(float(recent.max()))
    tab_features.append(float(recent.min()))

    # 3. Cyclic time features
    time_in_day = (time_offset % 96) / 96.0
    tab_features.append(float(np.sin(2 * np.pi * time_in_day)))
    tab_features.append(float(np.cos(2 * np.pi * time_in_day)))
    tab_features.append(float(np.sin(4 * np.pi * time_in_day)))
    tab_features.append(float(np.cos(4 * np.pi * time_in_day)))

    # 4. First-order difference
    diff1 = float(w[-1] - w[-2]) if seq_len >= 2 else 0.0
    tab_features.append(diff1)

    # 5. Second-order difference (acceleration)
    if seq_len >= 3:
        diff2 = float(w[-2] - w[-3])
        accel = diff1 - diff2
    else:
        accel = 0.0
    tab_features.append(accel)

    # 6. Horizon as feature
    tab_features.append(float(horizon))

    x_tab = np.array(tab_features, dtype=np.float32).reshape(1, -1)
    anchor = float(w[-1])   # e(t) — used to reconstruct from delta predictions

    return x_seq, x_tab, anchor



def reconstruct_from_diff(
    delta_preds: np.ndarray,
    series: np.ndarray,
    horizon: int,
    seq_len: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Reconstruct absolute predictions from difference predictions.

    For each sample i (starting at seq_len), the anchor is series[i-1].
    Absolute prediction = anchor + delta_pred.

    Parameters
    ----------
    delta_preds : np.ndarray, shape (M,) — predicted differences Δe(t+h)
    series      : np.ndarray, shape (N,) — original series used to build features
    horizon     : int — prediction horizon (not used directly, kept for signature clarity)
    seq_len     : int — lookback window

    Returns
    -------
    abs_preds : np.ndarray, shape (M,) — absolute predictions
    """
    anchors = np.array([
        series[i - 1]
        for i in range(seq_len, len(series) - horizon)
    ], dtype=np.float32)

    # anchors and delta_preds must be same length
    n = min(len(anchors), len(delta_preds))
    return anchors[:n] + delta_preds[:n]


def get_feature_names() -> list:
    """Return human-readable names for the tabular features."""
    names = []
    for lag in LAG_STEPS:
        names.append(f"lag_{lag}")
    names.extend([
        "roll_mean", "roll_std", "roll_max", "roll_min",
        "sin_24h", "cos_24h", "sin_12h", "cos_12h",
        "diff_1", "accel",
        "horizon_h"
    ])
    return names


def train_val_split(
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal train/validation split (no shuffling — preserves time order).

    Returns
    -------
    X_seq_tr, X_seq_val, X_tab_tr, X_tab_val, y_tr, y_val
    """
    n = len(y)
    split = int(n * (1 - val_frac))
    return (
        X_seq[:split], X_seq[split:],
        X_tab[:split], X_tab[split:],
        y[:split],     y[split:],
    )


def expanding_window_splits(n: int, n_folds: int = 3, min_train_frac: float = 0.40):
    """
    Generate expanding-window train/val index pairs for time-series CV.

    Fold k uses [0, split_k) for training and [split_k, split_{k+1}) for validation.
    The initial training window is at least min_train_frac of the data.

    Parameters
    ----------
    n          : total number of samples
    n_folds    : number of CV folds
    min_train_frac : minimum fraction used as the first training window

    Yields
    ------
    (train_indices, val_indices) for each fold
    """
    # Reserve the last (1 - min_train_frac) fraction for CV splits
    start = int(n * min_train_frac)
    remaining = n - start
    fold_size = remaining // (n_folds + 1)

    for k in range(n_folds):
        val_start = start + k * fold_size
        val_end   = val_start + fold_size
        if val_end > n:
            break
        train_idx = np.arange(0, val_start)
        val_idx   = np.arange(val_start, val_end)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    series = np.cumsum(np.random.randn(300).astype(np.float32))  # random walk

    for h in [1, 2, 4, 8, 16]:
        X_seq, X_tab, y = build_features(series, horizon=h)
        print(f"Horizon {h:>2}: X_seq={X_seq.shape}, X_tab={X_tab.shape}, "
              f"y={y.shape}, y_mean={y.mean():.4f}")

    print(f"\nFeature names ({len(get_feature_names())}): {get_feature_names()}")

    print("\nExpanding-window splits (n=200, folds=3):")
    for i, (tr, vl) in enumerate(expanding_window_splits(200, n_folds=3)):
        print(f"  Fold {i+1}: train=[0,{tr[-1]+1}), val=[{vl[0]},{vl[-1]+1})")
