"""
Feature Engineering (Paper §III-B)
===================================
Constructs two complementary representations:
  1. X_seq — raw univariate sequence for LSTM-GRU / Transformer
  2. X_tab — derived tabular features for XGBoost

Fixed bugs from original ridge_stacker_gnss.py:
  - Cyclic encoding now uses modular time-of-day (i % 96) not global fraction
  - Lag feature at t-96 handled correctly via window bounds check
"""

import numpy as np
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LENGTH, LAG_STEPS, ROLLING_WINDOW


def build_features(
    series: np.ndarray,
    horizon: int,
    seq_len: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X_seq, X_tab, y) for a single satellite error time-series.

    Parameters
    ----------
    series : np.ndarray, shape (N,)
        Full error time-series (e.g., 672 steps for 7-day train).
    horizon : int
        Prediction horizon h (steps ahead). E.g., h=1 → 15min, h=96 → 24hr.
    seq_len : int
        Lookback window length. Default 96 (1 day).

    Returns
    -------
    X_seq : np.ndarray, shape (M, seq_len, 1) — raw sequence for LSTM/Transformer
    X_tab : np.ndarray, shape (M, n_features)  — tabular features for XGBoost
    y     : np.ndarray, shape (M,)             — target value h steps ahead
    """
    n = len(series)
    X_seq_list, X_tab_list, y_list = [], [], []

    for i in range(seq_len, n - horizon):
        window = series[i - seq_len:i]
        target = series[i + horizon - 1]

        # ── Sequential representation (paper §III-B, item 1) ──
        # Raw error values preserving temporal ordering
        x_seq = window.reshape(-1, 1).astype(np.float32)

        # ── Derived feature representation (paper §III-B, item 2) ──
        tab_features = []

        # 1. Lag features at {1, 2, 4, 8, 96} steps (paper §III-B)
        for lag in LAG_STEPS:
            if lag <= seq_len:
                tab_features.append(window[-lag])
            else:
                # If lag > seq_len, use oldest available value
                tab_features.append(window[0])

        # 2. Rolling statistics over recent window (paper §III-B)
        recent = window[-ROLLING_WINDOW:]
        tab_features.append(recent.mean())           # moving average
        tab_features.append(recent.std() + 1e-8)     # spread (std)
        tab_features.append(recent.max())             # maximum
        tab_features.append(recent.min())             # minimum

        # 3. Cyclic time features — daily and half-daily (paper §III-B)
        # Use position within day (i % 96) for proper daily/half-daily encoding
        time_in_day = (i % 96) / 96.0  # normalized [0, 1)
        tab_features.append(np.sin(2 * np.pi * time_in_day))   # daily sin
        tab_features.append(np.cos(2 * np.pi * time_in_day))   # daily cos
        tab_features.append(np.sin(4 * np.pi * time_in_day))   # half-daily sin
        tab_features.append(np.cos(4 * np.pi * time_in_day))   # half-daily cos

        # 4. First-order difference — rate of change (paper §III-B)
        diff1 = window[-1] - window[-2]
        tab_features.append(diff1)

        # 5. Second-order difference — acceleration (paper §III-B)
        diff2 = window[-2] - window[-3]
        accel = diff1 - diff2
        tab_features.append(accel)

        # 6. Horizon h as input feature (paper §III-E)
        # "the forecast distance h gets tossed directly into the mix"
        tab_features.append(float(horizon))

        x_tab = np.array(tab_features, dtype=np.float32)

        X_seq_list.append(x_seq)
        X_tab_list.append(x_tab)
        y_list.append(target)

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(X_tab_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32)
    )


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
    val_frac: float = 0.15
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
        y[:split],     y[split:]
    )


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    series = np.random.randn(672).astype(np.float32)

    for h in [1, 2, 4, 8, 96]:
        X_seq, X_tab, y = build_features(series, horizon=h)
        print(f"Horizon {h:>2}: X_seq={X_seq.shape}, X_tab={X_tab.shape}, y={y.shape}")

    print(f"\nFeature names ({len(get_feature_names())}): {get_feature_names()}")
