"""
Synthetic Data Generator for GNSS Error Prediction  (v2 — grid-aware)
=======================================================================
The previous version jittered timestamps by ±1-7 minutes. Because the
data loader resamples everything to a 15-min grid, those copies fell in
the SAME bin and were averaged away — giving zero new time steps.

New strategy: Block-bootstrap on the RESAMPLED 15-min grid
-----------------------------------------------------------
1. Load + resample real data to the 15-min grid (exactly as the ML pipeline
   does) using the same GNSSDataset loader.
2. Slice the series into contiguous 24-hour blocks (96 steps × 15 min).
3. Randomly sample blocks with replacement, add per-column Gaussian noise
   (noise_std = 5% × column_std), and assign synthetic timestamps BEFORE
   the real series starts — creating genuinely new past days.
4. Write the extended series back to a new CSV file that the data loader
   can consume directly (already at 15-min spacing, no resampling needed).

Result
------
  GEO : 649 real steps  +  ~650 synthetic steps  → ~1,300 steps total (2×)
  MEO : 761 real steps  +  ~760 synthetic steps  → ~1,522 steps total (2×)

The multiplier is conservative (2×) because block-bootstrap preserves
autocorrelation structure — more copies just repeat the same patterns.
For sequence models (LSTM/Transformer) doubling the temporal extent is
the most beneficial augmentation available given the data we have.

Usage
-----
  python generate_synthetic_data.py

config.py is already updated to use the augmented files.
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Settings ──────────────────────────────────────────────────────────────────
SYNTHETIC_DAYS  = 7          # How many synthetic 24-hour days to prepend
NOISE_FRACTION  = 0.05       # noise_std = NOISE_FRACTION × per-column std
RANDOM_SEED     = 42
RESAMPLE_FREQ   = "15min"    # Must match ml/config.py RESAMPLE_INTERVAL
STEPS_PER_DAY   = 96         # 24 hr / 15 min

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml"))

# Columns in the resampled internal format → output CSV column names
INTERNAL_TO_CSV = {
    "x_error_m":     "x_error (m)",
    "y_error_m":     "y_error (m)",
    "z_error_m":     "z_error (m)",
    "clock_error_m": "satclockerror (m)",
}
ERROR_COLS_INTERNAL = list(INTERNAL_TO_CSV.keys())

JOBS = [
    {"sat_id": "GEO", "output": "DATA_GEO_Train_augmented.csv"},
    {"sat_id": "MEO", "output": "DATA_MEO_Train_augmented.csv"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_resampled(sat_id: str) -> pd.DataFrame:
    """Load and resample real data exactly as the ML pipeline does."""
    from ml.data.data_loader import GNSSDataset
    ds = GNSSDataset()
    df = ds.train_dfs[sat_id].copy()        # already resampled to 15-min grid
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def block_bootstrap_days(
    df: pd.DataFrame,
    n_days: int,
    noise_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate `n_days` synthetic days by block-bootstrapping 24-hr windows
    from `df` and adding per-column Gaussian noise.

    Synthetic timestamps are placed BEFORE the real series start so that
    after concatenation the series is temporally ordered.
    """
    cols = [c for c in ERROR_COLS_INTERNAL if c in df.columns]
    noise_stds = {c: noise_fraction * float(df[c].std()) for c in cols}

    real_start = df["timestamp"].iloc[0]
    n_real     = len(df)

    synthetic_blocks = []

    for day_idx in range(n_days, 0, -1):          # day_idx=n_days is the furthest back
        # Random starting position for the bootstrap window (must fit STEPS_PER_DAY)
        max_start = max(0, n_real - STEPS_PER_DAY)
        start_pos = int(rng.integers(0, max_start + 1))
        window    = df.iloc[start_pos : start_pos + STEPS_PER_DAY].copy()

        # Assign synthetic timestamps: day_idx synthetic days before real start
        n_window = len(window)
        syn_end   = real_start - pd.Timedelta(minutes=15 * (day_idx - 1) * STEPS_PER_DAY)
        syn_start = syn_end - pd.Timedelta(minutes=15 * (n_window - 1))
        window["timestamp"] = pd.date_range(syn_start, periods=n_window, freq=RESAMPLE_FREQ)

        # Add Gaussian noise
        for col, std in noise_stds.items():
            if std > 0:
                window[col] = window[col].values + rng.normal(0.0, std, size=n_window)

        # Drop satellite_id / satellite_type — will be re-added after concat
        synthetic_blocks.append(window)

    return pd.concat(synthetic_blocks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def to_csv_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert internal column names back to ISRO CSV format."""
    out = pd.DataFrame()
    out["utc_time"] = df["timestamp"].dt.strftime("%#m/%#d/%Y %#H:%M")   # Windows
    for int_col, csv_col in INTERNAL_TO_CSV.items():
        if int_col in df.columns:
            out[csv_col] = df[int_col].values
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 65)
    print("  GNSS Synthetic Data Generator  (v2 — resampled-grid aware)")
    print(f"  Synthetic days : {SYNTHETIC_DAYS}  |  Noise fraction : {NOISE_FRACTION}")
    print("=" * 65)

    for job in JOBS:
        sat_id = job["sat_id"]
        output = job["output"]

        print(f"\n[{sat_id}]  generating {SYNTHETIC_DAYS} synthetic days ...")

        real_df  = load_resampled(sat_id)
        syn_df   = block_bootstrap_days(real_df, SYNTHETIC_DAYS, NOISE_FRACTION, rng)

        # Combine: synthetic days come first (temporally), then real data
        combined = pd.concat([syn_df, real_df], ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        n_syn  = len(syn_df)
        n_real = len(real_df)
        n_tot  = len(combined)
        print(f"    Real steps      : {n_real:>6,}")
        print(f"    Synthetic steps : {n_syn:>6,}  ({SYNTHETIC_DAYS} days x ~{STEPS_PER_DAY} steps/day)")
        print(f"    Total steps     : {n_tot:>6,}  ({n_tot/n_real:.1f}x original)")

        # Save as CSV (15-min spaced, no further resampling needed)
        out_csv = to_csv_format(combined)
        out_path = os.path.join(TRAIN_DIR, output)
        out_csv.to_csv(out_path, index=False)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"    Saved -> {out_path}  ({size_kb:.1f} KB)")

    print("\n" + "=" * 65)
    print("  DONE! config.py already points to the augmented files.")
    print("  Next: python ml/pipeline/train.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
