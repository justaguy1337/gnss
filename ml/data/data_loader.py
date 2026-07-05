"""
Data Loader for GNSS Error Prediction
======================================
Loads the real ISRO dataset from d:/github/gnss/dataset/.

Dataset structure:
  GEO:  DATA_GEO_Train.csv  (Sep 1-7, 142 rows, ~26-min median interval)
        DATA_GEO_Test.csv   (Sep 8,   69 rows,  ~15-min median interval)
  MEO:  DATA_MEO_Train.csv  + DATA_MEO_Train2.csv  (334 rows combined)
        DATA_MEO_Test.csv   + DATA_MEO_Test2.csv   (41  rows combined)

All files are resampled to a uniform 15-minute grid (forward-fill + interpolate)
before feeding to the models.

Also supports:
  - Synthetic data (legacy, for testing without real data)
  - Single-file upload via the API
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Dict, Tuple, Optional, List
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEQUENCE_LENGTH, TRAIN_DAYS, TEST_STEPS,
    DATA_DIR, DATASET_DIR, DATASET_FILES,
    RESAMPLE_INTERVAL, SAMPLE_INTERVAL_MIN
)

# ─────────────────────────────────────────────
# Column name aliases — ISRO files use these
# ─────────────────────────────────────────────
ISRO_COLUMN_MAP = {
    "utc_time":          "timestamp",
    "x_error (m)":       "x_error_m",
    "y_error (m)":       "y_error_m",
    "y_error  (m)":      "y_error_m",   # MEO_Train has trailing space
    "z_error (m)":       "z_error_m",
    "satclockerror (m)": "clock_error_m",
}

ISRO_ERROR_COLS = ["clock_error_m", "x_error_m", "y_error_m", "z_error_m"]
SYNTHETIC_ERROR_COLS = ["clock_error_ns", "radial_error_m", "along_track_error_m", "cross_track_error_m"]


# =============================================================================
# Format Detection + Normalisation
# =============================================================================

def detect_format(df: pd.DataFrame) -> str:
    """Detect 'isro' or 'synthetic' based on column names."""
    cols = {c.strip().lower() for c in df.columns}
    if "utc_time" in cols or "satclockerror (m)" in cols:
        return "isro"
    if "clock_error_ns" in cols or "satellite_id" in cols:
        return "synthetic"
    return "isro"   # default guess


def _read_isro_file(
    path: str,
    satellite_id: str = "UNKNOWN",
    satellite_type: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Load one ISRO CSV, normalise column names, parse timestamps.
    Does NOT resample — call _resample() afterwards.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()         # strip trailing spaces in col names

    # Rename to internal names
    df = df.rename(columns=ISRO_COLUMN_MAP)

    # Parse timestamp
    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column in {path}. Cols: {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, format="mixed")

    # Add metadata
    df["satellite_id"]   = satellite_id
    df["satellite_type"] = satellite_type

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _resample_to_uniform(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """
    Resample an irregular time-series to a uniform grid.
    Uses linear interpolation for small gaps, forward-fill for larger ones.
    """
    df = df.copy()
    df = df.set_index("timestamp")

    # Drop duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Identify numeric error columns
    meta_cols = ["satellite_id", "satellite_type"]
    num_cols = [c for c in df.columns if c not in meta_cols]

    # Save metadata (constant per satellite)
    sat_id   = df["satellite_id"].iloc[0]   if "satellite_id"   in df.columns else "UNKNOWN"
    sat_type = df["satellite_type"].iloc[0] if "satellite_type" in df.columns else "UNKNOWN"

    # Resample numeric columns: interpolate, then forward-fill tails
    df_num = df[num_cols].resample(freq).mean()
    df_num = df_num.interpolate(method="time", limit=8)   # max 2-hour gap via interpolation
    df_num = df_num.ffill().bfill()                        # fill remaining edge NaNs

    df_num["satellite_id"]   = sat_id
    df_num["satellite_type"] = sat_type
    df_num = df_num.reset_index().rename(columns={"index": "timestamp"})
    return df_num


def _concat_files(
    paths: List[str],
    satellite_id: str,
    satellite_type: str,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Load multiple CSV files for the same satellite, concatenate, deduplicate,
    then resample to uniform grid.
    """
    dfs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: file not found: {path}")
            continue
        df = _read_isro_file(path, satellite_id=satellite_id, satellite_type=satellite_type)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No files loaded for {satellite_id}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Remove exact timestamp duplicates
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")

    return _resample_to_uniform(combined, freq=freq)


# =============================================================================
# GNSSDataset — main class
# =============================================================================

class GNSSDataset:
    """
    Loads and manages GNSS error data for training and evaluation.

    Supports two modes:
      1. Real ISRO dataset  — loads from DATASET_DIR (d:/github/gnss/dataset/)
      2. Single CSV upload  — from a user-provided path (API upload)
      3. Synthetic fallback — if real data not found
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        folder_path: Optional[str] = None,
        use_real_data: bool = True,
    ):
        self.scalers: Dict[str, Dict[str, StandardScaler]] = {}
        self.train_dfs: Dict[str, pd.DataFrame] = {}
        self.test_dfs:  Dict[str, pd.DataFrame] = {}

        if csv_path is not None:
            # Single uploaded file — treat as one satellite
            self._load_single_csv(csv_path)

        elif folder_path is not None:
            # Legacy: load all CSVs from a folder
            self._load_folder(folder_path)

        elif use_real_data and os.path.isdir(DATASET_DIR):
            # Primary path: load the real ISRO dataset
            self._load_isro_dataset()

        else:
            # Fallback: synthetic data
            synthetic_path = os.path.join(DATA_DIR, "gnss_errors_synthetic.csv")
            if os.path.exists(synthetic_path):
                self._load_synthetic(synthetic_path)
            else:
                raise FileNotFoundError(
                    f"No data found. Expected ISRO dataset at '{DATASET_DIR}' "
                    f"or synthetic data at '{synthetic_path}'.\n"
                    "Run `python data/generate_synthetic.py` to create synthetic data."
                )

        self.satellite_ids = list(self.train_dfs.keys())
        self._set_error_columns()
        self._prepare_scalers()

    # ─────────────────────────────────────────────────────────────────────
    # Loaders
    # ─────────────────────────────────────────────────────────────────────

    def _load_isro_dataset(self):
        """Load the real ISRO train/test file pairs from DATASET_DIR."""
        print(f"Loading ISRO dataset from: {DATASET_DIR}")
        for sat_id, cfg in DATASET_FILES.items():
            sat_type = cfg["type"]

            train_paths = [os.path.join(DATASET_DIR, f) for f in cfg["train"]]
            test_paths  = [os.path.join(DATASET_DIR, f) for f in cfg["test"]]

            try:
                train_df = _concat_files(train_paths, sat_id, sat_type, freq=RESAMPLE_INTERVAL)
                test_df  = _concat_files(test_paths,  sat_id, sat_type, freq=RESAMPLE_INTERVAL)

                self.train_dfs[sat_id] = train_df
                self.test_dfs[sat_id]  = test_df

                print(f"  {sat_id} ({sat_type}): "
                      f"train={len(train_df)} steps, "
                      f"test={len(test_df)} steps "
                      f"[{train_df['timestamp'].iloc[0].date()} to "
                      f"{test_df['timestamp'].iloc[-1].date()}]")
            except Exception as e:
                print(f"  WARNING: Could not load {sat_id}: {e}")

        if not self.train_dfs:
            raise FileNotFoundError(f"No satellite data loaded from {DATASET_DIR}")

    def _load_single_csv(self, csv_path: str):
        """Load a single uploaded CSV (auto-detect format)."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        raw = pd.read_csv(csv_path)
        fmt = detect_format(raw)
        filename = os.path.basename(csv_path)

        sat_type = "GEO" if "GEO" in filename.upper() else ("MEO" if "MEO" in filename.upper() else "UNKNOWN")
        sat_id   = filename.replace(".csv", "").replace("DATA_", "")

        if fmt == "isro":
            df = _read_isro_file(csv_path, satellite_id=sat_id, satellite_type=sat_type)
            df = _resample_to_uniform(df, freq=RESAMPLE_INTERVAL)
        else:
            # Synthetic format — treat whole file as train, no separate test
            raw["timestamp"] = pd.to_datetime(raw["timestamp"])
            raw = raw.sort_values("timestamp").reset_index(drop=True)
            df = raw

        # Split 80/20 into train/test
        split = int(len(df) * 0.8)
        self.train_dfs[sat_id] = df.iloc[:split].reset_index(drop=True)
        self.test_dfs[sat_id]  = df.iloc[split:].reset_index(drop=True)
        print(f"  Loaded {filename}: train={split}, test={len(df)-split} rows")

    def _load_folder(self, folder_path: str):
        """Legacy: load all CSVs from a folder, infer train/test from filename."""
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        for path in csv_files:
            fname = os.path.basename(path)
            sat_type = "GEO" if "GEO" in fname.upper() else "MEO"
            sat_id   = fname.replace(".csv", "").replace("DATA_", "")
            is_test  = "test" in fname.lower()

            df = _read_isro_file(path, satellite_id=sat_id, satellite_type=sat_type)
            df = _resample_to_uniform(df, freq=RESAMPLE_INTERVAL)

            if is_test:
                self.test_dfs[sat_id] = df
            else:
                # Default: register as train
                self.train_dfs[sat_id] = df

    def _load_synthetic(self, csv_path: str):
        """Load synthetic data (internal column format)."""
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values(["satellite_id", "timestamp"])
        for sat_id, group in df.groupby("satellite_id"):
            group = group.reset_index(drop=True)
            split = TRAIN_DAYS * SEQUENCE_LENGTH
            self.train_dfs[sat_id] = group.iloc[:split]
            self.test_dfs[sat_id]  = group.iloc[split:split + TEST_STEPS]
        print(f"  Loaded synthetic data: {list(self.train_dfs.keys())}")

    # ─────────────────────────────────────────────────────────────────────
    # Column detection + scaler
    # ─────────────────────────────────────────────────────────────────────

    def _set_error_columns(self):
        """Detect which error columns are present across all loaded data."""
        sample = next(iter(self.train_dfs.values()))
        cols = set(sample.columns)

        if "clock_error_m" in cols:
            self.error_columns = [c for c in ISRO_ERROR_COLS if c in cols]
        elif "clock_error_ns" in cols:
            self.error_columns = [c for c in SYNTHETIC_ERROR_COLS if c in cols]
        else:
            exclude = {"timestamp", "satellite_id", "satellite_type"}
            self.error_columns = [
                c for c in sample.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(sample[c])
            ]

        print(f"  Error columns: {self.error_columns}")

    def _prepare_scalers(self):
        """Fit RobustScaler (median/IQR) per satellite per error column on training data.
        RobustScaler is preferred over StandardScaler for GNSS data which has
        heavy-tailed distributions with large occasional spikes.
        """
        for sat_id, train_df in self.train_dfs.items():
            self.scalers[sat_id] = {}
            for col in self.error_columns:
                if col not in train_df.columns:
                    continue
                scaler = RobustScaler(quantile_range=(10, 90))  # robust to GNSS spikes
                vals = train_df[col].values.reshape(-1, 1)
                scaler.fit(vals)
                self.scalers[sat_id][col] = scaler

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def get_default_error_col(self) -> str:
        """Return primary error column name."""
        for preferred in ["clock_error_m", "clock_error_ns"]:
            if preferred in self.error_columns:
                return preferred
        return self.error_columns[0] if self.error_columns else "clock_error_m"

    def get_satellite_data(
        self,
        sat_id: str,
        error_col: Optional[str] = None,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Return (train_series, test_series, scaler) for one satellite.

        Parameters
        ----------
        sat_id    : satellite identifier
        error_col : column to return (auto-detected if None)
        normalize : apply StandardScaler (fit on train)
        """
        if error_col is None:
            error_col = self.get_default_error_col()

        if error_col not in self.error_columns:
            raise ValueError(f"Column '{error_col}' not in {self.error_columns}")

        train_vals = self.train_dfs[sat_id][error_col].values.astype(np.float32)
        test_vals  = self.test_dfs.get(sat_id, self.train_dfs[sat_id].iloc[-TEST_STEPS:])[error_col].values.astype(np.float32)

        scaler = self.scalers[sat_id].get(error_col)
        if scaler is None:
            scaler = RobustScaler(quantile_range=(10, 90))
            scaler.fit(train_vals.reshape(-1, 1))
            self.scalers[sat_id][error_col] = scaler

        if normalize:
            train_vals = scaler.transform(train_vals.reshape(-1, 1)).flatten().astype(np.float32)
            test_vals  = scaler.transform(test_vals.reshape(-1, 1)).flatten().astype(np.float32)

        return train_vals, test_vals, scaler

    def get_all_satellites(
        self,
        error_col: Optional[str] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, StandardScaler]]:
        """Get train/test data for all satellites."""
        if error_col is None:
            error_col = self.get_default_error_col()
        return {sat_id: self.get_satellite_data(sat_id, error_col)
                for sat_id in self.satellite_ids}

    def get_satellite_type(self, sat_id: str) -> str:
        """Return 'GEO' or 'MEO' for a satellite."""
        df = self.train_dfs.get(sat_id)
        if df is not None and "satellite_type" in df.columns:
            return df["satellite_type"].iloc[0]
        return "UNKNOWN"

    def summary(self) -> dict:
        return {
            "n_satellites": len(self.satellite_ids),
            "satellite_ids": self.satellite_ids,
            "n_geo": sum(1 for s in self.satellite_ids if self.get_satellite_type(s) == "GEO"),
            "n_meo": sum(1 for s in self.satellite_ids if self.get_satellite_type(s) == "MEO"),
            "total_train_rows": sum(len(df) for df in self.train_dfs.values()),
            "total_test_rows":  sum(len(df) for df in self.test_dfs.values()),
            "error_columns":    self.error_columns,
            "satellite_details": {
                sat_id: {
                    "type":        self.get_satellite_type(sat_id),
                    "train_steps": len(self.train_dfs[sat_id]),
                    "test_steps":  len(self.test_dfs.get(sat_id, pd.DataFrame())),
                    "train_start": str(self.train_dfs[sat_id]["timestamp"].iloc[0]),
                    "test_end":    str(self.test_dfs[sat_id]["timestamp"].iloc[-1])
                    if sat_id in self.test_dfs else "N/A",
                }
                for sat_id in self.satellite_ids
            }
        }


# =============================================================================
# CLI test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Loading ISRO Dataset")
    print("=" * 60)

    dataset = GNSSDataset()

    print("\nDataset Summary:")
    s = dataset.summary()
    for k, v in s.items():
        if k != "satellite_details":
            print(f"  {k}: {v}")

    print("\nPer-satellite details:")
    for sat_id in dataset.satellite_ids:
        ecol = dataset.get_default_error_col()
        train, test, scaler = dataset.get_satellite_data(sat_id, ecol)
        print(f"\n  {sat_id} ({dataset.get_satellite_type(sat_id)}) | {ecol}")
        print(f"    train: {train.shape}  mean={train.mean():.4f}  std={train.std():.4f}")
        print(f"    test:  {test.shape}   mean={test.mean():.4f}   std={test.std():.4f}")
