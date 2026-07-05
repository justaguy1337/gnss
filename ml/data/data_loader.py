"""
Data Loader for GNSS Error Prediction
======================================
Loads the real ISRO dataset from d:/github/gnss/dataset/.

Dataset files (all in DATASET_DIR):
  Train: DATA_GEO_Train.csv, DATA_MEO_Train.csv, DATA_MEO_Train2.csv
  Test:  DATA_GEO_Test.csv, DATA_MEO_Test.csv, DATA_MEO_Test2.csv

All files are resampled to a uniform 15-minute grid (forward-fill + interpolate).

NO synthetic/mock/fallback data — if the ISRO dataset is missing, an
explicit FileNotFoundError is raised immediately. Nothing silently falls
back to generated data.

Schema validation (`validate_schema`) is run before any training or
inference step to catch column mismatches early with a clear error message.
"""

from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEQUENCE_LENGTH, TRAIN_DAYS, TEST_STEPS,
    DATA_DIR, DATASET_DIR, TRAIN_DIR, TEST_DIR, DATASET_FILES,
    RESAMPLE_INTERVAL, SAMPLE_INTERVAL_MIN
)

# ─────────────────────────────────────────────
# Real ISRO column names → internal names
# ─────────────────────────────────────────────
ISRO_COLUMN_MAP = {
    "utc_time":          "timestamp",
    "x_error (m)":       "x_error_m",
    "y_error (m)":       "y_error_m",
    "y_error  (m)":      "y_error_m",   # MEO_Train.csv has extra trailing space
    "z_error (m)":       "z_error_m",
    "satclockerror (m)": "clock_error_m",
}

# Canonical internal column names after renaming
ISRO_ERROR_COLS = ["clock_error_m", "x_error_m", "y_error_m", "z_error_m"]

# Required numeric columns that must be present in both train and test
REQUIRED_ERROR_COLS = {"x_error_m", "y_error_m", "z_error_m", "clock_error_m"}


# =============================================================================
# Schema Validation
# =============================================================================

def validate_schema(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    context: str = "",
) -> None:
    """
    Validate that test_df has compatible schema with train_df.

    Checks:
      - All required numeric error columns present in both DataFrames
      - All shared error columns are numeric in the test set

    Raises
    ------
    ValueError with a precise diff on mismatch — never a generic downstream error.
    """
    train_cols = set(train_df.columns) & REQUIRED_ERROR_COLS
    test_cols  = set(test_df.columns)  & REQUIRED_ERROR_COLS

    missing_in_test = train_cols - test_cols
    extra_in_test   = test_cols  - train_cols   # extra cols in test are OK but flagged

    errors = []

    if missing_in_test:
        errors.append(
            f"Missing columns in test data: {sorted(missing_in_test)}\n"
            f"  Expected (from training): {sorted(train_cols)}\n"
            f"  Found in test:            {sorted(test_cols)}"
        )

    if extra_in_test:
        # Not fatal — warn only
        print(f"  [schema] Note: test has extra columns not in training: "
              f"{sorted(extra_in_test)} — they will be ignored.")

    # Type check shared columns
    for col in train_cols & test_cols:
        if not pd.api.types.is_numeric_dtype(test_df[col]):
            errors.append(
                f"Column '{col}' has non-numeric dtype in test data "
                f"(dtype={test_df[col].dtype}, expected float64). "
                f"Check for string values or missing-value markers."
            )

    if errors:
        ctx = f" ({context})" if context else ""
        raise ValueError(
            f"Schema mismatch{ctx}:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


# =============================================================================
# Internal helpers
# =============================================================================

def _read_isro_file(
    path: str,
    satellite_id: str = "UNKNOWN",
    satellite_type: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Read one ISRO CSV, strip column names, rename to internal names,
    parse timestamps. Returns un-resampled DataFrame sorted by time.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()     # handles trailing spaces like "y_error  (m)"
    df = df.rename(columns=ISRO_COLUMN_MAP)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"No timestamp column in '{path}'.\n"
            f"  Columns found: {list(df.columns)}\n"
            f"  Expected 'utc_time' (maps to 'timestamp')."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, format="mixed")
    df["satellite_id"]   = satellite_id
    df["satellite_type"] = satellite_type
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _resample_to_uniform(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """
    Resample an irregular time-series DataFrame to a uniform grid.
    Uses linear time-interpolation for gaps, forward/back-fill for edges.
    """
    df = df.copy()
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    meta_cols = {"satellite_id", "satellite_type"}
    num_cols  = [c for c in df.columns if c not in meta_cols]

    sat_id   = df["satellite_id"].iloc[0]   if "satellite_id"   in df.columns else "UNKNOWN"
    sat_type = df["satellite_type"].iloc[0] if "satellite_type" in df.columns else "UNKNOWN"

    df_num = df[num_cols].resample(freq).mean()
    df_num = df_num.interpolate(method="time", limit=8)   # up to 2-hr gap via interpolation
    df_num = df_num.ffill().bfill()

    df_num["satellite_id"]   = sat_id
    df_num["satellite_type"] = sat_type
    return df_num.reset_index().rename(columns={"index": "timestamp"})


def _concat_files(
    paths: List[str],
    satellite_id: str,
    satellite_type: str,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Load multiple ISRO CSV files for the same satellite type,
    concatenate by timestamp, deduplicate, then resample.
    """
    dfs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: file not found: {path}")
            continue
        dfs.append(_read_isro_file(path, satellite_id=satellite_id,
                                   satellite_type=satellite_type))
    if not dfs:
        raise FileNotFoundError(
            f"No files loaded for satellite '{satellite_id}'. "
            f"Searched: {paths}"
        )

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    ).reset_index(drop=True)

    return _resample_to_uniform(combined, freq=freq)


# =============================================================================
# GNSSDataset — main public class
# =============================================================================

class GNSSDataset:
    """
    Manages GNSS error data for training and evaluation.

    On construction, always loads the real ISRO dataset from DATASET_DIR.
    No synthetic fallback exists — a missing dataset raises FileNotFoundError.

    For test-data uploads, use `load_test_csv(path)` which validates schema
    against the loaded training data before accepting the file.
    """

    def __init__(self):
        self.scalers:   Dict[str, Dict[str, RobustScaler]] = {}
        self.train_dfs: Dict[str, pd.DataFrame] = {}
        self.test_dfs:  Dict[str, pd.DataFrame] = {}

        self._load_isro_dataset()
        self.satellite_ids = list(self.train_dfs.keys())
        self._set_error_columns()
        self._prepare_scalers()

    # ─────────────────────────────────────────────────────────────────────
    # Dataset loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_isro_dataset(self):
        """
        Scan DATASET_DIR for all train/test file pairs defined in
        DATASET_FILES (config.py) and load them.

        Strategy: Concatenate all train files per satellite type into one
        combined series (GEO → GEO_Train; MEO → MEO_Train + MEO_Train2).
        This matches the sequential/combined architecture of GNSSEnsemble.
        """
        if not os.path.isdir(DATASET_DIR):
            raise FileNotFoundError(
                f"ISRO dataset directory not found: '{DATASET_DIR}'\n"
                f"Expected the dataset folder at: {DATASET_DIR}\n"
                f"Please ensure the folder exists and contains the ISRO CSV files."
            )
        if not os.path.isdir(TRAIN_DIR):
            raise FileNotFoundError(
                f"Training data directory not found: '{TRAIN_DIR}'\n"
                f"Expected train files at: {TRAIN_DIR}\n"
                f"Please ensure the folder exists and contains the ISRO training CSV files."
            )
        if not os.path.isdir(TEST_DIR):
            raise FileNotFoundError(
                f"Test data directory not found: '{TEST_DIR}'\n"
                f"Expected test files at: {TEST_DIR}\n"
                f"Please ensure the folder exists and contains the ISRO test CSV files."
            )

        print(f"Loading ISRO dataset...")
        print(f"  Train dir: {TRAIN_DIR}")
        print(f"  Test  dir: {TEST_DIR}")
        loaded = 0

        for sat_id, cfg in DATASET_FILES.items():
            sat_type  = cfg["type"]
            t_dir     = cfg.get("train_dir", TRAIN_DIR)
            ts_dir    = cfg.get("test_dir",  TEST_DIR)
            train_paths = [os.path.join(t_dir,  f) for f in cfg["train"]]
            test_paths  = [os.path.join(ts_dir, f) for f in cfg["test"]]

            try:
                train_df = _concat_files(train_paths, sat_id, sat_type,
                                         freq=RESAMPLE_INTERVAL)
                test_df  = _concat_files(test_paths,  sat_id, sat_type,
                                         freq=RESAMPLE_INTERVAL)

                # Schema check: train vs bundled test
                validate_schema(train_df, test_df,
                                context=f"{sat_id} train vs bundled test")

                self.train_dfs[sat_id] = train_df
                self.test_dfs[sat_id]  = test_df
                loaded += 1

                print(f"  {sat_id} ({sat_type}): "
                      f"train={len(train_df)} steps "
                      f"[{train_df['timestamp'].iloc[0].date()} to "
                      f"{train_df['timestamp'].iloc[-1].date()}], "
                      f"test={len(test_df)} steps "
                      f"[{test_df['timestamp'].iloc[0].date()}]")

            except Exception as e:
                print(f"  ERROR loading {sat_id}: {e}")
                raise   # hard fail — no silent skipping

        if loaded == 0:
            raise FileNotFoundError(
                f"No satellite data could be loaded from '{DATASET_DIR}'. "
                f"Check that the ISRO CSV files are present."
            )

    # ─────────────────────────────────────────────────────────────────────
    # Test-data upload (user-supplied test CSV)
    # ─────────────────────────────────────────────────────────────────────

    def load_test_csv(self, path: str) -> str:
        """
        Load a user-supplied test CSV and replace the in-memory test set
        for the matching satellite type.

        Validates:
          1. File is parseable as ISRO format
          2. Schema matches training data (raises ValueError on mismatch)

        Returns the satellite_id that was updated.

        Raises
        ------
        ValueError : on schema mismatch (with specific column diff)
        FileNotFoundError : if path does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Uploaded file not found: {path}")

        fname    = os.path.basename(path).upper()
        sat_id   = "GEO" if "GEO" in fname else "MEO"
        sat_type = sat_id

        raw_df   = _read_isro_file(path, satellite_id=sat_id, satellite_type=sat_type)
        test_df  = _resample_to_uniform(raw_df, freq=RESAMPLE_INTERVAL)

        # Validate against the training schema
        if sat_id not in self.train_dfs:
            raise ValueError(
                f"Cannot validate test CSV: no training data loaded for "
                f"satellite '{sat_id}'. Training data must be loaded first."
            )

        validate_schema(
            self.train_dfs[sat_id], test_df,
            context=f"uploaded test file vs {sat_id} training data"
        )

        self.test_dfs[sat_id] = test_df
        print(f"  Test data updated for {sat_id}: {len(test_df)} steps")
        return sat_id

    # ─────────────────────────────────────────────────────────────────────
    # Column detection + scalers
    # ─────────────────────────────────────────────────────────────────────

    def _set_error_columns(self):
        """Detect which ISRO error columns are present in loaded data."""
        sample = next(iter(self.train_dfs.values()))
        self.error_columns = [c for c in ISRO_ERROR_COLS if c in sample.columns]

        if not self.error_columns:
            raise ValueError(
                f"No recognised error columns found in dataset.\n"
                f"  Expected any of: {ISRO_ERROR_COLS}\n"
                f"  Found: {list(sample.columns)}"
            )

        print(f"  Error columns: {self.error_columns}")

    def _prepare_scalers(self):
        """
        Fit RobustScaler(10-90 IQR) per satellite per error column on
        training data. RobustScaler is used instead of StandardScaler
        because GNSS clock errors are heavy-tailed with occasional large spikes.
        """
        for sat_id, train_df in self.train_dfs.items():
            self.scalers[sat_id] = {}
            for col in self.error_columns:
                if col not in train_df.columns:
                    continue
                scaler = RobustScaler(quantile_range=(10, 90))
                scaler.fit(train_df[col].values.reshape(-1, 1))
                self.scalers[sat_id][col] = scaler

    # ─────────────────────────────────────────────────────────────────────
    # Public data access
    # ─────────────────────────────────────────────────────────────────────

    def get_default_error_col(self) -> str:
        """Return primary error column (clock_error_m preferred)."""
        for preferred in ["clock_error_m", "x_error_m"]:
            if preferred in self.error_columns:
                return preferred
        return self.error_columns[0]

    def get_satellite_data(
        self,
        sat_id: str,
        error_col: Optional[str] = None,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
        """
        Return (train_series, test_series, scaler) for one satellite.

        Parameters
        ----------
        sat_id    : satellite identifier ('GEO' or 'MEO')
        error_col : column name, auto-detected if None
        normalize : apply RobustScaler fitted on training data
        """
        if error_col is None:
            error_col = self.get_default_error_col()
        if sat_id not in self.train_dfs:
            raise KeyError(
                f"Unknown satellite '{sat_id}'. "
                f"Available: {self.satellite_ids}"
            )

        train_vals = self.train_dfs[sat_id][error_col].values.astype(np.float32)
        test_df    = self.test_dfs.get(sat_id)
        test_vals  = (test_df[error_col].values.astype(np.float32)
                      if test_df is not None and error_col in test_df.columns
                      else np.array([], dtype=np.float32))

        scaler = self.scalers[sat_id].get(error_col)
        if scaler is None:
            scaler = RobustScaler(quantile_range=(10, 90))
            scaler.fit(train_vals.reshape(-1, 1))
            self.scalers[sat_id][error_col] = scaler

        if normalize:
            train_vals = scaler.transform(
                train_vals.reshape(-1, 1)).flatten().astype(np.float32)
            if len(test_vals) > 0:
                test_vals = scaler.transform(
                    test_vals.reshape(-1, 1)).flatten().astype(np.float32)

        return train_vals, test_vals, scaler

    def get_all_satellites(
        self,
        error_col: Optional[str] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, RobustScaler]]:
        """Get train/test data for all satellites."""
        if error_col is None:
            error_col = self.get_default_error_col()
        return {sat_id: self.get_satellite_data(sat_id, error_col)
                for sat_id in self.satellite_ids}

    def get_satellite_type(self, sat_id: str) -> str:
        """Return 'GEO' or 'MEO'."""
        df = self.train_dfs.get(sat_id)
        if df is not None and "satellite_type" in df.columns:
            return df["satellite_type"].iloc[0]
        return "UNKNOWN"

    def summary(self) -> dict:
        return {
            "n_satellites":   len(self.satellite_ids),
            "satellite_ids":  self.satellite_ids,
            "n_geo":          sum(1 for s in self.satellite_ids
                                  if self.get_satellite_type(s) == "GEO"),
            "n_meo":          sum(1 for s in self.satellite_ids
                                  if self.get_satellite_type(s) == "MEO"),
            "total_train_rows": sum(len(df) for df in self.train_dfs.values()),
            "total_test_rows":  sum(len(df) for df in self.test_dfs.values()),
            "error_columns":    self.error_columns,
            "dataset_dir":      DATASET_DIR,
            "satellite_details": {
                sat_id: {
                    "type":        self.get_satellite_type(sat_id),
                    "train_steps": len(self.train_dfs[sat_id]),
                    "test_steps":  len(self.test_dfs.get(sat_id, pd.DataFrame())),
                    "train_start": str(self.train_dfs[sat_id]["timestamp"].iloc[0]),
                    "train_end":   str(self.train_dfs[sat_id]["timestamp"].iloc[-1]),
                    "test_start":  str(self.test_dfs[sat_id]["timestamp"].iloc[0])
                                   if sat_id in self.test_dfs else "N/A",
                }
                for sat_id in self.satellite_ids
            }
        }


# =============================================================================
# CLI self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Loading ISRO Dataset (no synthetic fallback)")
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

    print("\nSchema validation test (train vs test for each sat):")
    for sat_id in dataset.satellite_ids:
        validate_schema(dataset.train_dfs[sat_id], dataset.test_dfs[sat_id],
                        context=f"{sat_id} self-check")
        print(f"  {sat_id}: schema OK")
