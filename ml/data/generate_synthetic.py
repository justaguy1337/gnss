"""
Synthetic GNSS Error Data Generator
====================================
Generates realistic 8-day satellite clock & ephemeris error data
for testing the pipeline when real ISRO data is unavailable.

Error model:
  - Clock bias: quadratic drift + orbital sinusoid + random walk + noise
  - Ephemeris:  solar radiation pressure + gravity harmonics + thermal cycling
  - GEO sats:   ~24hr period, higher stability, smaller errors
  - MEO sats:   ~12hr period (GPS-like), larger drift rates
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, SAMPLE_INTERVAL_MIN


def generate_satellite_errors(
    sat_id: str,
    sat_type: str,
    n_days: int = 8,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic error time-series for one satellite.

    Parameters
    ----------
    sat_id : str      — satellite identifier (e.g., "G01")
    sat_type : str    — "GEO" or "MEO"
    n_days : int      — total days of data (7 train + 1 test)
    seed : int        — random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, satellite_id, satellite_type,
        clock_error_ns, radial_error_m,
        along_track_error_m, cross_track_error_m
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    steps_per_day = 96  # 24hr / 15min
    n_steps = n_days * steps_per_day
    t = np.arange(n_steps, dtype=np.float64)

    # Time in hours
    t_hours = t * (SAMPLE_INTERVAL_MIN / 60.0)

    # Orbital period parameters
    if sat_type == "GEO":
        orbital_period_steps = 96     # ~24hr for GEO
        clock_drift_rate = 0.02       # slower drift
        clock_noise_std = 0.5         # ns
        eph_base_amplitude = 0.3      # meters (smaller for GEO)
        random_walk_std = 0.01
    else:  # MEO
        orbital_period_steps = 48     # ~12hr for MEO (GPS-like)
        clock_drift_rate = 0.05       # faster drift
        clock_noise_std = 1.0         # ns
        eph_base_amplitude = 1.0      # meters
        random_walk_std = 0.03

    # ── Clock bias error (nanoseconds) ──
    # Quadratic drift + orbital sinusoid + random walk + white noise
    clock_drift = clock_drift_rate * (t / steps_per_day) ** 2
    clock_orbital = 2.0 * np.sin(2 * np.pi * t / orbital_period_steps)
    clock_harmonic = 0.8 * np.sin(4 * np.pi * t / orbital_period_steps)
    clock_walk = random_walk_std * np.cumsum(rng.randn(n_steps))
    clock_noise = clock_noise_std * rng.randn(n_steps)

    # Periodic upload corrections (every ~2 hours = 8 steps)
    upload_interval = 8
    for i in range(upload_interval, n_steps, upload_interval):
        # Correction reduces accumulated drift
        correction = -0.3 * clock_drift[i]
        clock_drift[i:] += correction

    clock_error = clock_drift + clock_orbital + clock_harmonic + clock_walk + clock_noise

    # ── Ephemeris errors (meters) ──
    # Radial: smallest, dominated by gravity harmonics
    radial = (
        eph_base_amplitude * 0.3 * np.sin(2 * np.pi * t / orbital_period_steps + 0.5)
        + eph_base_amplitude * 0.1 * np.sin(4 * np.pi * t / orbital_period_steps)
        + 0.1 * rng.randn(n_steps)
    )

    # Along-track: largest, accumulates from velocity errors
    along_track = (
        eph_base_amplitude * 1.0 * np.sin(2 * np.pi * t / orbital_period_steps + 1.2)
        + eph_base_amplitude * 0.4 * np.sin(4 * np.pi * t / orbital_period_steps + 0.8)
        + 0.02 * np.cumsum(rng.randn(n_steps))  # velocity drift
        + 0.2 * rng.randn(n_steps)
    )

    # Cross-track: medium, affected by inclination perturbations
    cross_track = (
        eph_base_amplitude * 0.5 * np.sin(2 * np.pi * t / orbital_period_steps + 2.0)
        + eph_base_amplitude * 0.15 * np.cos(2 * np.pi * t / (orbital_period_steps * 2))
        + 0.15 * rng.randn(n_steps)
    )

    # Solar radiation pressure perturbation (daily pattern)
    srp = 0.2 * eph_base_amplitude * np.sin(2 * np.pi * t / steps_per_day + 0.3)
    along_track += srp
    radial += 0.5 * srp

    # Build timestamps
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(
        start=start_time,
        periods=n_steps,
        freq=f"{SAMPLE_INTERVAL_MIN}min"
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "satellite_id": sat_id,
        "satellite_type": sat_type,
        "clock_error_ns": clock_error.astype(np.float32),
        "radial_error_m": radial.astype(np.float32),
        "along_track_error_m": along_track.astype(np.float32),
        "cross_track_error_m": cross_track.astype(np.float32),
    })


def generate_full_dataset(
    n_meo: int = 4,
    n_geo: int = 2,
    n_days: int = 8,
    save: bool = True
) -> pd.DataFrame:
    """
    Generate a complete synthetic multi-satellite dataset.

    Parameters
    ----------
    n_meo : int  — number of MEO satellites
    n_geo : int  — number of GEO satellites
    n_days : int — total days (7 train + 1 test)
    save : bool  — save to CSV

    Returns
    -------
    pd.DataFrame — concatenated data for all satellites
    """
    frames = []

    for i in range(n_meo):
        sat_id = f"MEO-{i+1:02d}"
        df = generate_satellite_errors(sat_id, "MEO", n_days, seed=42 + i)
        frames.append(df)
        print(f"  Generated {sat_id}: {len(df)} steps, "
              f"clock range [{df['clock_error_ns'].min():.2f}, {df['clock_error_ns'].max():.2f}] ns")

    for i in range(n_geo):
        sat_id = f"GEO-{i+1:02d}"
        df = generate_satellite_errors(sat_id, "GEO", n_days, seed=100 + i)
        frames.append(df)
        print(f"  Generated {sat_id}: {len(df)} steps, "
              f"clock range [{df['clock_error_ns'].min():.2f}, {df['clock_error_ns'].max():.2f}] ns")

    dataset = pd.concat(frames, ignore_index=True)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, "gnss_errors_synthetic.csv")
        dataset.to_csv(path, index=False)
        print(f"\nSaved {len(dataset)} rows to {path}")

    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic GNSS error data")
    parser.add_argument("--n-meo", type=int, default=4, help="Number of MEO satellites")
    parser.add_argument("--n-geo", type=int, default=2, help="Number of GEO satellites")
    parser.add_argument("--days", type=int, default=8, help="Total days of data")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    args = parser.parse_args()

    print("=" * 55)
    print("Generating Synthetic GNSS Error Data")
    print("=" * 55)

    df = generate_full_dataset(args.n_meo, args.n_geo, args.days)

    if args.validate:
        print("\n── Validation ──")
        for sat_id in df["satellite_id"].unique():
            sat_df = df[df["satellite_id"] == sat_id]
            n = len(sat_df)
            expected = args.days * 96
            status = "[OK]" if n == expected else "[!!]"
            print(f"  {status} {sat_id}: {n} steps (expected {expected})")

            # Check no NaN
            nan_count = sat_df.isna().sum().sum()
            print(f"    NaN values: {nan_count}")

            # Check time intervals
            dt = sat_df["timestamp"].diff().dropna()
            uniform = (dt == pd.Timedelta(minutes=15)).all()
            print(f"    Uniform 15-min intervals: {uniform}")

        print("\nValidation complete.")
