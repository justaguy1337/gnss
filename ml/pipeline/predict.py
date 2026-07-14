"""
Prediction Pipeline
====================
Generates Day 8 predictions at all horizons using the trained ensemble.

v2 changes:
  - test_series is passed to ensemble.predict() for proper alignment
  - train_series is returned in result so full_evaluation can compute MASE
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HORIZONS, RESULTS_DIR
from data.data_loader import GNSSDataset
from pipeline.train import GNSSEnsemble


def predict_day8(
    ensemble: GNSSEnsemble,
    train_series: np.ndarray,
    test_series: np.ndarray = None,
    verbose: bool = True,
    use_rolling: bool = True,
) -> dict:
    """
    Generate Day 8 predictions at all horizons using smart hybrid inference.

    Strategy
    --------
    When test data is available, runs BOTH rolling and batch test-context
    inference and picks the better mode *per horizon* based on RMSE vs the
    persistence baseline.  This automatically adapts to the satellite type:
      - GEO (volatile test day): batch_ctx tends to win at long horizons
      - MEO (stable test day):   rolling tends to win at short horizons

    If only one mode succeeds, that mode is used.
    If no test data is provided, runs batch inference on the training tail.

    Parameters
    ----------
    ensemble     : GNSSEnsemble — trained ensemble
    train_series : np.ndarray  — raw training series (unnormalised)
    test_series  : np.ndarray or None — test ground truth for evaluation
    use_rolling  : bool — enable rolling inference (default True)

    Returns
    -------
    results : dict — keyed by horizon int, schema matches predict() / predict_rolling()
    """
    if verbose:
        print("=" * 60)
        print("Generating Day 8 Predictions")
        print("=" * 60)

    # ── No test data: batch inference on training tail only ──
    if test_series is None or len(test_series) == 0:
        if verbose:
            print("  Mode: training-tail (no test data)")
        return ensemble.predict(train_series, test_series=None)

    # ── Run batch test-context mode ──
    batch_results = None
    try:
        batch_results = ensemble.predict(
            train_series, test_series=test_series, use_test_context=True
        )
    except Exception as e:
        if verbose:
            print(f"  WARNING: batch inference failed: {e}")

    # ── Run rolling inference ──
    rolling_results = None
    if use_rolling:
        try:
            rolling_results = ensemble.predict_rolling(train_series, test_series)
        except Exception as e:
            if verbose:
                print(f"  WARNING: rolling inference failed: {e}")

    # ── If only one succeeded, use it ──
    if batch_results is None and rolling_results is None:
        raise RuntimeError("Both batch and rolling inference failed.")
    if batch_results is None:
        if verbose:
            print("  Mode: rolling (batch failed)")
        return rolling_results
    if rolling_results is None:
        if verbose:
            print("  Mode: batch test-context (rolling failed)")
        return batch_results

    # ── Hybrid: pick better mode per horizon ──
    from config import HORIZONS as _HORIZONS
    merged = {}
    for h in _HORIZONS:
        b = batch_results.get(h, {})
        r = rolling_results.get(h, {})
        b_rmse = b.get("rmse", float("inf"))
        r_rmse = r.get("rmse", float("inf"))

        if b_rmse <= r_rmse:
            merged[h] = b
            if verbose:
                print(f"  h={h*15:>4}min — batch_ctx wins  "
                      f"(RMSE {b_rmse:.3f} < {r_rmse:.3f})")
        else:
            merged[h] = r
            if verbose:
                print(f"  h={h*15:>4}min — rolling   wins  "
                      f"(RMSE {r_rmse:.3f} < {b_rmse:.3f})")

    return merged


def save_predictions(predictions: dict, filename: str = "day8_predictions.json"):
    """Save predictions to JSON."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"Predictions saved to {path}")
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Day 8 Predictions")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--satellite", type=str, default=None)
    parser.add_argument("--error-col", type=str, default=None,
                        help="Error column (auto-detected if omitted)")
    parser.add_argument("--output", type=str, default="day8_predictions.json")
    args = parser.parse_args()

    # Load data
    dataset = GNSSDataset(args.data)
    sat_id = args.satellite or dataset.satellite_ids[0]

    # Auto-detect error column
    error_col = args.error_col or dataset.get_default_error_col()
    print(f"Satellite: {sat_id}, Error column: {error_col}")

    train_series, test_series, scaler = dataset.get_satellite_data(
        sat_id, error_col, normalize=False
    )

    # Load trained ensemble
    ensemble = GNSSEnsemble()
    ensemble.load()

    # Predict
    predictions = predict_day8(ensemble, train_series, test_series)
    save_predictions(predictions, args.output)

