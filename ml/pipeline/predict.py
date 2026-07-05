"""
Prediction Pipeline
====================
Generates Day 8 predictions at all horizons using the trained ensemble.

Output: JSON with predictions, uncertainties, per-model contributions
for all 96 Day-8 intervals at each horizon.
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
    verbose: bool = True
) -> dict:
    """
    Generate Day 8 predictions at all horizons.

    Parameters
    ----------
    ensemble : GNSSEnsemble — trained ensemble
    train_series : np.ndarray, shape (672,) — 7-day training data
    test_series : np.ndarray or None — Day 8 ground truth (for evaluation)

    Returns
    -------
    results : dict — complete prediction results
    """
    if verbose:
        print("=" * 60)
        print("Generating Day 8 Predictions")
        print("=" * 60)

    # Use the full training series as context — the ensemble's feature builder
    # will use a sliding window over this to generate features for each step.
    # The last SEQUENCE_LENGTH steps are the most recent context for the forecast.
    predictions = ensemble.predict(train_series)

    # Compare final predictions against test_series ground truth.
    # predictions[h] has len = len(train_series) - SEQUENCE_LENGTH - h + 1
    # test_series has the actual held-out day values.
    # We compare: the LAST min(n_test, n_pred) points of predictions vs test_series.
    if test_series is not None and len(test_series) > 0:
        for h in HORIZONS:
            preds_all = np.array(predictions[h]["predictions"])
            n_test = len(test_series)

            # Take the last n_test predictions (most recent forecasts)
            # that correspond to forecasting from positions near the test boundary
            n = min(len(preds_all), n_test)
            preds_aligned = preds_all[-n:]
            truth_aligned = test_series[:n]

            residuals = truth_aligned - preds_aligned
            predictions[h]["ground_truth"] = truth_aligned.tolist()
            predictions[h]["residuals"]    = residuals.tolist()
            predictions[h]["rmse"]         = float(np.sqrt(np.mean(residuals ** 2)))
            predictions[h]["mae"]          = float(np.mean(np.abs(residuals)))
            predictions[h]["n_test"]       = n

    if verbose:
        for h, info in predictions.items():
            print(f"  Horizon {h*15:>4} min | "
                  f"{info['n_predictions']} predictions | "
                  f"uncertainty range [{min(info['uncertainties']):.4f}, "
                  f"{max(info['uncertainties']):.4f}]")
            if "rmse" in info:
                print(f"                    | RMSE={info['rmse']:.5f} MAE={info['mae']:.5f} "
                      f"(n_test={info.get('n_test', '?')})")

    return predictions


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

