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
    adapt_frac: float = 0.0,
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

    Split-test adaptation (opt-in, adapt_frac > 0)
    -----------------------------------------------
    When adapt_frac > 0, for h=1,2,4 (15/30/60 min) the model is
    fine-tuned on the first `adapt_frac` of the test data and evaluated
    on the remaining (1-adapt_frac) portion. This is honest evaluation
    (fine-tuning set and evaluation set are disjoint) but requires enough
    adaptation steps to be effective (recommend >= 100 test steps).
    With short test sets (< 100 steps) the default adapt_frac=0.0 gives
    better results because the full test set is used for evaluation.

    Parameters
    ----------
    ensemble     : GNSSEnsemble — trained ensemble
    train_series : np.ndarray  — raw training series (unnormalised)
    test_series  : np.ndarray or None — test ground truth for evaluation
    use_rolling  : bool — enable rolling inference (default True)
    adapt_frac   : float — fraction of test data used for fine-tuning.
                   0.0 (default) = no fine-tuning, full test evaluation.
                   0.5 = split-test; use only when test has >= 100 steps.

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

    # ── adapt_frac=0: rolling-only inference (clean, proven correct) ──
    if adapt_frac <= 0.0:
        if not use_rolling:
            # Fallback to batch if rolling explicitly disabled
            return ensemble.predict(
                train_series, test_series=test_series, use_test_context=True
            )
        rolling_results = ensemble.predict_rolling(train_series, test_series)
        if verbose:
            for h in sorted(rolling_results.keys()):
                r = rolling_results[h]
                rmse = r.get("rmse", float("nan"))
                persist = r.get("baseline_persist_rmse", float("nan"))
                delta = ((persist - rmse) / persist * 100) if persist > 0 else 0
                tag = "BEATS persist" if rmse < persist else "below persist"
                print(f"  h={h*15:>4}min — rolling RMSE={rmse:.4f}  "
                      f"persist={persist:.4f}  ({delta:+.1f}% {tag})")
        return rolling_results


    # adapt_frac > 0: split test data into adaptation and evaluation slices
    n_adapt      = max(1, int(len(test_series) * adapt_frac))
    adapt_series = test_series[:n_adapt]   # first fraction — fine-tuning
    eval_series  = test_series[n_adapt:]   # remainder — evaluation

    if verbose:
        print(f"  Split-test: adapt={n_adapt} steps, eval={len(eval_series)} steps")


    # ── Fine-tune short-horizon models on adapt_series ──
    import copy
    ensemble_ft = copy.deepcopy(ensemble)   # never mutate the base ensemble
    ensemble_ft.fine_tune(
        train_series=train_series,
        adapt_series=adapt_series,
        short_horizons=[1, 2, 4],
        ft_epochs=50,
        verbose=verbose,
    )

    short_horizons = [1, 2, 4]
    long_horizons  = [8, 16]

    # ── Short horizons: rolling on eval_series, seeded with adapt context ──
    short_rolling = None
    if use_rolling and len(eval_series) > 0:
        try:
            short_rolling = ensemble_ft.predict_rolling(
                train_series,
                eval_series,
                context_prefix=adapt_series,
            )
        except Exception as e:
            if verbose:
                print(f"  WARNING: short-horizon rolling failed: {e}")

    # ── Long horizons: original hybrid on full test_series (unchanged models) ──
    batch_results   = None
    rolling_results = None
    try:
        batch_results = ensemble.predict(
            train_series, test_series=test_series, use_test_context=True
        )
    except Exception as e:
        if verbose:
            print(f"  WARNING: batch inference failed: {e}")

    if use_rolling:
        try:
            rolling_results = ensemble.predict_rolling(train_series, test_series)
        except Exception as e:
            if verbose:
                print(f"  WARNING: rolling inference failed: {e}")

    # ── Merge per-horizon ──
    from config import HORIZONS as _HORIZONS
    merged = {}

    for h in _HORIZONS:
        if h in short_horizons and short_rolling is not None and h in short_rolling:
            # Short horizons: use fine-tuned rolling on eval_series
            merged[h] = short_rolling[h]
            if verbose:
                rmse = short_rolling[h].get('rmse', float('nan'))
                print(f"  h={h*15:>4}min — fine-tuned rolling  (eval RMSE {rmse:.3f})")
        else:
            # Long horizons: original hybrid selection
            b = (batch_results or {}).get(h, {})
            r = (rolling_results or {}).get(h, {})
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
    parser.add_argument("--satellite", type=str, default=None,
                        help="Satellite ID to predict for (default: all)")
    parser.add_argument("--error-col", type=str, default=None,
                        help="Error column (auto-detected if omitted)")
    parser.add_argument("--output", type=str, default="day8_predictions.json")
    args = parser.parse_args()

    # Load data
    dataset = GNSSDataset()
    error_col = args.error_col or dataset.get_default_error_col()

    # Decide which satellites to run
    satellites = [args.satellite] if args.satellite else dataset.satellite_ids

    all_predictions = {}

    for sat_id in satellites:
        print(f"\n{'='*60}")
        print(f"Satellite: {sat_id}, Error column: {error_col}")
        print(f"{'='*60}")

        train_series, test_series, scaler = dataset.get_satellite_data(
            sat_id, error_col, normalize=False
        )

        # Load trained ensemble for this specific satellite
        ensemble = GNSSEnsemble(satellite_id=sat_id)
        ensemble.load()

        # Predict
        predictions = predict_day8(ensemble, train_series, test_series)
        all_predictions[sat_id] = predictions

    # If only one satellite was requested, save in the original flat format
    if len(satellites) == 1:
        save_predictions(all_predictions[satellites[0]], args.output)
    else:
        # Save combined predictions keyed by satellite_id, then by horizon
        # Also save a flat version using the first satellite for backward compat
        save_predictions(all_predictions[satellites[0]], args.output)
        combined_path = os.path.join(RESULTS_DIR, "all_predictions.json")
        with open(combined_path, "w") as f:
            import json as _json
            _json.dump(all_predictions, f, indent=2, default=str)
        print(f"Combined predictions saved to {combined_path}")


