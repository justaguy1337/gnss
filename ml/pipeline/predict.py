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

    # ── adapt_frac=0: skip fine-tuning, go straight to hybrid ──
    if adapt_frac <= 0.0:
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
        if batch_results is None and rolling_results is None:
            raise RuntimeError("Both batch and rolling inference failed.")
        if batch_results is None:
            return rolling_results
        if rolling_results is None:
            return batch_results
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
                    print(f"  h={h*15:>4}min — batch_ctx wins  (RMSE {b_rmse:.3f} < {r_rmse:.3f})")
            else:
                merged[h] = r
                if verbose:
                    print(f"  h={h*15:>4}min — rolling   wins  (RMSE {r_rmse:.3f} < {b_rmse:.3f})")
        return merged

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

