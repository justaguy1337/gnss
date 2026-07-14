"""
Evaluation Module
==================
Comprehensive evaluation matching competition criteria:
  - RMSE, MAE, R² per horizon
  - Normality tests: Shapiro-Wilk, Anderson-Darling, K-S test
  - Q-Q plots and histogram generation
  - Skewness and kurtosis measurements

v2 additions:
  - Naive baselines: persistence (last value) and mean prediction
  - MASE: Mean Absolute Scaled Error relative to persistence baseline
  - Relative improvement over naive baselines
"""

import numpy as np
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HORIZONS, RESULTS_DIR


def naive_baselines(
    y_true: np.ndarray,
    train_series: np.ndarray = None,
) -> dict:
    """
    Compute naive baseline metrics for comparison.

    Baselines:
      - Persistence: always predict the last observed value
      - Mean:        always predict the training mean

    Parameters
    ----------
    y_true       : ground truth test values
    train_series : training series (used for persistence anchor and mean)

    Returns
    -------
    dict with rmse/mae for each baseline, plus MASE denominator
    """
    n = len(y_true)
    results = {}

    if train_series is not None and len(train_series) > 0:
        last_val  = float(train_series[-1])
        train_mean = float(train_series.mean())
    else:
        last_val   = float(y_true[0]) if n > 0 else 0.0
        train_mean = float(y_true.mean()) if n > 0 else 0.0

    # Persistence baseline
    p_pred = np.full(n, last_val, dtype=np.float32)
    p_res  = y_true - p_pred
    results["persistence"] = {
        "rmse": float(np.sqrt(np.mean(p_res ** 2))),
        "mae":  float(np.mean(np.abs(p_res))),
    }

    # Mean baseline
    m_pred = np.full(n, train_mean, dtype=np.float32)
    m_res  = y_true - m_pred
    results["mean"] = {
        "rmse": float(np.sqrt(np.mean(m_res ** 2))),
        "mae":  float(np.mean(np.abs(m_res))),
    }

    # MASE denominator = MAE of persistence (used to scale model MAE)
    results["mase_denominator"] = results["persistence"]["mae"] + 1e-10

    return results


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = None,
    train_series: np.ndarray = None,
) -> dict:
    """
    Full evaluation of prediction quality.

    Parameters
    ----------
    y_true       : ground truth values
    y_pred       : predicted values
    horizon      : prediction horizon (for labeling)
    train_series : training series used for naive baselines and MASE

    Returns
    -------
    metrics : dict with all evaluation metrics
    """
    from scipy.stats import shapiro, normaltest, kstest, skew, kurtosis
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    residuals = y_true - y_pred
    n = len(residuals)

    # Basic metrics
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if n > 1 else 0.0

    # Residual statistics
    res_mean = float(residuals.mean())
    res_std = float(residuals.std())
    res_skewness = float(skew(residuals))
    res_kurtosis = float(kurtosis(residuals))

    # ── Normality tests (critical for competition scoring) ──

    # Shapiro-Wilk (most powerful for small-medium samples)
    try:
        sw_stat, sw_pval = shapiro(residuals[:min(5000, n)])
    except Exception:
        sw_stat, sw_pval = 0.0, 0.0

    # D'Agostino-Pearson (K² test)
    try:
        da_stat, da_pval = normaltest(residuals)
    except Exception:
        da_stat, da_pval = 0.0, 0.0

    # Kolmogorov-Smirnov test against normal
    try:
        ks_stat, ks_pval = kstest(
            (residuals - res_mean) / (res_std + 1e-10),
            'norm'
        )
    except Exception:
        ks_stat, ks_pval = 0.0, 0.0

    # Anderson-Darling
    try:
        import warnings
        from scipy.stats import anderson
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ad_result = anderson(residuals, dist='norm')
        ad_stat = float(ad_result.statistic)
        # Compare against 5% significance level (index 2 = 5%)
        ad_critical_5pct = float(ad_result.critical_values[2])
        ad_is_normal = bool(ad_stat < ad_critical_5pct)
    except Exception:
        ad_stat, ad_critical_5pct, ad_is_normal = 0.0, 0.0, False

    # ── Naive baselines + MASE ──
    baselines = naive_baselines(y_true, train_series)
    persist_mae = baselines["mase_denominator"]
    mase = mae / persist_mae

    # Relative improvement over persistence baseline
    rel_rmse_improve = (baselines["persistence"]["rmse"] - rmse) / \
                        (baselines["persistence"]["rmse"] + 1e-10) * 100.0
    rel_mae_improve  = (baselines["persistence"]["mae"] - mae) / \
                        (baselines["persistence"]["mae"] + 1e-10) * 100.0

    return {
        "horizon": horizon,
        "horizon_min": horizon * 15 if horizon else None,
        "n_samples": n,
        # Prediction quality
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mase": float(mase),
        "rmse_vs_persistence_pct": float(rel_rmse_improve),
        "mae_vs_persistence_pct":  float(rel_mae_improve),
        # Naive baselines
        "baselines": baselines,
        # Residual distribution
        "residual_mean": res_mean,
        "residual_std": res_std,
        "residual_skewness": res_skewness,
        "residual_kurtosis": res_kurtosis,
        # Normality tests
        "shapiro_wilk": {
            "statistic": float(sw_stat),
            "p_value":   float(sw_pval),
            "is_normal": bool(sw_pval > 0.05),
        },
        "dagostino_pearson": {
            "statistic": float(da_stat),
            "p_value":   float(da_pval),
            "is_normal": bool(da_pval > 0.05),
        },
        "kolmogorov_smirnov": {
            "statistic": float(ks_stat),
            "p_value":   float(ks_pval),
            "is_normal": bool(ks_pval > 0.05),
        },
        "anderson_darling": {
            "statistic":    ad_stat,
            "critical_5pct": ad_critical_5pct,
            "is_normal":    ad_is_normal,
        },
    }


def generate_qq_data(residuals: np.ndarray) -> dict:
    """
    Generate Q-Q plot data for frontend visualization.

    Returns
    -------
    dict with 'theoretical' and 'sample' quantiles
    """
    from scipy.stats import norm
    sorted_res = np.sort(residuals)
    n = len(sorted_res)

    # Theoretical quantiles from standard normal
    theoretical = norm.ppf(np.arange(1, n + 1) / (n + 1))

    # Standardize sample quantiles
    sample = (sorted_res - sorted_res.mean()) / (sorted_res.std() + 1e-10)

    return {
        "theoretical": theoretical.tolist(),
        "sample": sample.tolist(),
        "n_points": n,
    }


def generate_histogram_data(
    residuals: np.ndarray,
    n_bins: int = 30
) -> dict:
    """
    Generate histogram data with normal curve overlay for frontend.

    Returns
    -------
    dict with bin edges, counts, and fitted normal PDF
    """
    from scipy.stats import norm

    counts, bin_edges = np.histogram(residuals, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fitted normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x_smooth = np.linspace(bin_edges[0], bin_edges[-1], 200)
    normal_pdf = norm.pdf(x_smooth, mu, sigma)

    return {
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "normal_curve": {
            "x": x_smooth.tolist(),
            "y": normal_pdf.tolist(),
        },
        "fitted_mean": float(mu),
        "fitted_std": float(sigma),
    }


def full_evaluation(
    predictions: dict,
    train_series: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """
    Run full evaluation across all horizons.

    Parameters
    ----------
    predictions  : dict — output from predict_day8()
    train_series : np.ndarray or None — training data for MASE / baseline computation
    verbose      : bool

    Returns
    -------
    evaluation : dict — complete evaluation results
    """
    evaluation = {}

    for h_key, pred_info in predictions.items():
        h = int(h_key) if isinstance(h_key, str) else h_key

        if "ground_truth" not in pred_info:
            if verbose:
                print(f"  Horizon {h}: no ground truth available, skipping")
            continue

        y_true = np.array(pred_info["ground_truth"])
        y_pred = np.array(pred_info["predictions"][:len(y_true)])

        # Metrics
        metrics = evaluate_predictions(y_true, y_pred, horizon=h,
                                       train_series=train_series)
        residuals = y_true - y_pred

        # Visualization data
        metrics["qq_data"]        = generate_qq_data(residuals)
        metrics["histogram_data"] = generate_histogram_data(residuals)

        # Pull in pre-computed baselines from prediction dict if available
        if "baseline_persist_rmse" in pred_info:
            metrics["baselines"]["persistence"]["rmse"] = pred_info["baseline_persist_rmse"]
        if "baseline_mean_rmse" in pred_info:
            metrics["baselines"]["mean"]["rmse"] = pred_info["baseline_mean_rmse"]

        evaluation[h] = metrics

        if verbose:
            sw  = metrics["shapiro_wilk"]
            tag = "[OK] NORMAL" if sw["is_normal"] else "[!!] non-normal"
            mase = metrics.get("mase", float("nan"))
            imp  = metrics.get("rmse_vs_persistence_pct", float("nan"))
            print(f"  h={h:>2} ({h*15:>4}min) | "
                  f"RMSE={metrics['rmse']:.4f} | "
                  f"MAE={metrics['mae']:.4f} | "
                  f"R2={metrics['r2_score']:.4f} | "
                  f"MASE={mase:.3f} | "
                  f"vs persist={imp:+.1f}% | "
                  f"[{tag}]")

    return evaluation


def save_evaluation(evaluation: dict, filename: str = "evaluation_results.json"):
    """Save evaluation results to JSON."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(evaluation, f, indent=2, default=str)
    print(f"Evaluation saved to {path}")
    return path


def print_summary_table(evaluation: dict):
    """Print a formatted summary table including baselines and MASE."""
    print("\n" + "=" * 100)
    print(f"{'Horizon':>9} {'RMSE':>9} {'MAE':>9} {'R2':>7} "
          f"{'MASE':>7} {'vs Persist':>11} {'Persist RMSE':>13} {'Normal':>8}")
    print("-" * 100)

    for h in sorted(evaluation.keys()):
        m  = evaluation[h]
        sw = m["shapiro_wilk"]
        tag = "[OK]" if sw["is_normal"] else "[!!]"
        p_rmse = m.get("baselines", {}).get("persistence", {}).get("rmse", float("nan"))
        imp    = m.get("rmse_vs_persistence_pct", float("nan"))
        mase   = m.get("mase", float("nan"))
        print(f"{m['horizon_min']:>7}min "
              f"{m['rmse']:>9.4f} "
              f"{m['mae']:>9.4f} "
              f"{m['r2_score']:>7.4f} "
              f"{mase:>7.3f} "
              f"{imp:>+10.1f}% "
              f"{p_rmse:>13.4f} "
              f"{tag:>8}")

    print("=" * 100)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GNSS Predictions")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions JSON")
    parser.add_argument("--check-normality", action="store_true")
    parser.add_argument("--check-distribution", action="store_true")
    args = parser.parse_args()

    if args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
    else:
        pred_path = os.path.join(RESULTS_DIR, "day8_predictions.json")
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                predictions = json.load(f)
        else:
            print("No predictions found. Run predict.py first.")
            sys.exit(1)

    print("=" * 60)
    print("GNSS Error Prediction Evaluation")
    print("=" * 60)

    evaluation = full_evaluation(predictions)
    print_summary_table(evaluation)
    save_evaluation(evaluation)
