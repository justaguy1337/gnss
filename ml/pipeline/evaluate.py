"""
Evaluation Module
==================
Comprehensive evaluation matching competition criteria:
  - RMSE, MAE, R² per horizon
  - Normality tests: Shapiro-Wilk, Anderson-Darling, K-S test
  - Q-Q plots and histogram generation
  - Skewness and kurtosis measurements
"""

import numpy as np
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HORIZONS, RESULTS_DIR


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = None
) -> dict:
    """
    Full evaluation of prediction quality.

    Parameters
    ----------
    y_true : ground truth values
    y_pred : predicted values
    horizon : prediction horizon (for labeling)

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

    return {
        "horizon": horizon,
        "horizon_min": horizon * 15 if horizon else None,
        "n_samples": n,
        # Prediction quality
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
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
    verbose: bool = True
) -> dict:
    """
    Run full evaluation across all horizons.

    Parameters
    ----------
    predictions : dict — output from predict_day8()

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
        metrics = evaluate_predictions(y_true, y_pred, horizon=h)
        residuals = y_true - y_pred

        # Visualization data
        metrics["qq_data"] = generate_qq_data(residuals)
        metrics["histogram_data"] = generate_histogram_data(residuals)

        evaluation[h] = metrics

        if verbose:
            sw = metrics["shapiro_wilk"]
            tag = "[OK] NORMAL" if sw["is_normal"] else "[!!] non-normal"
            print(f"  h={h:>2} ({h*15:>4}min) | "
                  f"RMSE={metrics['rmse']:.5f} | "
                  f"MAE={metrics['mae']:.5f} | "
                  f"R²={metrics['r2_score']:.4f} | "
                  f"Shapiro p={sw['p_value']:.4f} [{tag}]")

    return evaluation


def save_evaluation(evaluation: dict, filename: str = "evaluation_results.json"):
    """Save evaluation results to JSON."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(evaluation, f, indent=2, default=str)
    print(f"Evaluation saved to {path}")
    return path


def print_summary_table(evaluation: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print(f"{'Horizon':>10} {'RMSE':>10} {'MAE':>10} {'R²':>8} "
          f"{'Skew':>8} {'Kurt':>8} {'S-W p':>10} {'Normal':>8}")
    print("-" * 80)

    for h in sorted(evaluation.keys()):
        m = evaluation[h]
        sw = m["shapiro_wilk"]
        tag = "  [OK]" if sw["is_normal"] else "  [!!]"
        print(f"{m['horizon_min']:>7}min "
              f"{m['rmse']:>10.5f} "
              f"{m['mae']:>10.5f} "
              f"{m['r2_score']:>8.4f} "
              f"{m['residual_skewness']:>8.3f} "
              f"{m['residual_kurtosis']:>8.3f} "
              f"{sw['p_value']:>10.4f} "
              f"{tag}")

    print("=" * 80)


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
