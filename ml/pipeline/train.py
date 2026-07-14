"""
Training Pipeline
==================
Full training orchestrator matching the paper's stacking methodology:
  1. Load & preprocess data (with optional differencing)
  2. Feature engineering → X_seq + X_tab per horizon
  3. Expanding-window time-series CV for stacking OOF predictions (paper §III-F)
  4. Train final base models on full training data
  5. Train per-horizon Ridge stackers on OOF predictions
  6. Compute stacker residuals → train GP (paper §III-G)
  7. Save all model checkpoints

v2 changes:
  - Expanding-window cross-validation replaces simple train/val split
    (prevents temporal leakage into stacker)
  - DIFFERENCE_TARGET: models predict Δe instead of raw e(t+h)
  - TEST_NORMALISE: test series is scaled to train distribution before inference
  - Right-sized hyperparameters (see config.py)
"""

import numpy as np
import torch
import json
import os
import sys
import time
import pickle
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HORIZONS, SEQUENCE_LENGTH, CHECKPOINT_DIR, RESULTS_DIR,
    EPOCHS, N_CV_FOLDS, VAL_FRACTION,
    DIFFERENCE_TARGET, TEST_NORMALISE,
)
from data.data_loader import GNSSDataset
from features.feature_engineering import (
    build_features, build_single_window, reconstruct_from_diff,
    train_val_split, expanding_window_splits,
)
from models.lstm_gru import LSTMGRUModel, train_lstm_gru, predict_lstm_gru
from models.transformer import TimeSeriesTransformer, train_transformer, predict_transformer
from models.xgboost_model import train_xgboost, predict_xgboost
from models.ridge_stacker import RidgeStacker
from models.gaussian_process import train_gp, predict_gp, apply_gp_correction


# ---------------------------------------------------------------------------
# Normalisation helpers for TEST_NORMALISE
# ---------------------------------------------------------------------------

def _norm_stats(series: np.ndarray):
    """Return (mean, std) computed on *series* (training data)."""
    return float(series.mean()), float(series.std() + 1e-8)


def _normalise(series: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (series - mean) / std


def _denormalise(series: np.ndarray, mean: float, std: float) -> np.ndarray:
    return series * std + mean


class GNSSEnsemble:
    """
    Full ensemble pipeline for GNSS error prediction.

    Matches the paper's architecture (Figure 1):
      Raw Data → Features → [LSTM-GRU, Transformer, XGBoost]
      → Ridge Stacker → GP Residual → Final Prediction

    v2 additions:
      - Expanding-window CV OOF generation for unbiased stacker training
      - Difference-based target (DIFFERENCE_TARGET)
      - Test-time normalisation (TEST_NORMALISE)
    """

    def __init__(self, satellite_id: str = None):
        self.lstm_models: Dict[int, tuple] = {}
        self.transformer_models: Dict[int, tuple] = {}
        self.xgb_models: Dict[int, Any] = {}
        self.stackers: Dict[int, RidgeStacker] = {}
        self.gp_models: Dict[int, tuple] = {}
        self.training_history: Dict[str, Any] = {}
        self.train_mean: float = 0.0   # saved for TEST_NORMALISE at inference
        self.train_std: float = 1.0
        self.satellite_id = satellite_id
        self.is_trained = False

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        series: np.ndarray,
        epochs: int = EPOCHS,
        quick: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the full ensemble pipeline on a single satellite error series.

        Parameters
        ----------
        series : np.ndarray — raw (unnormalised) training series
        epochs : int
        quick  : bool — 5 epochs if True
        verbose: bool
        """
        if quick:
            epochs = min(5, epochs)

        if verbose:
            print("=" * 60)
            print("Training GNSS Ensemble Pipeline")
            print("=" * 60)
            print(f"  Series length: {len(series)} steps")
            print(f"  Horizons: {HORIZONS} ({[h*15 for h in HORIZONS]} min)")
            print(f"  Epochs: {epochs}  |  CV folds: {N_CV_FOLDS}")
            print(f"  Difference target: {DIFFERENCE_TARGET}")
            print(f"  Test normalise:    {TEST_NORMALISE}")
            print()

        # ── Normalise training series ──
        self.train_mean, self.train_std = _norm_stats(series)
        normed = _normalise(series, self.train_mean, self.train_std)

        results = {}
        start_time = time.time()

        for h in HORIZONS:
            horizon_start = time.time()
            if verbose:
                print(f"--- Horizon h={h} ({h*15} min) ---")

            # ── Step 1: Build features on normalised series with amplitude augmentation ──
            # augment=True applies random scale factors 0.15–7× to each training sample.
            # This teaches the model to predict correctly under distribution shift.
            X_seq, X_tab, y = build_features(normed, horizon=h, augment=True)

            if verbose:
                print(f"  Features: X_seq={X_seq.shape}, X_tab={X_tab.shape}, "
                      f"y={y.shape}  (diff={DIFFERENCE_TARGET}, augment=True)")

            # ── Step 2: Expanding-window OOF for unbiased stacker training ──
            if verbose:
                print(f"  Generating OOF predictions ({N_CV_FOLDS}-fold expanding CV)...")

            oof_lstm  = np.zeros(len(y), dtype=np.float32)
            oof_trans = np.zeros(len(y), dtype=np.float32)
            oof_xgb   = np.zeros(len(y), dtype=np.float32)
            oof_mask  = np.zeros(len(y), dtype=bool)

            splits = list(expanding_window_splits(len(y), n_folds=N_CV_FOLDS))

            for fold_idx, (tr_idx, vl_idx) in enumerate(splits):
                X_seq_tr, X_seq_vl = X_seq[tr_idx], X_seq[vl_idx]
                X_tab_tr, X_tab_vl = X_tab[tr_idx], X_tab[vl_idx]
                y_tr, y_vl         = y[tr_idx],     y[vl_idx]

                if verbose:
                    print(f"    Fold {fold_idx+1}/{len(splits)}: "
                          f"train={len(y_tr)}, val={len(y_vl)}")

                # Fold-level base models (quick=True to keep CV fast)
                fold_epochs = max(20, epochs // 3)
                m_lstm, sy_lstm, _ = train_lstm_gru(
                    X_seq_tr, y_tr, X_seq_vl, y_vl,
                    epochs=fold_epochs, verbose=False,
                )
                m_trans, sy_trans, _ = train_transformer(
                    X_seq_tr, y_tr, X_seq_vl, y_vl,
                    epochs=fold_epochs, verbose=False,
                )
                m_xgb = train_xgboost(X_tab_tr, y_tr, X_tab_vl, y_vl)

                oof_lstm[vl_idx]  = predict_lstm_gru(m_lstm, sy_lstm, X_seq_vl)
                oof_trans[vl_idx] = predict_transformer(m_trans, sy_trans, X_seq_vl)
                oof_xgb[vl_idx]   = predict_xgboost(m_xgb, X_tab_vl)
                oof_mask[vl_idx]  = True

            # Only use samples that appeared in at least one validation fold
            valid = oof_mask
            if valid.sum() < 10:
                # Fallback to simple val split if not enough OOF samples
                if verbose:
                    print("  WARNING: too few OOF samples, falling back to val split")
                (X_seq_tr, X_seq_vl,
                 X_tab_tr, X_tab_vl,
                 y_tr, y_vl) = train_val_split(X_seq, X_tab, y)
                valid = np.zeros(len(y), dtype=bool)
                valid[-len(y_vl):] = True
                # Retrain quick fold models for fallback OOF
                m_lstm, sy_lstm, _ = train_lstm_gru(
                    X_seq_tr, y_tr, X_seq_vl, y_vl, epochs=min(30, epochs), verbose=False)
                m_trans, sy_trans, _ = train_transformer(
                    X_seq_tr, y_tr, X_seq_vl, y_vl, epochs=min(30, epochs), verbose=False)
                m_xgb = train_xgboost(X_tab_tr, y_tr, X_tab_vl, y_vl)
                oof_lstm[-len(y_vl):]  = predict_lstm_gru(m_lstm, sy_lstm, X_seq_vl)
                oof_trans[-len(y_vl):] = predict_transformer(m_trans, sy_trans, X_seq_vl)
                oof_xgb[-len(y_vl):]  = predict_xgboost(m_xgb, X_tab_vl)

            oof_y     = y[valid]
            oof_l     = oof_lstm[valid]
            oof_t     = oof_trans[valid]
            oof_x     = oof_xgb[valid]

            # ── Step 3: Train final base models on full training data ──
            if verbose:
                print(f"  Training final base models on full {len(y)} samples...")

            # Use last 20% as validation for early stopping (not for stacker)
            (X_seq_tr, X_seq_vl,
             X_tab_tr, X_tab_vl,
             y_tr, y_vl) = train_val_split(X_seq, X_tab, y)

            if verbose:
                print("  [1/3] Training LSTM-GRU...")
            lstm_m, lstm_sy, lstm_hist = train_lstm_gru(
                X_seq_tr, y_tr, X_seq_vl, y_vl,
                epochs=epochs, verbose=verbose,
            )
            self.lstm_models[h] = (lstm_m, lstm_sy)

            if verbose:
                print("  [2/3] Training Transformer...")
            trans_m, trans_sy, trans_hist = train_transformer(
                X_seq_tr, y_tr, X_seq_vl, y_vl,
                epochs=epochs, verbose=verbose,
            )
            self.transformer_models[h] = (trans_m, trans_sy)

            if verbose:
                print("  [3/3] Training XGBoost...")
            xgb_m = train_xgboost(X_tab_tr, y_tr, X_tab_vl, y_vl)
            self.xgb_models[h] = xgb_m

            # ── Step 4: Train Ridge stacker on OOF predictions (paper §III-F) ──
            if verbose:
                print(f"  Training Ridge stacker on {valid.sum()} OOF samples...")
            stacker = RidgeStacker().fit(oof_l, oof_t, oof_x, oof_y, h)
            self.stackers[h] = stacker
            if verbose:
                print(f"    {stacker}")

            # ── Step 5: Residual GP on OOF stacker residuals (paper §III-G) ──
            stacker_oof_pred = stacker.predict(oof_l, oof_t, oof_x)
            residuals = oof_y - stacker_oof_pred
            time_idx = np.arange(len(residuals), dtype=np.float32) / len(residuals)

            if verbose:
                print("  Training Gaussian Process on OOF residuals...")
            gp, lik = train_gp(time_idx, residuals.astype(np.float32))
            self.gp_models[h] = (gp, lik)

            # ── Step 6: Validation-set evaluation ──
            p_lstm_vl  = predict_lstm_gru(lstm_m, lstm_sy, X_seq_vl)
            p_trans_vl = predict_transformer(trans_m, trans_sy, X_seq_vl)
            p_xgb_vl   = predict_xgboost(xgb_m, X_tab_vl)
            stk_vl     = stacker.predict(p_lstm_vl, p_trans_vl, p_xgb_vl)

            gp_t_vl = np.linspace(0, 1, len(stk_vl), dtype=np.float32)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_mean_vl, gp_std_vl = predict_gp(gp, lik, gp_t_vl)
            final_vl = apply_gp_correction(stk_vl, gp_mean_vl)
            final_res = y_vl - final_vl

            from scipy.stats import shapiro, normaltest
            rmse = float(np.sqrt(np.mean(final_res ** 2)))
            mae  = float(np.mean(np.abs(final_res)))
            try:
                sw_stat, sw_pval = shapiro(final_res[:min(5000, len(final_res))])
            except Exception:
                sw_stat, sw_pval = 0.0, 0.0
            try:
                da_stat, da_pval = normaltest(final_res)
            except Exception:
                da_stat, da_pval = 0.0, 0.0

            horizon_time = time.time() - horizon_start
            results[h] = {
                "horizon": h,
                "horizon_min": h * 15,
                "rmse": rmse,
                "mae": mae,
                "shapiro_wilk_stat": float(sw_stat),
                "shapiro_wilk_pval": float(sw_pval),
                "dagostino_stat": float(da_stat),
                "dagostino_pval": float(da_pval),
                "is_normal_sw": sw_pval > 0.05,
                "is_normal_da": da_pval > 0.05,
                "residual_mean": float(final_res.mean()),
                "residual_std": float(final_res.std()),
                "stacker_weights": stacker.get_weights_info()["weights"],
                "training_time_s": horizon_time,
                "n_oof_samples": int(valid.sum()),
            }

            if verbose:
                normal_tag = "[OK] NORMAL" if sw_pval > 0.05 else "[!!] non-normal"
                print(f"  Val RMSE={rmse:.5f} | MAE={mae:.5f} (normalised units)")
                print(f"  Shapiro-Wilk p={sw_pval:.4f} [{normal_tag}]")
                print(f"  Time: {horizon_time:.1f}s")
                print()

        self.is_trained = True
        self.training_history = results

        total_time = time.time() - start_time
        if verbose:
            print("=" * 60)
            print(f"Training complete in {total_time:.1f}s")
            print("=" * 60)

        return results

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        series: np.ndarray,
        horizon: int = None,
        test_series: np.ndarray = None,
        use_test_context: bool = True,
    ) -> Dict[int, dict]:
        """
        Generate predictions using the trained ensemble.

        Parameters
        ----------
        series           : np.ndarray — training series (raw, unnormalised)
        horizon          : int or None — specific horizon or None for all
        test_series      : np.ndarray or None — uploaded test data
        use_test_context : bool — if True (default), prepend the first
                           SEQUENCE_LENGTH steps of test_series as the
                           inference context window, so predictions are
                           anchored in the test regime rather than the
                           training tail.  The remaining test steps after
                           the context window become ground truth.

        Returns
        -------
        predictions : dict — horizon → {predictions, uncertainties, ...}
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        horizons_to_run = [horizon] if horizon is not None else HORIZONS

        # ── Normalise training series using stored train stats ──
        normed_train = _normalise(series, self.train_mean, self.train_std)

        # ── Build the inference context window ──
        # When we have test data and use_test_context=True:
        #   context  = train[-OVERLAP:] + test[:SEQUENCE_LENGTH]
        #   gt       = test[SEQUENCE_LENGTH:]  (the portion we actually predict)
        # This grounds the model in the test-day signal before asking it to forecast.
        #
        # When use_test_context=False (legacy):
        #   context = entire training series (original behaviour)
        #   gt      = test[:]  aligned against the last N training predictions

        has_test = test_series is not None and len(test_series) > 0

        norm_mean = self.train_mean
        norm_std  = self.train_std

        if has_test and use_test_context and len(test_series) > SEQUENCE_LENGTH:
            max_h = max(HORIZONS)

            # How many test steps to use as warm-up context (≥ seq_len, ≤ half test)
            ctx_n_test        = min(len(test_series) // 2, 48, len(test_series) - max_h - 2)
            ctx_n_test        = max(ctx_n_test, SEQUENCE_LENGTH)
            ctx_train_overlap = SEQUENCE_LENGTH + max_h + 4

            test_context = test_series[:ctx_n_test]
            ground_truth = test_series[ctx_n_test:]

            # Stitch train tail + test context, normalised with training stats.
            stitched_raw    = np.concatenate([series[-ctx_train_overlap:], test_context])
            normed_stitched = _normalise(stitched_raw, self.train_mean, self.train_std)

            inference_series = normed_stitched
            gt_raw           = ground_truth
            mode             = "test_context"

        elif has_test and not use_test_context:
            # Legacy: inference over training series, align against test
            if TEST_NORMALISE and len(test_series) > 1:
                t_mean = float(test_series.mean())
                t_std  = float(test_series.std() + 1e-8)
                normed_test  = _normalise(test_series, t_mean, t_std)
                normed_test  = normed_test * (t_std / self.train_std) + \
                               (t_mean - self.train_mean) / self.train_std
            else:
                normed_test = _normalise(test_series, self.train_mean, self.train_std)
            inference_series = normed_train
            gt_raw           = test_series
            mode             = "legacy"

        else:
            inference_series = normed_train
            gt_raw           = None
            mode             = "no_test"

        results = {}

        for h in horizons_to_run:
            # ── Run each base model over the inference series ──
            X_seq, X_tab, _ = build_features(inference_series, horizon=h)

            p_lstm  = predict_lstm_gru(*self.lstm_models[h], X_seq)
            p_trans = predict_transformer(*self.transformer_models[h], X_seq)
            p_xgb   = predict_xgboost(self.xgb_models[h], X_tab)

            stacker_pred = self.stackers[h].predict(p_lstm, p_trans, p_xgb)

            import warnings
            gp, lik = self.gp_models[h]
            time_idx = np.arange(len(stacker_pred), dtype=np.float32) / max(len(stacker_pred) - 1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_mean, gp_std = predict_gp(gp, lik, time_idx)
            final_pred_normed = apply_gp_correction(stacker_pred, gp_mean)

            # ── Reconstruct from differences (if DIFFERENCE_TARGET) ──
            if DIFFERENCE_TARGET:
                final_pred_normed = reconstruct_from_diff(
                    final_pred_normed, inference_series, h
                )

            # ── Denormalize with training stats ──
            final_pred = _denormalise(final_pred_normed, norm_mean, norm_std)

            # ── Alignment ──
            if mode in ("test_context", "legacy"):
                n = min(len(final_pred), len(gt_raw))
                preds_aln = final_pred[-n:]
                truth_aln = gt_raw[:n]
            else:
                preds_aln = final_pred
                truth_aln = None

            # ── Build result dict ──
            results[h] = {
                "predictions":   preds_aln.tolist() if preds_aln is not None else final_pred.tolist(),
                "uncertainties": (gp_std[-len(preds_aln):] * norm_std).tolist()
                                  if preds_aln is not None else (gp_std * norm_std).tolist(),
                "base_predictions": {
                    "lstm_gru":    _denormalise(
                        reconstruct_from_diff(p_lstm, inference_series, h) if DIFFERENCE_TARGET else p_lstm,
                        norm_mean, norm_std).tolist(),
                    "transformer": _denormalise(
                        reconstruct_from_diff(p_trans, inference_series, h) if DIFFERENCE_TARGET else p_trans,
                        norm_mean, norm_std).tolist(),
                    "xgboost":     _denormalise(
                        reconstruct_from_diff(p_xgb, inference_series, h) if DIFFERENCE_TARGET else p_xgb,
                        norm_mean, norm_std).tolist(),
                },
                "stacker_predictions": _denormalise(
                    reconstruct_from_diff(stacker_pred, inference_series, h) if DIFFERENCE_TARGET else stacker_pred,
                    norm_mean, norm_std).tolist(),
                "horizon":        h,
                "horizon_min":    h * 15,
                "n_predictions":  len(preds_aln) if preds_aln is not None else len(final_pred),
                "inference_mode": mode,
            }

            # ── Metrics vs ground truth ──
            if truth_aln is not None and len(truth_aln) > 0:
                residuals = truth_aln - preds_aln
                results[h]["ground_truth"] = truth_aln.tolist()
                results[h]["residuals"]    = residuals.tolist()
                results[h]["rmse"]         = float(np.sqrt(np.mean(residuals ** 2)))
                results[h]["mae"]          = float(np.mean(np.abs(residuals)))
                results[h]["n_test"]       = len(truth_aln)

                # Persistence baseline: last observed test-context value
                last_known   = test_series[ctx_n_test - 1] if mode == "test_context" else series[-1]
                persist_pred = np.full(len(truth_aln), last_known)
                mean_pred    = np.full(len(truth_aln), series.mean())
                results[h]["baseline_persist_rmse"] = float(
                    np.sqrt(np.mean((truth_aln - persist_pred) ** 2)))
                results[h]["baseline_mean_rmse"] = float(
                    np.sqrt(np.mean((truth_aln - mean_pred) ** 2)))

        return results

    # ------------------------------------------------------------------
    # fine_tune() — transductive adaptation for short horizons
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        train_series: np.ndarray,
        adapt_series: np.ndarray,
        short_horizons: list = None,
        ft_epochs: int = 50,
        verbose: bool = True,
    ) -> None:
        """
        Fine-tune SHORT-horizon models using the first part of test data.

        The training series and the adaptation slice (first half of the
        test day) are concatenated.  Models are updated from their current
        weights using a low learning rate — they do NOT retrain from scratch.
        Long-horizon models are left completely unchanged.

        This is the *split-test adaptation* technique:
          - adapt_series  (first 50% of test) → update model weights
          - eval_series   (second 50% of test) → evaluate with fine-tuned model

        Parameters
        ----------
        train_series  : np.ndarray — original training data (raw units)
        adapt_series  : np.ndarray — first half of test data (raw units)
        short_horizons: list or None — horizons to fine-tune (default: [1,2,4])
        ft_epochs     : int — max fine-tuning epochs (early stopping applies)
        verbose       : bool
        """
        if short_horizons is None:
            short_horizons = [1, 2, 4]   # 15, 30, 60 min

        if verbose:
            print("=" * 60)
            print("Fine-tuning short-horizon models (split-test adaptation)")
            print(f"  Adapt steps: {len(adapt_series)}")
            print(f"  Horizons:    {short_horizons} ({[h*15 for h in short_horizons]} min)")
            print(f"  Epochs:      {ft_epochs} (with early stopping)")
            print(f"  LR:          5e-5 (continuation from base weights)")
            print("="*60)

        # Build combined series and normalise with EXISTING train stats
        combined_raw   = np.concatenate([train_series, adapt_series])
        combined_normed = _normalise(combined_raw, self.train_mean, self.train_std)

        for h in short_horizons:
            if h not in self.lstm_models:
                if verbose:
                    print(f"  [SKIP] h={h} not found in model store")
                continue

            if verbose:
                print(f"\n--- Fine-tuning h={h} ({h*15}min) ---")

            # Build features from combined series (augment=True for regularisation)
            X_seq, X_tab, y = build_features(
                combined_normed, horizon=h, augment=True
            )
            (X_seq_tr, X_seq_vl,
             X_tab_tr, X_tab_vl,
             y_tr, y_vl) = train_val_split(X_seq, X_tab, y)

            if verbose:
                print(f"  Combined features: X_seq={X_seq.shape}, X_tab={X_tab.shape}")

            # ── Fine-tune LSTM-GRU from existing weights ──
            if verbose:
                print("  [1/3] Fine-tuning LSTM-GRU...")
            lstm_m, lstm_sy = self.lstm_models[h]
            lstm_m, lstm_sy, _ = train_lstm_gru(
                X_seq_tr, y_tr, X_seq_vl, y_vl,
                epochs=ft_epochs,
                verbose=verbose,
                pretrained_model=lstm_m,
                pretrained_scaler_y=lstm_sy,
            )
            self.lstm_models[h] = (lstm_m, lstm_sy)

            # ── Fine-tune Transformer from existing weights ──
            if verbose:
                print("  [2/3] Fine-tuning Transformer...")
            trans_m, trans_sy = self.transformer_models[h]
            trans_m, trans_sy, _ = train_transformer(
                X_seq_tr, y_tr, X_seq_vl, y_vl,
                epochs=ft_epochs,
                verbose=verbose,
                pretrained_model=trans_m,
                pretrained_scaler_y=trans_sy,
            )
            self.transformer_models[h] = (trans_m, trans_sy)

            # ── Retrain XGBoost on combined data (no warm-starting needed) ──
            if verbose:
                print("  [3/3] Retraining XGBoost on combined data...")
            xgb_m = train_xgboost(X_tab_tr, y_tr, X_tab_vl, y_vl)
            self.xgb_models[h] = xgb_m

        if verbose:
            print("\nFine-tuning complete.")
            print("Long-horizon models (h=8,16) unchanged.")

    # ------------------------------------------------------------------
    # predict_rolling()
    # ------------------------------------------------------------------

    def predict_rolling(
        self,
        series: np.ndarray,
        test_series: np.ndarray,
        horizon: int = None,
        context_prefix: np.ndarray = None,
    ) -> Dict[int, dict]:
        """
        Rolling multi-step-ahead inference.

        At each test step t the model is given the last SEQUENCE_LENGTH
        *observed* values (from train tail + test[0:t]) and asked to
        predict test[t + h - 1].  This is the most honest evaluation:

          - The context is always up-to-date with observed values
          - No regime mismatch: the model sees actual test-day signal
          - Mirrors real production usage (new readings arrive every 15min)

        For h=1 this gives true one-step-ahead predictions.
        For h=k this gives true k-step-ahead predictions.

        Parameters
        ----------
        context_prefix : np.ndarray or None
            Extra observed values (e.g., adaptation slice) prepended to the
            rolling buffer before predicting on test_series.  The last
            SEQUENCE_LENGTH values of (series tail + context_prefix) are
            used as the initial buffer, so the model starts with full
            test-day context rather than only training-tail context.

        Parameters
        ----------
        series      : np.ndarray — training series (raw, unnormalised)
        test_series : np.ndarray — test observations (raw, unnormalised)
        horizon     : int or None — specific horizon or all HORIZONS

        Returns
        -------
        dict — same schema as predict() keyed by horizon integer
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        import warnings

        horizons_to_run = [horizon] if horizon is not None else HORIZONS

        # Build the sliding observation buffer: start with training tail
        obs_buffer = list(series[-SEQUENCE_LENGTH:])
        n_test = len(test_series)

        results = {}

        for h in horizons_to_run:
            preds_raw    = []   # model predictions (original units)
            truth_raw    = []   # corresponding ground truth values

            # Reset buffer for each horizon.
            # If context_prefix is provided, seed the buffer with the last
            # SEQUENCE_LENGTH values of (series tail + context_prefix) so the
            # rolling model sees the volatile test-regime from the first step.
            if context_prefix is not None:
                combined_ctx = np.concatenate([series[-SEQUENCE_LENGTH:], context_prefix])
                buf = list(combined_ctx[-SEQUENCE_LENGTH:])
            else:
                buf = list(series[-SEQUENCE_LENGTH:])

            for t in range(n_test):
                # The target we want to predict is test_series[t + h - 1]
                target_idx = t + h - 1
                if target_idx >= n_test:
                    break    # not enough test steps left for this horizon

                # ── Build normalised window ──
                window_raw   = np.array(buf[-SEQUENCE_LENGTH:], dtype=np.float32)
                window_normed = _normalise(window_raw, self.train_mean, self.train_std)

                # ── Per-window instance normalisation (amplitude rescaling) ──
                # The test-day window std can be 4–8× the training std.
                # Rescaling to unit std puts the input in the model's familiar
                # operating range, then we multiply predictions back.
                # The true amplitude (rescale_factor) is passed to XGBoost via
                # amplitude_override so it can calibrate by regime.
                w_std = float(window_normed.std() + 1e-8)
                # Only rescale when significantly outside expected range (±1 std)
                rescale_threshold = 1.5
                if w_std > rescale_threshold:
                    window_for_model = (window_normed / w_std).astype(np.float32)
                    amplitude_factor = w_std
                else:
                    window_for_model = window_normed
                    amplitude_factor = None   # no override needed, already in range

                # Time offset: training length + current step
                time_off = len(series) + t

                x_seq, x_tab, anchor_model = build_single_window(
                    window_for_model, h,
                    time_offset=time_off,
                    amplitude_override=amplitude_factor,
                )

                # ── Run ensemble ──
                p_lstm  = predict_lstm_gru(*self.lstm_models[h], x_seq)
                p_trans = predict_transformer(*self.transformer_models[h], x_seq)
                p_xgb   = predict_xgboost(self.xgb_models[h], x_tab)
                stk     = self.stackers[h].predict(p_lstm, p_trans, p_xgb)

                # GP correction (use a fixed time index of 0.5 for single steps)
                gp, lik = self.gp_models[h]
                gp_t = np.array([0.5], dtype=np.float32)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp_mean, _ = predict_gp(gp, lik, gp_t)
                final_normed = apply_gp_correction(stk, gp_mean)

                # ── Reconstruct from difference ──
                # anchor_model is in model space (rescaled if amplitude_factor set).
                # After reconstruction multiply by amplitude_factor to undo scaling.
                af = amplitude_factor if amplitude_factor is not None else 1.0
                if DIFFERENCE_TARGET:
                    # pred in model-space, then scale back to normed space
                    pred_normed = (anchor_model + float(final_normed[0])) * af
                else:
                    pred_normed = float(final_normed[0]) * af

                # Denormalise to original units
                pred_raw = _denormalise(
                    np.array([pred_normed]), self.train_mean, self.train_std
                )[0]

                preds_raw.append(float(pred_raw))
                truth_raw.append(float(test_series[target_idx]))

                # Advance buffer: append the *observed* test value at step t
                buf.append(float(test_series[t]))

            preds = np.array(preds_raw, dtype=np.float64)
            truth = np.array(truth_raw,  dtype=np.float64)
            n     = len(preds)

            if n == 0:
                results[h] = {
                    "predictions": [], "ground_truth": [], "uncertainties": [],
                    "rmse": float('nan'), "mae": float('nan'),
                    "n_test": 0, "inference_mode": "rolling",
                    "horizon": h, "horizon_min": h * 15, "n_predictions": 0,
                }
                continue

            residuals = truth - preds
            rmse      = float(np.sqrt(np.mean(residuals ** 2)))
            mae       = float(np.mean(np.abs(residuals)))

            # ── Naive persistence baseline ──
            # For target test_series[t + h - 1], persistence = last observed
            # value BEFORE the horizon window, i.e. test_series[t - 1],
            # or series[-1] (last training value) when t == 0.
            persist_preds = []
            for t in range(min(n_test - h + 1, n)):
                if t == 0:
                    persist_preds.append(float(series[-1]))
                else:
                    persist_preds.append(float(test_series[t - 1]))
            persist_preds = np.array(persist_preds, dtype=np.float64)
            p_truth      = truth[:len(persist_preds)]
            persist_rmse = float(np.sqrt(np.mean((p_truth - persist_preds) ** 2)))
            mean_rmse    = float(np.sqrt(np.mean((truth - float(series.mean())) ** 2)))

            results[h] = {
                "predictions":          preds.tolist(),
                "ground_truth":         truth.tolist(),
                "residuals":            residuals.tolist(),
                "uncertainties":        [],   # GP uncertainty omitted for speed
                "rmse":                 rmse,
                "mae":                  mae,
                "n_test":               n,
                "baseline_persist_rmse": persist_rmse,
                "baseline_mean_rmse":   mean_rmse,
                "horizon":              h,
                "horizon_min":          h * 15,
                "n_predictions":        n,
                "inference_mode":       "rolling",
                # base/stacker predictions not stored (too large for rolling)
                "base_predictions":     {},
                "stacker_predictions":  [],
            }

        return results

    # ------------------------------------------------------------------

    def save(self, path: str = None):
        """Save all model checkpoints."""
        if path is None:
            path = CHECKPOINT_DIR
        if self.satellite_id:
            path = os.path.join(path, self.satellite_id)
        os.makedirs(path, exist_ok=True)

        for h in HORIZONS:
            if h in self.lstm_models:
                lstm_m, lstm_sy = self.lstm_models[h]
                torch.save(lstm_m.state_dict(), os.path.join(path, f"lstm_gru_h{h}.pt"))
                with open(os.path.join(path, f"lstm_scaler_h{h}.pkl"), "wb") as f:
                    pickle.dump(lstm_sy, f)

            if h in self.transformer_models:
                trans_m, trans_sy = self.transformer_models[h]
                torch.save(trans_m.state_dict(), os.path.join(path, f"transformer_h{h}.pt"))
                with open(os.path.join(path, f"trans_scaler_h{h}.pkl"), "wb") as f:
                    pickle.dump(trans_sy, f)

            if h in self.xgb_models:
                self.xgb_models[h].save_model(os.path.join(path, f"xgboost_h{h}.json"))

            if h in self.stackers:
                with open(os.path.join(path, f"stacker_h{h}.pkl"), "wb") as f:
                    pickle.dump(self.stackers[h], f)

            if h in self.gp_models:
                gp, lik = self.gp_models[h]
                gp_state = {
                    "model_state":      gp.state_dict(),
                    "likelihood_state": lik.state_dict(),
                    "train_x":          gp.train_inputs[0].numpy(),
                    "train_y":          gp.train_targets.numpy(),
                }
                with open(os.path.join(path, f"gp_h{h}.pkl"), "wb") as f:
                    pickle.dump(gp_state, f)

        # Save training history + normalisation stats
        meta = {
            "training_history": self.training_history,
            "train_mean": self.train_mean,
            "train_std":  self.train_std,
            "difference_target": DIFFERENCE_TARGET,
            "test_normalise":    TEST_NORMALISE,
        }
        with open(os.path.join(path, "training_history.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"Models saved to {path}")

    def load(self, path: str = None):
        """Load model checkpoints."""
        if path is None:
            path = CHECKPOINT_DIR
        if self.satellite_id:
            path = os.path.join(path, self.satellite_id)

        from config import DEVICE

        # Load normalisation stats
        history_path = os.path.join(path, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                meta = json.load(f)
            # Support both old format (flat dict) and new format (with meta keys)
            if "train_mean" in meta:
                self.train_mean = meta["train_mean"]
                self.train_std  = meta["train_std"]
                self.training_history = meta.get("training_history", {})
            else:
                self.training_history = meta
                self.train_mean = 0.0
                self.train_std  = 1.0

        for h in HORIZONS:
            # LSTM-GRU
            lstm_path = os.path.join(path, f"lstm_gru_h{h}.pt")
            if os.path.exists(lstm_path):
                model = LSTMGRUModel().to(DEVICE)
                model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
                model.eval()
                with open(os.path.join(path, f"lstm_scaler_h{h}.pkl"), "rb") as f:
                    scaler = pickle.load(f)
                self.lstm_models[h] = (model, scaler)

            # Transformer
            trans_path = os.path.join(path, f"transformer_h{h}.pt")
            if os.path.exists(trans_path):
                model = TimeSeriesTransformer().to(DEVICE)
                model.load_state_dict(torch.load(trans_path, map_location=DEVICE))
                model.eval()
                with open(os.path.join(path, f"trans_scaler_h{h}.pkl"), "rb") as f:
                    scaler = pickle.load(f)
                self.transformer_models[h] = (model, scaler)

            # XGBoost
            xgb_path = os.path.join(path, f"xgboost_h{h}.json")
            if os.path.exists(xgb_path):
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(xgb_path)
                self.xgb_models[h] = model

            # Stacker
            stacker_path = os.path.join(path, f"stacker_h{h}.pkl")
            if os.path.exists(stacker_path):
                with open(stacker_path, "rb") as f:
                    self.stackers[h] = pickle.load(f)

            # GP model
            gp_path = os.path.join(path, f"gp_h{h}.pkl")
            if os.path.exists(gp_path):
                import gpytorch
                from models.gaussian_process import ExactGPModel
                with open(gp_path, "rb") as f:
                    gp_state = pickle.load(f)
                train_x = torch.tensor(gp_state["train_x"], dtype=torch.float32)
                train_y = torch.tensor(gp_state["train_y"], dtype=torch.float32)
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                gp_model = ExactGPModel(train_x, train_y, likelihood)
                gp_model.load_state_dict(gp_state["model_state"])
                likelihood.load_state_dict(gp_state["likelihood_state"])
                gp_model.eval()
                likelihood.eval()
                self.gp_models[h] = (gp_model, likelihood)

        self.is_trained = True
        print(f"Models loaded from {path}")


def train_all_satellites(
    dataset: GNSSDataset,
    error_col: str = None,
    epochs: int = EPOCHS,
    quick: bool = False,
    verbose: bool = True,
) -> Dict[str, "GNSSEnsemble"]:
    """
    Train ensembles for all satellites in the dataset.

    Returns
    -------
    ensembles : dict — satellite_id → GNSSEnsemble
    """
    if error_col is None:
        error_col = dataset.get_default_error_col()

    ensembles = {}

    for sat_id in dataset.satellite_ids:
        if verbose:
            sat_type = dataset.get_satellite_type(sat_id)
            print(f"\n{'=' * 60}")
            print(f"  Satellite: {sat_id} ({sat_type})")
            print(f"  Error column: {error_col}")
            print(f"{'=' * 60}")

        train_series, test_series, scaler = dataset.get_satellite_data(
            sat_id, error_col, normalize=False
        )

        ensemble = GNSSEnsemble(satellite_id=sat_id)
        ensemble.fit(train_series, epochs=epochs, quick=quick, verbose=verbose)
        ensembles[sat_id] = ensemble

    return ensembles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GNSS Ensemble Pipeline")
    parser.add_argument("--error-col", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--satellite", type=str, default=None)
    args = parser.parse_args()

    dataset = GNSSDataset()
    error_col = args.error_col or dataset.get_default_error_col()
    print(f"Dataset: {dataset.summary()['n_satellites']} satellites | col={error_col}")

    if args.satellite:
        train_series, _, _ = dataset.get_satellite_data(
            args.satellite, error_col, normalize=False
        )
        ensemble = GNSSEnsemble(satellite_id=args.satellite)
        results = ensemble.fit(train_series, epochs=args.epochs, quick=args.quick)
        ensemble.save()
        with open(os.path.join(RESULTS_DIR, f"train_results_{args.satellite}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        ensembles = train_all_satellites(dataset, error_col, args.epochs, args.quick)
        # Save all trained ensembles to their respective folders
        for sat_id, ensemble in ensembles.items():
            ensemble.save()
