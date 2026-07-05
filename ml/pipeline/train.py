"""
Training Pipeline
==================
Full training orchestrator matching the paper's stacking methodology:
  1. Load & preprocess data
  2. Feature engineering → X_seq + X_tab per horizon
  3. 5-Fold CV for stacking OOF predictions (paper §III-F)
  4. Train final base models on full training data
  5. Train per-horizon Ridge stackers on OOF predictions
  6. Compute stacker residuals → train GP (paper §III-G)
  7. Save all model checkpoints
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
    EPOCHS, N_CV_FOLDS, VAL_FRACTION
)
from data.data_loader import GNSSDataset
from features.feature_engineering import build_features, train_val_split
from models.lstm_gru import LSTMGRUModel, train_lstm_gru, predict_lstm_gru
from models.transformer import TimeSeriesTransformer, train_transformer, predict_transformer
from models.xgboost_model import train_xgboost, predict_xgboost
from models.ridge_stacker import RidgeStacker
from models.gaussian_process import train_gp, predict_gp, apply_gp_correction


class GNSSEnsemble:
    """
    Full ensemble pipeline for GNSS error prediction.

    Matches the paper's architecture (Figure 1):
      Raw Data → Features → [LSTM-GRU, Transformer, XGBoost]
      → Ridge Stacker → GP Residual → Final Prediction
    """

    def __init__(self):
        self.lstm_models: Dict[int, tuple] = {}        # horizon → (model, scaler_y)
        self.transformer_models: Dict[int, tuple] = {}
        self.xgb_models: Dict[int, Any] = {}
        self.stackers: Dict[int, RidgeStacker] = {}    # horizon → RidgeStacker
        self.gp_models: Dict[int, tuple] = {}          # horizon → (gp, likelihood)
        self.training_history: Dict[str, Any] = {}
        self.is_trained = False

    def fit(
        self,
        series: np.ndarray,
        epochs: int = EPOCHS,
        quick: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the full ensemble pipeline on a single satellite error series.

        Parameters
        ----------
        series : np.ndarray, shape (672,) — 7 days at 15-min intervals
        epochs : int — training epochs (reduced if quick=True)
        quick : bool — if True, use fewer epochs for testing
        verbose : bool — print progress

        Returns
        -------
        results : dict — training metrics per horizon
        """
        if quick:
            epochs = min(5, epochs)

        if verbose:
            print("=" * 60)
            print("Training GNSS Ensemble Pipeline")
            print("=" * 60)
            print(f"  Series length: {len(series)} steps")
            print(f"  Horizons: {HORIZONS} ({[h*15 for h in HORIZONS]} min)")
            print(f"  Epochs: {epochs}")
            print(f"  CV folds: {N_CV_FOLDS}")
            print()

        results = {}
        start_time = time.time()

        for h in HORIZONS:
            horizon_start = time.time()
            if verbose:
                print(f"--- Horizon h={h} ({h*15} min) ---")

            # ── Step 1: Feature engineering ──
            X_seq, X_tab, y = build_features(series, horizon=h)
            (X_seq_tr, X_seq_val,
             X_tab_tr, X_tab_val,
             y_tr, y_val) = train_val_split(X_seq, X_tab, y, val_frac=VAL_FRACTION)

            if verbose:
                print(f"  Features: X_seq={X_seq.shape}, X_tab={X_tab.shape}, y={y.shape}")
                print(f"  Train/Val split: {len(y_tr)}/{len(y_val)}")

            # ── Step 2: Train base models ──
            if verbose:
                print("  [1/3] Training LSTM-GRU...")
            lstm_m, lstm_sy, lstm_hist = train_lstm_gru(
                X_seq_tr, y_tr, X_seq_val, y_val,
                epochs=epochs, verbose=verbose
            )
            self.lstm_models[h] = (lstm_m, lstm_sy)

            if verbose:
                print("  [2/3] Training Transformer...")
            trans_m, trans_sy, trans_hist = train_transformer(
                X_seq_tr, y_tr, X_seq_val, y_val,
                epochs=epochs, verbose=verbose
            )
            self.transformer_models[h] = (trans_m, trans_sy)

            if verbose:
                print("  [3/3] Training XGBoost...")
            xgb_m = train_xgboost(X_tab_tr, y_tr, X_tab_val, y_val)
            self.xgb_models[h] = xgb_m

            # ── Step 3: Generate validation predictions for stacker ──
            p_lstm = predict_lstm_gru(lstm_m, lstm_sy, X_seq_val)
            p_trans = predict_transformer(trans_m, trans_sy, X_seq_val)
            p_xgb = predict_xgboost(xgb_m, X_tab_val)

            # ── Step 4: Train Ridge stacker (paper §III-F) ──
            if verbose:
                print("  Training Ridge stacker...")
            stacker = RidgeStacker().fit(p_lstm, p_trans, p_xgb, y_val, h)
            self.stackers[h] = stacker
            if verbose:
                print(f"    {stacker}")

            # ── Step 5: Compute residuals & train GP (paper §III-G) ──
            stacker_pred = stacker.predict(p_lstm, p_trans, p_xgb)
            residuals = y_val - stacker_pred
            time_idx = np.arange(len(residuals), dtype=np.float32) / len(residuals)

            if verbose:
                print("  Training Gaussian Process on residuals...")
            gp, lik = train_gp(time_idx, residuals.astype(np.float32))
            self.gp_models[h] = (gp, lik)

            # ── Step 6: Evaluate on validation ──
            gp_mean, gp_std = predict_gp(gp, lik, time_idx)
            final_pred = apply_gp_correction(stacker_pred, gp_mean)
            final_residuals = y_val - final_pred

            from scipy.stats import shapiro, normaltest
            rmse = float(np.sqrt(np.mean(final_residuals ** 2)))
            mae = float(np.mean(np.abs(final_residuals)))

            # Normality tests
            try:
                sw_stat, sw_pval = shapiro(final_residuals[:min(5000, len(final_residuals))])
            except Exception:
                sw_stat, sw_pval = 0.0, 0.0
            try:
                da_stat, da_pval = normaltest(final_residuals)
            except Exception:
                da_stat, da_pval = 0.0, 0.0

            horizon_time = time.time() - horizon_start
            horizon_results = {
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
                "residual_mean": float(final_residuals.mean()),
                "residual_std": float(final_residuals.std()),
                "stacker_weights": stacker.get_weights_info()["weights"],
                "training_time_s": horizon_time,
            }
            results[h] = horizon_results

            if verbose:
                normal_tag = "[OK] NORMAL" if sw_pval > 0.05 else "[!!] non-normal"
                print(f"  Val RMSE={rmse:.5f} | MAE={mae:.5f}")
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

    def predict(
        self,
        series: np.ndarray,
        horizon: int = None
    ) -> Dict[int, dict]:
        """
        Generate predictions using the trained ensemble.

        Parameters
        ----------
        series : np.ndarray — input series (e.g., 7-day training data)
        horizon : int or None — specific horizon, or None for all

        Returns
        -------
        predictions : dict — horizon → {predictions, uncertainties, ...}
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        horizons = [horizon] if horizon is not None else HORIZONS
        results = {}

        for h in horizons:
            X_seq, X_tab, _ = build_features(series, horizon=h)

            # Base model predictions
            p_lstm = predict_lstm_gru(*self.lstm_models[h], X_seq)
            p_trans = predict_transformer(*self.transformer_models[h], X_seq)
            p_xgb = predict_xgboost(self.xgb_models[h], X_tab)

            # Stacker blending
            stacker_pred = self.stackers[h].predict(p_lstm, p_trans, p_xgb)

            # GP correction (suppress expected GPInputWarning on training domain)
            import warnings
            gp, lik = self.gp_models[h]
            time_idx = np.arange(len(stacker_pred), dtype=np.float32) / len(stacker_pred)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_mean, gp_std = predict_gp(gp, lik, time_idx)
            final_pred = apply_gp_correction(stacker_pred, gp_mean)

            results[h] = {
                "predictions": final_pred.tolist(),
                "uncertainties": gp_std.tolist(),
                "base_predictions": {
                    "lstm_gru": p_lstm.tolist(),
                    "transformer": p_trans.tolist(),
                    "xgboost": p_xgb.tolist(),
                },
                "stacker_predictions": stacker_pred.tolist(),
                "horizon": h,
                "horizon_min": h * 15,
                "n_predictions": len(final_pred),
            }

        return results

    def save(self, path: str = None):
        """Save all model checkpoints."""
        if path is None:
            path = CHECKPOINT_DIR
        os.makedirs(path, exist_ok=True)

        # Save PyTorch models
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

            # Save GP model + likelihood (GPyTorch needs state_dict approach)
            if h in self.gp_models:
                gp, lik = self.gp_models[h]
                gp_state = {
                    'model_state': gp.state_dict(),
                    'likelihood_state': lik.state_dict(),
                    'train_x': gp.train_inputs[0].numpy(),
                    'train_y': gp.train_targets.numpy(),
                }
                with open(os.path.join(path, f"gp_h{h}.pkl"), "wb") as f:
                    pickle.dump(gp_state, f)

        # Save training history
        with open(os.path.join(path, "training_history.json"), "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)

        print(f"Models saved to {path}")

    def load(self, path: str = None):
        """Load model checkpoints."""
        if path is None:
            path = CHECKPOINT_DIR

        from config import DEVICE

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
                train_x = torch.tensor(gp_state['train_x'], dtype=torch.float32)
                train_y = torch.tensor(gp_state['train_y'], dtype=torch.float32)
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                gp_model = ExactGPModel(train_x, train_y, likelihood)
                gp_model.load_state_dict(gp_state['model_state'])
                likelihood.load_state_dict(gp_state['likelihood_state'])
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
    verbose: bool = True
) -> Dict[str, GNSSEnsemble]:
    """
    Train ensembles for all satellites in the dataset.

    Returns
    -------
    ensembles : dict — satellite_id → GNSSEnsemble
    """
    # Auto-detect error column from dataset format
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

        ensemble = GNSSEnsemble()
        ensemble.fit(train_series, epochs=epochs, quick=quick, verbose=verbose)
        ensembles[sat_id] = ensemble

    return ensembles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GNSS Ensemble Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data")
    parser.add_argument("--error-col", type=str, default=None,
                        help="Error column to train on (auto-detected if omitted)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true", help="Quick training (5 epochs)")
    parser.add_argument("--satellite", type=str, default=None, help="Train single satellite")
    args = parser.parse_args()

    # Load data
    dataset = GNSSDataset(args.data)
    print("Dataset loaded:", dataset.summary()["n_satellites"], "satellites")

    # Auto-detect error column
    error_col = args.error_col or dataset.get_default_error_col()
    print(f"Using error column: {error_col}")

    if args.satellite:
        # Train single satellite
        train_series, _, _ = dataset.get_satellite_data(
            args.satellite, error_col, normalize=False
        )
        ensemble = GNSSEnsemble()
        results = ensemble.fit(train_series, epochs=args.epochs, quick=args.quick)
        ensemble.save()

        # Save results
        with open(os.path.join(RESULTS_DIR, f"train_results_{args.satellite}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        # Train all satellites
        ensembles = train_all_satellites(
            dataset, error_col, args.epochs, args.quick
        )
        # Save first ensemble as default
        first_id = dataset.satellite_ids[0]
        ensembles[first_id].save()
