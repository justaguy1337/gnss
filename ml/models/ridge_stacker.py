"""
Ridge Regression Meta-Learner (Paper §III-F)
==============================================
Per-horizon stacking of base model predictions.

Paper §III-F (eqs. 14-15):
  "Ridge regression weaves these together using past predictions
   made during cross-validation rounds"
  "One unique meta-learner handles each forecast window"

Improvement over original ridge_stacker_gnss.py:
  - 5-fold cross-validation for stacking (not simple train/val split)
  - Prevents information leakage into the stacking layer
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import Dict, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RIDGE_ALPHA, N_CV_FOLDS


class RidgeStacker:
    """
    Per-horizon Ridge regression meta-learner.

    Paper §III-F:
      ŷ_h = w₁·p_LSTM + w₂·p_Transformer + w₃·p_XGB + b
      L = Σ(yᵢ - ŷᵢ)² + λ·Σwⱼ²

    One instance per prediction horizon h.
    """

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.scaler = StandardScaler()
        self.horizon = None
        self.weights = None
        self.bias = None

    def fit(
        self,
        lstm_pred: np.ndarray,
        transformer_pred: np.ndarray,
        xgb_pred: np.ndarray,
        y_true: np.ndarray,
        horizon: int
    ) -> 'RidgeStacker':
        """
        Fit the Ridge stacker on out-of-fold predictions.

        Parameters
        ----------
        lstm_pred : (N,) — LSTM-GRU out-of-fold predictions
        transformer_pred : (N,) — Transformer out-of-fold predictions
        xgb_pred : (N,) — XGBoost out-of-fold predictions
        y_true : (N,) — ground truth
        horizon : int — prediction horizon (for logging)
        """
        self.horizon = horizon

        # Stack predictions into meta-feature matrix [p_LSTM, p_Trans, p_XGB]
        X_meta = np.column_stack([lstm_pred, transformer_pred, xgb_pred])
        X_meta = self.scaler.fit_transform(X_meta)

        self.model.fit(X_meta, y_true)
        self.weights = self.model.coef_
        self.bias = self.model.intercept_

        return self

    def predict(
        self,
        lstm_pred: np.ndarray,
        transformer_pred: np.ndarray,
        xgb_pred: np.ndarray
    ) -> np.ndarray:
        """
        Blend base model predictions.

        Returns
        -------
        blended : np.ndarray — combined predictions
        """
        X_meta = np.column_stack([lstm_pred, transformer_pred, xgb_pred])
        X_meta = self.scaler.transform(X_meta)
        return self.model.predict(X_meta)

    def get_weights_info(self) -> dict:
        """Return model weights for analysis/visualization."""
        return {
            "horizon": self.horizon,
            "horizon_min": self.horizon * 15,
            "weights": {
                "LSTM-GRU": float(self.weights[0]),
                "Transformer": float(self.weights[1]),
                "XGBoost": float(self.weights[2]),
            },
            "bias": float(self.bias),
        }

    def __repr__(self):
        if self.weights is not None:
            return (
                f"RidgeStacker(h={self.horizon}) | "
                f"LSTM={self.weights[0]:.3f} "
                f"Trans={self.weights[1]:.3f} "
                f"XGB={self.weights[2]:.3f}"
            )
        return "RidgeStacker(not fitted)"


def generate_oof_predictions(
    train_fn,
    predict_fn,
    X_data,
    y_data,
    n_folds: int = N_CV_FOLDS,
    **train_kwargs
) -> np.ndarray:
    """
    Generate out-of-fold (OOF) predictions using K-fold cross-validation.

    Paper §III-F:
      "Those earlier guesses sit inside a structure... matched against
       actual outcomes. Only data held aside for checking — not used
       when first models were built — feeds into this step."

    Parameters
    ----------
    train_fn : callable — function(X_train, y_train, X_val, y_val, **kwargs) → model
    predict_fn : callable — function(model, X) → predictions
    X_data : training data (can be tuple for (X_seq, X_tab))
    y_data : np.ndarray of targets
    n_folds : int — number of CV folds

    Returns
    -------
    oof_preds : np.ndarray of shape (len(y_data),) — out-of-fold predictions
    """
    oof_preds = np.zeros(len(y_data), dtype=np.float32)
    kf = KFold(n_splits=n_folds, shuffle=False)  # temporal order preserved

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(y_data)):
        # Handle both single array and tuple inputs
        if isinstance(X_data, tuple):
            X_tr = tuple(x[train_idx] for x in X_data)
            X_vl = tuple(x[val_idx] for x in X_data)
        else:
            X_tr = X_data[train_idx]
            X_vl = X_data[val_idx]

        y_tr = y_data[train_idx]
        y_vl = y_data[val_idx]

        # Train on fold
        result = train_fn(X_tr, y_tr, X_vl, y_vl, **train_kwargs)

        # Predict on held-out fold
        preds = predict_fn(result, X_vl)
        oof_preds[val_idx] = preds

    return oof_preds
