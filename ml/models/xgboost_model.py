"""
Base Model 3: XGBoost (Paper §III-E)
======================================
Gradient boosted trees for feature-based pattern recognition.

Paper §III-E (eqs. 12-13):
  "built on boosted trees, specifically XGBoost, fed only tabular data X_tab"
  "the forecast distance h gets tossed directly into the mix as just another column"
  "one unified model handles every prediction window at once"
"""

import numpy as np
import xgboost as xgb
from typing import Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE, XGB_EARLY_STOPPING
)


def train_xgboost(
    X_tab_tr: np.ndarray,
    y_tr: np.ndarray,
    X_tab_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = False
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor on tabular features.

    Paper §III-E:
      "Training minimizes a regularized objective"
      "When the validation error begins to rise, training stops"
      "Part of each dataset gets used at every step, along with
       a fraction of the features"

    Parameters
    ----------
    X_tab_tr : (N_train, n_features) — training tabular features
    y_tr : (N_train,)               — training targets
    X_tab_val : (N_val, n_features) — validation tabular features
    y_val : (N_val,)                — validation targets

    Returns
    -------
    model : xgb.XGBRegressor
    """
    model = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        early_stopping_rounds=XGB_EARLY_STOPPING,
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
        tree_method="hist",  # faster training
    )

    model.fit(
        X_tab_tr, y_tr,
        eval_set=[(X_tab_val, y_val)],
        verbose=verbose
    )

    return model


def predict_xgboost(
    model: xgb.XGBRegressor,
    X_tab: np.ndarray
) -> np.ndarray:
    """Generate predictions from XGBoost model."""
    return model.predict(X_tab)
