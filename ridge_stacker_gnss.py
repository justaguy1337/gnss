"""
Ridge Stacker Ensemble for GNSS Error Prediction
=================================================
Pipeline:
  1. Train base models: LSTM-GRU, Transformer (stub), XGBoost
  2. Generate out-of-fold (OOF) predictions on validation data
  3. Train one Ridge stacker per prediction horizon
  4. At inference: base models predict → stacker blends → GP corrects residuals

Dependencies:
  pip install numpy pandas scikit-learn torch xgboost gpytorch
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import normaltest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import gpytorch
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HORIZONS = [1, 2, 4, 8, 96]        # 15min, 30min, 1hr, 2hr, 24hr steps
SEQ_LEN   = 96                      # lookback window (24 hrs of 15-min data)
BATCH     = 32
EPOCHS    = 50
LR        = 1e-3
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def make_features(series: np.ndarray, horizon: int, seq_len: int = SEQ_LEN):
    """
    Build (X_seq, X_tab, y) for a single satellite error time series.

    X_seq  : (N, seq_len, 1)       — raw sequence for LSTM/Transformer
    X_tab  : (N, n_features)       — tabular features for XGBoost
    y      : (N,)                  — target value h steps ahead
    """
    n = len(series)
    X_seq, X_tab, y = [], [], []

    for i in range(seq_len, n - horizon):
        window = series[i - seq_len:i]

        # --- tabular features ---
        lags      = window[-1], window[-2], window[-4], window[-8], window[-96 % seq_len]
        roll_mean = window[-12:].mean()
        roll_std  = window[-12:].std() + 1e-8
        roll_max  = window[-12:].max()
        roll_min  = window[-12:].min()

        # sin/cos time encoding (index as proxy for time-of-day)
        t         = i / n
        sin_24h   = np.sin(2 * np.pi * t)
        cos_24h   = np.cos(2 * np.pi * t)
        sin_12h   = np.sin(4 * np.pi * t)
        cos_12h   = np.cos(4 * np.pi * t)

        # rate of change
        diff1     = window[-1] - window[-2]
        diff2     = window[-2] - window[-3]
        accel     = diff1 - diff2

        tab = np.array([
            *lags, roll_mean, roll_std, roll_max, roll_min,
            sin_24h, cos_24h, sin_12h, cos_12h,
            diff1, diff2, accel,
            float(horizon)               # horizon as a feature
        ], dtype=np.float32)

        X_seq.append(window.reshape(-1, 1).astype(np.float32))
        X_tab.append(tab)
        y.append(series[i + horizon - 1])

    return (
        np.array(X_seq),
        np.array(X_tab),
        np.array(y, dtype=np.float32)
    )


def train_val_split(X_seq, X_tab, y, val_frac=0.15):
    n     = len(y)
    split = int(n * (1 - val_frac))
    return (
        X_seq[:split], X_seq[split:],
        X_tab[:split], X_tab[split:],
        y[:split],     y[split:]
    )


# ─────────────────────────────────────────────
# 2. BASE MODEL A — LSTM-GRU
# ─────────────────────────────────────────────

class LSTM_GRU(nn.Module):
    def __init__(self, input_size=1, hidden=64, lstm_layers=1, gru_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.gru  = nn.GRU(hidden, hidden, gru_layers,
                           batch_first=True, dropout=dropout if gru_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _  = self.lstm(x)         # (B, T, hidden)
        out, _  = self.gru(out)        # (B, T, hidden)
        out     = out[:, -1, :]        # last timestep
        return self.head(out).squeeze(-1)


def train_lstm_gru(X_seq_tr, y_tr, X_seq_val, y_val):
    model    = LSTM_GRU().to(DEVICE)
    opt      = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn  = nn.MSELoss()
    scaler_y = StandardScaler()

    y_tr_s   = scaler_y.fit_transform(y_tr.reshape(-1,1)).flatten()
    y_val_s  = scaler_y.transform(y_val.reshape(-1,1)).flatten()

    ds_tr  = TensorDataset(torch.tensor(X_seq_tr), torch.tensor(y_tr_s))
    loader = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)

    best_val, best_state = np.inf, None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_seq_val).to(DEVICE)
            val_loss = loss_fn(model(xv), torch.tensor(y_val_s).to(DEVICE)).item()
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, scaler_y


def predict_lstm_gru(model, scaler_y, X_seq):
    model.eval()
    with torch.no_grad():
        preds_s = model(torch.tensor(X_seq).to(DEVICE)).cpu().numpy()
    return scaler_y.inverse_transform(preds_s.reshape(-1,1)).flatten()


# ─────────────────────────────────────────────
# 3. BASE MODEL B — TRANSFORMER
# ─────────────────────────────────────────────

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.input_proj  = nn.Linear(input_size, d_model)
        self.pos_emb     = nn.Embedding(seq_len, d_model)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder     = nn.TransformerEncoder(enc_layer, num_layers)
        self.head        = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, T, _   = x.shape
        pos       = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out       = self.input_proj(x) + self.pos_emb(pos)
        out       = self.encoder(out)
        out       = out[:, -1, :]     # last token
        return self.head(out).squeeze(-1)


def train_transformer(X_seq_tr, y_tr, X_seq_val, y_val):
    model    = TimeSeriesTransformer(seq_len=X_seq_tr.shape[1]).to(DEVICE)
    opt      = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn  = nn.MSELoss()
    scaler_y = StandardScaler()

    y_tr_s  = scaler_y.fit_transform(y_tr.reshape(-1,1)).flatten()
    y_val_s = scaler_y.transform(y_val.reshape(-1,1)).flatten()

    ds_tr   = TensorDataset(torch.tensor(X_seq_tr), torch.tensor(y_tr_s))
    loader  = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)

    best_val, best_state = np.inf, None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_seq_val).to(DEVICE)
            val_loss = loss_fn(model(xv), torch.tensor(y_val_s).to(DEVICE)).item()
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, scaler_y


def predict_transformer(model, scaler_y, X_seq):
    model.eval()
    with torch.no_grad():
        preds_s = model(torch.tensor(X_seq).to(DEVICE)).cpu().numpy()
    return scaler_y.inverse_transform(preds_s.reshape(-1,1)).flatten()


# ─────────────────────────────────────────────
# 4. BASE MODEL C — XGBOOST
# ─────────────────────────────────────────────

def train_xgboost(X_tab_tr, y_tr, X_tab_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        early_stopping_rounds = 20,
        eval_metric      = "rmse",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(
        X_tab_tr, y_tr,
        eval_set=[(X_tab_val, y_val)],
        verbose=False
    )
    return model


# ─────────────────────────────────────────────
# 5. GAUSSIAN PROCESS (residual layer)
# ─────────────────────────────────────────────

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        # Matern + Periodic kernel: captures smooth drift + orbital cycle
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5) +
            gpytorch.kernels.PeriodicKernel()
        )

    def forward(self, x):
        mean  = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def train_gp(X_train: np.ndarray, residuals: np.ndarray, n_iter=100):
    """
    X_train   : (N,) — e.g. time index or stacker prediction as input
    residuals : (N,) — actual - stacker_pred
    """
    train_x    = torch.tensor(X_train, dtype=torch.float32)
    train_y    = torch.tensor(residuals, dtype=torch.float32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp         = ExactGPModel(train_x, train_y, likelihood)

    gp.train(); likelihood.train()
    opt = torch.optim.Adam(gp.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    for _ in range(n_iter):
        opt.zero_grad()
        output = gp(train_x)
        loss   = -mll(output, train_y)
        loss.backward()
        opt.step()

    gp.eval(); likelihood.eval()
    return gp, likelihood


def predict_gp(gp, likelihood, X_test: np.ndarray):
    """Returns (mean, std) of GP posterior."""
    test_x = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(gp(test_x))
    return pred.mean.numpy(), pred.stddev.numpy()


# ─────────────────────────────────────────────
# 6. RIDGE STACKER (one per horizon)
# ─────────────────────────────────────────────

class RidgeStacker:
    """
    Trained on out-of-fold predictions from the 3 base models.
    One instance per prediction horizon.
    """
    def __init__(self, alpha=1.0):
        self.model   = Ridge(alpha=alpha, fit_intercept=True)
        self.scaler  = StandardScaler()
        self.horizon = None

    def fit(self, lstm_pred, transformer_pred, xgb_pred, y_true, horizon):
        self.horizon = horizon
        X_meta = np.column_stack([lstm_pred, transformer_pred, xgb_pred])
        X_meta = self.scaler.fit_transform(X_meta)
        self.model.fit(X_meta, y_true)
        print(f"  Horizon {horizon:>2} steps | weights → "
              f"LSTM={self.model.coef_[0]:.3f}  "
              f"Transformer={self.model.coef_[1]:.3f}  "
              f"XGB={self.model.coef_[2]:.3f}")
        return self

    def predict(self, lstm_pred, transformer_pred, xgb_pred):
        X_meta = np.column_stack([lstm_pred, transformer_pred, xgb_pred])
        X_meta = self.scaler.transform(X_meta)
        return self.model.predict(X_meta)


# ─────────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────────

class GNSSEnsemble:
    def __init__(self):
        self.lstm_models        = {}    # horizon → (model, scaler_y)
        self.transformer_models = {}
        self.xgb_models         = {}
        self.stackers           = {}    # horizon → RidgeStacker
        self.gp_models          = {}    # horizon → (gp, likelihood)

    def fit(self, series: np.ndarray):
        """
        series : 1D numpy array of error values, shape (672,) for 7 days @ 15-min
        """
        print("=" * 55)
        print("Training GNSS Ensemble")
        print("=" * 55)

        for h in HORIZONS:
            print(f"\n[Horizon = {h} step(s) = {h*15} min]")

            X_seq, X_tab, y = make_features(series, horizon=h)
            (X_seq_tr, X_seq_val,
             X_tab_tr, X_tab_val,
             y_tr,     y_val)       = train_val_split(X_seq, X_tab, y)

            # ── train base models ──
            print("  Training LSTM-GRU...")
            lstm_m, lstm_sy = train_lstm_gru(X_seq_tr, y_tr, X_seq_val, y_val)
            self.lstm_models[h] = (lstm_m, lstm_sy)

            print("  Training Transformer...")
            trans_m, trans_sy = train_transformer(X_seq_tr, y_tr, X_seq_val, y_val)
            self.transformer_models[h] = (trans_m, trans_sy)

            print("  Training XGBoost...")
            xgb_m = train_xgboost(X_tab_tr, y_tr, X_tab_val, y_val)
            self.xgb_models[h] = xgb_m

            # ── generate val predictions for stacker ──
            p_lstm  = predict_lstm_gru(lstm_m, lstm_sy, X_seq_val)
            p_trans = predict_transformer(trans_m, trans_sy, X_seq_val)
            p_xgb   = xgb_m.predict(X_tab_val)

            # ── train stacker ──
            print("  Training Ridge stacker...")
            stacker = RidgeStacker(alpha=1.0).fit(p_lstm, p_trans, p_xgb, y_val, h)
            self.stackers[h] = stacker

            # ── compute residuals and train GP ──
            stacker_pred = stacker.predict(p_lstm, p_trans, p_xgb)
            residuals    = y_val - stacker_pred
            time_idx     = np.arange(len(residuals), dtype=np.float32) / len(residuals)

            print("  Training Gaussian Process on residuals...")
            gp, lik = train_gp(time_idx, residuals.astype(np.float32))
            self.gp_models[h] = (gp, lik)

            # ── report val RMSE ──
            final_pred = stacker_pred + predict_gp(gp, lik, time_idx)[0]
            rmse       = np.sqrt(mean_squared_error(y_val, final_pred))
            stat, pval = normaltest(y_val - final_pred)
            print(f"  Val RMSE={rmse:.5f} | Residual normality p={pval:.4f}"
                  f" {'[NORMAL]' if pval > 0.05 else '[non-normal]'}")

        print("\nTraining complete.")

    def predict_day8(self, series_7day: np.ndarray):
        """
        Given the full 7-day series, predict 96 intervals for Day 8.
        Returns dict: horizon → array of predictions
        """
        results = {}
        for h in HORIZONS:
            X_seq, X_tab, _ = make_features(series_7day, horizon=h)

            # use only the last available window (most recent context)
            X_seq_last = X_seq[[-1]]
            X_tab_last = X_tab[[-1]]

            p_lstm  = predict_lstm_gru(*self.lstm_models[h], X_seq_last)
            p_trans = predict_transformer(*self.transformer_models[h], X_seq_last)
            p_xgb   = self.xgb_models[h].predict(X_tab_last)

            stacker_pred = self.stackers[h].predict(p_lstm, p_trans, p_xgb)

            # GP correction
            gp, lik      = self.gp_models[h]
            time_idx     = np.array([1.0], dtype=np.float32)   # end of known series
            gp_mean, gp_std = predict_gp(gp, lik, time_idx)

            final   = stacker_pred + gp_mean
            results[h] = {
                "prediction"  : float(final[0]),
                "uncertainty" : float(gp_std[0]),
                "horizon_min" : h * 15,
            }

        return results


# ─────────────────────────────────────────────
# 8. EVALUATION UTILITIES
# ─────────────────────────────────────────────

def evaluate_normality(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Checks whether residuals are normally distributed.
    This is the primary evaluation criterion in the problem statement.
    """
    residuals = y_true - y_pred
    stat, pval = normaltest(residuals)
    return {
        "rmse"          : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae"           : float(np.mean(np.abs(residuals))),
        "residual_mean" : float(residuals.mean()),
        "residual_std"  : float(residuals.std()),
        "normality_stat": float(stat),
        "normality_pval": float(pval),
        "is_normal"     : pval > 0.05,
    }


def print_evaluation(results: dict):
    print("\n" + "=" * 55)
    print("Evaluation Report")
    print("=" * 55)
    for k, v in results.items():
        print(f"  {k:<20}: {v}")


# ─────────────────────────────────────────────
# 9. DEMO / ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Synthetic demo data (replace with real GNSS error series) ──
    np.random.seed(42)
    t       = np.linspace(0, 7 * 2 * np.pi, 7 * 96)
    series  = (
        0.5  * np.sin(t)                          # 24-hr orbital cycle
      + 0.2  * np.sin(2 * t)                      # 12-hr harmonic
      + 0.05 * np.cumsum(np.random.randn(len(t))) # random walk drift
      + 0.1  * np.random.randn(len(t))            # measurement noise
    ).astype(np.float32)

    # ── Train ──
    ensemble = GNSSEnsemble()
    ensemble.fit(series)

    # ── Predict Day 8 ──
    day8_preds = ensemble.predict_day8(series)

    print("\n" + "=" * 55)
    print("Day 8 Predictions")
    print("=" * 55)
    for h, info in day8_preds.items():
        print(f"  +{info['horizon_min']:>4} min | "
              f"pred={info['prediction']:+.5f}  "
              f"± {info['uncertainty']:.5f}")
