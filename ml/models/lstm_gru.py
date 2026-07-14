"""
Base Model 1: LSTM-GRU Network (Paper §III-C)
===============================================
Hybrid recurrent network for short-term drift capture.

Architecture (paper eqs. 1-8):
  Input(96×1) → LSTM(hidden, layers) → GRU(hidden, layers)
  → last hidden → Dense(32) → ReLU → Dense(1)

Improvements over original ridge_stacker_gnss.py:
  - Learning rate scheduling (ReduceLROnPlateau)
  - Early stopping with patience
  - Configurable via config.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, GRAD_CLIP,
    LSTM_HIDDEN, LSTM_LAYERS, GRU_HIDDEN, GRU_LAYERS,
    LSTM_DROPOUT, LSTM_HEAD_DIM, PATIENCE, LR_PATIENCE
)


class LSTMGRUModel(nn.Module):
    """
    LSTM-GRU hybrid for sequential error prediction.

    Paper §III-C:
      "the LSTM layer handles the input series {x_t}, keeping vital
       earlier data alive via controlled adjustments"
      "From the LSTM, signals move into a GRU section"
    """

    def __init__(
        self,
        input_size: int = 1,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        gru_hidden: int = GRU_HIDDEN,
        gru_layers: int = GRU_LAYERS,
        dropout: float = LSTM_DROPOUT,
        head_dim: int = LSTM_HEAD_DIM
    ):
        super().__init__()

        # LSTM encoder — captures long-term dependencies (paper eqs. 1-6)
        self.lstm = nn.LSTM(
            input_size, lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # GRU refinement — lighter gating (paper eqs. 7-10)
        self.gru = nn.GRU(
            lstm_hidden, gru_hidden, gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        # Output head — two dense layers (paper §III-C)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, 1)

        Returns
        -------
        predictions : (batch,)
        """
        out, _ = self.lstm(x)          # (B, T, lstm_hidden)
        out, _ = self.gru(out)         # (B, T, gru_hidden)
        out = out[:, -1, :]            # last timestep
        return self.head(out).squeeze(-1)


def train_lstm_gru(
    X_seq_tr: np.ndarray,
    y_tr: np.ndarray,
    X_seq_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    verbose: bool = True,
    pretrained_model: Optional["LSTMGRUModel"] = None,
    pretrained_scaler_y: Optional[StandardScaler] = None,
    ft_lr: float = 5e-5,
) -> Tuple["LSTMGRUModel", StandardScaler, list]:
    """
    Train (or fine-tune) the LSTM-GRU model.

    Paper §III-C:
      "Adam optimizer together with gradient clipping"
      "target values get scaled down — then returned to normal size"
      "When that held-back portion shows no real progress,
       the system slows down on its own"

    Parameters
    ----------
    pretrained_model : LSTMGRUModel or None
        If provided, fine-tunes from these weights (transfer learning).
        Uses ft_lr instead of LEARNING_RATE for the optimizer.
    pretrained_scaler_y : StandardScaler or None
        If provided, skips refitting the target scaler (keeps training
        distribution scale consistent with the existing model).
    ft_lr : float
        Learning rate used when fine-tuning from pretrained_model.

    Returns
    -------
    model : LSTMGRUModel
    scaler_y : StandardScaler (for inverse transform at prediction)
    history : list of (train_loss, val_loss) per epoch
    """
    # ── Initialise model and optimizer ──
    if pretrained_model is not None:
        model = pretrained_model   # continue from existing weights
        lr = ft_lr
    else:
        model = LSTMGRUModel().to(DEVICE)
        lr = LEARNING_RATE

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE
    )
    loss_fn = nn.MSELoss()

    # ── Scale targets ──
    if pretrained_scaler_y is not None:
        scaler_y = pretrained_scaler_y   # reuse existing scale
        y_tr_scaled  = scaler_y.transform(y_tr.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    else:
        scaler_y = StandardScaler()
        y_tr_scaled  = scaler_y.fit_transform(y_tr.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    # DataLoader
    ds_tr = TensorDataset(
        torch.tensor(X_seq_tr, dtype=torch.float32),
        torch.tensor(y_tr_scaled, dtype=torch.float32)
    )
    loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

    # Validation tensors
    xv = torch.tensor(X_seq_val, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val_scaled, dtype=torch.float32).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_pred = model(xv)
            val_loss = loss_fn(val_pred, yv).item()

        train_loss = np.mean(train_losses)
        history.append((train_loss, val_loss))
        scheduler.step(val_loss)

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1}")
                break

        if verbose and (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:>3}/{epochs} | "
                  f"train={train_loss:.6f} val={val_loss:.6f} lr={lr_now:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, scaler_y, history


def predict_lstm_gru(
    model: LSTMGRUModel,
    scaler_y: StandardScaler,
    X_seq: np.ndarray
) -> np.ndarray:
    """
    Generate predictions, inverse-transforming back to original scale.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        preds_scaled = model(x).cpu().numpy()
    return scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
