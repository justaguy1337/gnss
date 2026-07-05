"""
Base Model 2: Temporal Transformer (Paper §III-D)
==================================================
Transformer encoder for long-range orbital cycle capture.

Architecture (paper eqs. 9-11):
  Input(96×1) → Linear(1→d_model) + LearnedPositionalEncoding
  → TransformerEncoder(layers, heads) → LastToken → Dense(1)

Paper §III-D:
  "trained position markers to notice patterns across distant points"
  "attention heads scan all moment pairs at once"
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, GRAD_CLIP,
    SEQUENCE_LENGTH, TRANS_D_MODEL, TRANS_NHEAD, TRANS_NUM_LAYERS,
    TRANS_DIM_FF, TRANS_DROPOUT, TRANS_HEAD_DIM,
    PATIENCE, LR_PATIENCE
)


class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder for time-series forecasting.

    Paper §III-D:
      "a Transformer encoder uses trained position markers to notice
       patterns across distant points in time, also catching repeating cycles"

    Uses learned positional embeddings (nn.Embedding), not sinusoidal,
    because the paper says "trained position markers".
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = TRANS_D_MODEL,
        nhead: int = TRANS_NHEAD,
        num_layers: int = TRANS_NUM_LAYERS,
        dim_ff: int = TRANS_DIM_FF,
        dropout: float = TRANS_DROPOUT,
        seq_len: int = SEQUENCE_LENGTH,
        head_dim: int = TRANS_HEAD_DIM
    ):
        super().__init__()

        # Project input to d_model dimensions
        self.input_proj = nn.Linear(input_size, d_model)

        # Learned positional encoding (paper: "trained position markers")
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # Transformer encoder layers (paper eq. 9-11)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output head — maps last token to prediction
        self.head = nn.Sequential(
            nn.Linear(d_model, head_dim),
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
        B, T, _ = x.shape

        # Position indices
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        # Project + add positional encoding
        out = self.input_proj(x) + self.pos_emb(pos)

        # Self-attention across all timestamp pairs (paper eq. 9)
        out = self.encoder(out)

        # Last token representation → prediction (paper §III-D)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


def train_transformer(
    X_seq_tr: np.ndarray,
    y_tr: np.ndarray,
    X_seq_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    verbose: bool = True
) -> Tuple[TimeSeriesTransformer, StandardScaler, list]:
    """
    Train the Transformer model.

    Paper §III-D:
      "training runs the same way the LSTM-GRU setup does —
       same settings across the board"

    Returns
    -------
    model : TimeSeriesTransformer
    scaler_y : StandardScaler
    history : list of (train_loss, val_loss)
    """
    seq_len = X_seq_tr.shape[1]
    model = TimeSeriesTransformer(seq_len=seq_len).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE
    )
    loss_fn = nn.MSELoss()

    # Scale targets
    scaler_y = StandardScaler()
    y_tr_scaled = scaler_y.fit_transform(y_tr.reshape(-1, 1)).flatten()
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
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:>3}/{epochs} | "
                  f"train={train_loss:.6f} val={val_loss:.6f} lr={lr:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, scaler_y, history


def predict_transformer(
    model: TimeSeriesTransformer,
    scaler_y: StandardScaler,
    X_seq: np.ndarray
) -> np.ndarray:
    """Generate predictions, inverse-transforming to original scale."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        preds_scaled = model(x).cpu().numpy()
    return scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
