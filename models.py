"""
NeuroTrade — Deep Learning Models
models.py — LSTM, TCN, and TFT models for time-series price prediction.

Each model:
  - Accepts OHLCV + indicator DataFrames from the data pipeline
  - Predicts next-bar returns (regression) or direction (classification)
  - Exposes a unified train / predict / signal interface
  - Generates PositionSide signals compatible with backtester.py
"""

import numpy as np
import pandas as pd
import warnings
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Try importing backtester enums for signal compatibility
try:
    from backtester import PositionSide
except ImportError:
    try:
        from Backtester import PositionSide
    except ImportError:
        from enum import Enum

        class PositionSide(Enum):
            LONG = "long"
            SHORT = "short"
            FLAT = "flat"


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Shared configuration for all deep-learning models."""
    # data
    sequence_length: int = 30          # lookback window (bars)
    forecast_horizon: int = 1          # predict N bars ahead
    train_split: float = 0.8           # fraction for training
    target_col: str = "Close"          # column to predict returns from

    # training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10                 # early-stopping patience
    device: str = "cpu"

    # signal thresholds
    long_threshold: float = 0.001      # predicted return > this → LONG
    short_threshold: float = -0.001    # predicted return < this → SHORT


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_EXCLUDE = {"Ticker", "Date", "Datetime"}

def prepare_features(df: pd.DataFrame,
                     target_col: str = "Close",
                     forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y (forward returns) from OHLCV+indicator df.

    Returns
    -------
    features : pd.DataFrame   – numeric columns, NaN-free
    target   : pd.Series      – forward log-return shifted by forecast_horizon
    """
    df = df.copy()

    # target: forward log-return
    future_close = df[target_col].shift(-forecast_horizon)
    target = np.log(future_close / df[target_col])
    target.name = "target"

    # drop non-numeric and excluded columns
    numeric = df.select_dtypes(include=[np.number])
    drop_cols = [c for c in numeric.columns if c in FEATURE_EXCLUDE]
    numeric = numeric.drop(columns=drop_cols, errors="ignore")

    # drop rows with NaN in either features or target
    valid = numeric.notna().all(axis=1) & target.notna()
    return numeric.loc[valid], target.loc[valid]


def build_sequences(features: np.ndarray,
                    target: np.ndarray,
                    seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length `seq_len` over features/target arrays.

    Returns
    -------
    X : (N, seq_len, n_features)
    y : (N,)
    """
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset wrapping numpy sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeatureScaler:
    """Per-feature z-score scaler that stores mean/std for inference."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # avoid div-by-zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — LSTM
# ══════════════════════════════════════════════════════════════════════════════

class LSTMNet(nn.Module):
    """
    Multi-layer LSTM with dropout for time-series regression.

    Architecture:
      Input(seq_len, n_features) → LSTM(hidden×layers) → FC → output(1)
    """

    def __init__(self, n_features: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]       # take last timestep
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — TCN  (Temporal Convolutional Network)
# ══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """1-D causal convolution with dilation — no future leakage."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Residual block for TCN with two causal convolutions,
    weight normalisation, ReLU, and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation).conv
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation).conv
        )
        self.padding1 = (kernel_size - 1) * dilation
        self.padding2 = (kernel_size - 1) * dilation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        if self.padding1 > 0:
            out = out[:, :, :-self.padding1]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.padding2 > 0:
            out = out[:, :, :-self.padding2]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNet(nn.Module):
    """
    Temporal Convolutional Network with stacked dilated causal conv blocks.

    Architecture:
      Input(seq_len, features) → [TCNBlock(dilation=2^k)]×layers → FC → output(1)
    """

    def __init__(self, n_features: int, num_channels: List[int] = None,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]

        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) → transpose to (batch, features, seq_len)
        out = x.transpose(1, 2)
        out = self.network(out)
        out = out[:, :, -1]                    # last timestep
        out = self.fc(out)
        return out.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 3 — TFT  (Temporal Fusion Transformer)
# ══════════════════════════════════════════════════════════════════════════════

class GatedLinearUnit(nn.Module):
    """GLU activation: splits input in half, applies sigmoid gate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT with skip connection + layer norm."""

    def __init__(self, d_model: int, d_hidden: int = None,
                 dropout: float = 0.1):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = GatedLinearUnit(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.elu(self.fc1(x))
        out = self.dropout(self.fc2(out))
        out = self.gate(out)
        return self.layer_norm(out + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network: learns which input features matter most.
    Produces per-variable importance weights via softmax over GRN outputs.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model * n_features)
        self.grn_var = nn.ModuleList([
            GatedResidualNetwork(d_model, dropout=dropout)
            for _ in range(n_features)
        ])
        self.grn_flat = GatedResidualNetwork(d_model * n_features,
                                              d_model * n_features,
                                              dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features)
        batch, seq_len, _ = x.shape

        # flatten projection → variable weights
        flat = x.reshape(batch * seq_len, self.n_features)
        proj = self.input_proj(flat)
        proj = proj.reshape(batch * seq_len, self.n_features, self.d_model)

        # per-variable GRN
        var_outputs = []
        for i in range(self.n_features):
            var_outputs.append(self.grn_var[i](proj[:, i, :]))
        var_outputs = torch.stack(var_outputs, dim=1)  # (B*T, n_feat, d_model)

        # flat GRN for selection weights
        flat_input = proj.reshape(batch * seq_len, -1)
        flat_out = self.grn_flat(flat_input)
        weights = flat_out.reshape(batch * seq_len, self.n_features, self.d_model)
        weights = self.softmax(weights.mean(dim=-1))  # (B*T, n_feat)

        # weighted sum
        weighted = (var_outputs * weights.unsqueeze(-1)).sum(dim=1)  # (B*T, d_model)
        weighted = weighted.reshape(batch, seq_len, self.d_model)

        # reshape weights for interpretability
        importance = weights.reshape(batch, seq_len, self.n_features).mean(dim=1)

        return weighted, importance


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask for temporal data."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        return self.layer_norm(attn_out + x)


class TFTNet(nn.Module):
    """
    Simplified Temporal Fusion Transformer for time-series prediction.

    Architecture:
      Input → Variable Selection → Positional Encoding →
      LSTM Encoder → Multi-Head Attention → GRN → FC → output(1)

    Key components:
      - Variable Selection Network: learns feature importance
      - LSTM encoder: captures local temporal patterns
      - Self-attention: captures long-range dependencies
      - Gated Residual Networks: control information flow
    """

    def __init__(self, n_features: int, d_model: int = 64,
                 n_heads: int = 4, lstm_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # variable selection
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        # positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        # temporal processing
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # self-attention
        self.attention = TemporalSelfAttention(d_model, n_heads, dropout)

        # post-attention GRN
        self.grn = GatedResidualNetwork(d_model, dropout=dropout)

        # output
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape

        # variable selection
        selected, importance = self.vsn(x)

        # add positional encoding
        selected = selected + self.pos_encoder[:, :seq_len, :]

        # LSTM encoder
        lstm_out, _ = self.lstm(selected)

        # self-attention
        attn_out = self.attention(lstm_out)

        # GRN
        grn_out = self.grn(attn_out[:, -1, :])  # last timestep

        # prediction
        pred = self.fc(grn_out).squeeze(-1)

        return pred, importance


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED TRAINER / PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class DeepLearningModel:
    """
    Unified wrapper that handles training, prediction, and signal generation
    for any of the three architectures (LSTM, TCN, TFT).

    Usage
    -----
    >>> model = DeepLearningModel("lstm", config)
    >>> history = model.train(df)
    >>> predictions = model.predict(df)
    >>> signal = model.get_signal(df, i)
    """

    ARCHITECTURES = {"lstm", "tcn", "tft"}

    def __init__(self, arch: str = "lstm",
                 config: ModelConfig = None):
        if arch not in self.ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{arch}'. "
                             f"Choose from {self.ARCHITECTURES}")
        self.arch = arch
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device)
        self.model: Optional[nn.Module] = None
        self.scaler = FeatureScaler()
        self.feature_names: List[str] = []
        self.trained = False
        self._feature_importance: Optional[np.ndarray] = None

    def _build_model(self, n_features: int) -> nn.Module:
        """Instantiate the chosen architecture."""
        if self.arch == "lstm":
            return LSTMNet(n_features, hidden_size=128,
                           num_layers=2, dropout=0.2)
        elif self.arch == "tcn":
            return TCNNet(n_features, num_channels=[64, 64, 64, 64],
                          kernel_size=3, dropout=0.2)
        elif self.arch == "tft":
            return TFTNet(n_features, d_model=64, n_heads=4,
                          lstm_layers=1, dropout=0.1)
        raise ValueError(f"Unknown architecture: {self.arch}")

    def train(self, df: pd.DataFrame,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model on an OHLCV+indicator DataFrame.

        Returns
        -------
        history : dict with 'train_loss' and 'val_loss' lists per epoch.
        """
        cfg = self.config

        # prepare features and target
        features, target = prepare_features(
            df, cfg.target_col, cfg.forecast_horizon
        )
        self.feature_names = list(features.columns)
        n_features = len(self.feature_names)

        feat_arr = features.values
        tgt_arr = target.values

        # train/val split (temporal — no shuffle)
        split_idx = int(len(feat_arr) * cfg.train_split)
        train_feat, val_feat = feat_arr[:split_idx], feat_arr[split_idx:]
        train_tgt, val_tgt = tgt_arr[:split_idx], tgt_arr[split_idx:]

        # scale features
        train_feat = self.scaler.fit_transform(train_feat)
        val_feat = self.scaler.transform(val_feat)

        # build sequences
        X_train, y_train = build_sequences(train_feat, train_tgt, cfg.sequence_length)
        X_val, y_val = build_sequences(val_feat, val_tgt, cfg.sequence_length)

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                f"Not enough data to build sequences. Need at least "
                f"{cfg.sequence_length + 1} rows after cleaning. "
                f"Got {len(train_feat)} train, {len(val_feat)} val rows."
            )

        # data loaders
        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds = TimeSeriesDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)

        # build model
        self.model = self._build_model(n_features).to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        wait = 0

        for epoch in range(cfg.epochs):
            # — train —
            self.model.train()
            train_losses = []
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                if self.arch == "tft":
                    pred, _ = self.model(xb)
                else:
                    pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # — validate —
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    if self.arch == "tft":
                        pred, _ = self.model(xb)
                    else:
                        pred = self.model(xb)
                    val_losses.append(criterion(pred, yb).item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            scheduler.step(avg_val)

            if verbose and (epoch + 1) % max(1, cfg.epochs // 10) == 0:
                print(f"  Epoch {epoch+1:3d}/{cfg.epochs}  "
                      f"train={avg_train:.6f}  val={avg_val:.6f}")

            # early stopping
            if avg_val < best_val:
                best_val = avg_val
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

        # restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.trained = True

        if verbose:
            print(f"  [{self.arch.upper()}] Training complete — "
                  f"best val loss: {best_val:.6f}")

        return history

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions (forward returns) for each valid row in df.

        Returns
        -------
        predictions : 1-D numpy array of predicted returns, length ≤ len(df).
        """
        if not self.trained or self.model is None:
            raise RuntimeError("Model not trained. Call .train(df) first.")

        cfg = self.config
        features, _ = prepare_features(df, cfg.target_col, cfg.forecast_horizon)

        # use only columns the model was trained on
        available = [c for c in self.feature_names if c in features.columns]
        if len(available) < len(self.feature_names):
            missing = set(self.feature_names) - set(available)
            for col in missing:
                features[col] = 0.0
        features = features[self.feature_names]

        feat_arr = self.scaler.transform(features.values)

        # build sequences (target placeholder)
        dummy_target = np.zeros(len(feat_arr))
        X, _ = build_sequences(feat_arr, dummy_target, cfg.sequence_length)

        if len(X) == 0:
            return np.array([])

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            if self.arch == "tft":
                preds, importance = self.model(X_t)
                self._feature_importance = importance.cpu().numpy().mean(axis=0)
            else:
                preds = self.model(X_t)

        return preds.cpu().numpy()

    def predict_latest(self, df: pd.DataFrame) -> float:
        """Return the single latest prediction (most recent bar)."""
        preds = self.predict(df)
        if len(preds) == 0:
            return 0.0
        return float(preds[-1])

    def get_signal(self, df: pd.DataFrame, i: int = -1,
                   **kwargs) -> Optional[PositionSide]:
        """
        Generate a trading signal for bar `i` — compatible with
        backtester.py's signal_func(df, i) interface.

        Returns PositionSide.LONG, SHORT, or None.
        """
        if not self.trained:
            return None

        cfg = self.config

        # need at least sequence_length + some warmup bars
        end = i if i >= 0 else len(df)
        start = max(0, end - cfg.sequence_length - 100)
        subset = df.iloc[start:end].copy()

        if len(subset) < cfg.sequence_length + 2:
            return None

        pred = self.predict_latest(subset)

        if pred > cfg.long_threshold:
            return PositionSide.LONG
        elif pred < cfg.short_threshold:
            return PositionSide.SHORT
        return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return feature importance (only available for TFT after prediction).
        """
        if self._feature_importance is None or not self.feature_names:
            return None
        return dict(zip(self.feature_names, self._feature_importance.tolist()))

    def save(self, path: str):
        """Save model weights and scaler state."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        torch.save({
            "arch": self.arch,
            "config": self.config,
            "state_dict": self.model.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_std": self.scaler.std_,
            "feature_names": self.feature_names,
        }, path)

    def load(self, path: str):
        """Load model weights and scaler state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.arch = checkpoint["arch"]
        self.config = checkpoint["config"]
        self.feature_names = checkpoint["feature_names"]
        self.scaler.mean_ = checkpoint["scaler_mean"]
        self.scaler.std_ = checkpoint["scaler_std"]

        n_features = len(self.feature_names)
        self.model = self._build_model(n_features).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.trained = True


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL FUNCTIONS  — plug into Backtester.run(df, signal_func)
# ══════════════════════════════════════════════════════════════════════════════

def _make_dl_signal(model: DeepLearningModel):
    """
    Create a signal function closure compatible with Backtester.run().

    The backtester calls:  signal_func(df, i, **signal_kwargs)
    """

    def signal_func(df: pd.DataFrame, i: int,
                    **kwargs) -> Optional[PositionSide]:
        return model.get_signal(df, i, **kwargs)

    signal_func.__name__ = f"{model.arch}_signal"
    return signal_func


def make_lstm_signal(df_train: pd.DataFrame,
                     config: ModelConfig = None) -> Tuple:
    """
    Train an LSTM model on df_train and return (signal_func, model, history).
    """
    model = DeepLearningModel("lstm", config)
    history = model.train(df_train)
    return _make_dl_signal(model), model, history


def make_tcn_signal(df_train: pd.DataFrame,
                    config: ModelConfig = None) -> Tuple:
    """
    Train a TCN model on df_train and return (signal_func, model, history).
    """
    model = DeepLearningModel("tcn", config)
    history = model.train(df_train)
    return _make_dl_signal(model), model, history


def make_tft_signal(df_train: pd.DataFrame,
                    config: ModelConfig = None) -> Tuple:
    """
    Train a TFT model on df_train and return (signal_func, model, history).
    """
    model = DeepLearningModel("tft", config)
    history = model.train(df_train)
    return _make_dl_signal(model), model, history


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  NeuroTrade — Deep Learning Models Quick Test")
    print("=" * 60)

    # generate synthetic data
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    df_test = pd.DataFrame({
        "Open":   close * (1 + np.random.normal(0, 0.002, n)),
        "High":   close * (1 + np.abs(np.random.normal(0, 0.008, n))),
        "Low":    close * (1 - np.abs(np.random.normal(0, 0.008, n))),
        "Close":  close,
        "Volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=idx)

    # add basic indicators
    df_test["SMA_20"] = df_test["Close"].rolling(20).mean()
    df_test["EMA_20"] = df_test["Close"].ewm(span=20).mean()
    df_test["RSI"] = 50 + np.random.normal(0, 15, n)
    df_test["MACD"] = np.random.normal(0, 2, n)
    df_test["ATR"] = (df_test["High"] - df_test["Low"]).rolling(14).mean()
    df_test.dropna(inplace=True)

    cfg = ModelConfig(epochs=5, sequence_length=20, batch_size=16)

    for arch in ["lstm", "tcn", "tft"]:
        print(f"\n{'─' * 60}")
        print(f"  Testing {arch.upper()} …")
        print(f"{'─' * 60}")

        model = DeepLearningModel(arch, cfg)
        history = model.train(df_test, verbose=True)
        preds = model.predict(df_test)
        signal = model.get_signal(df_test)

        print(f"  Predictions shape : {preds.shape}")
        print(f"  Last prediction   : {preds[-1]:.6f}")
        print(f"  Signal            : {signal}")

        if arch == "tft":
            fi = model.get_feature_importance()
            if fi:
                top5 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top-5 features    : {top5}")

    print(f"\n{'=' * 60}")
    print("  All models tested successfully!")
    print(f"{'=' * 60}")
