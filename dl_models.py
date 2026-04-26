import os
import math
import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum

warnings.filterwarnings("ignore")

# ── PyTorch (graceful fallback) ──────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
    _TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH = False
    DEVICE = None
    print("[dl_models] PyTorch not found — pip install torch")

# ── scikit-learn (graceful fallback) ─────────────────────────────────────────
try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    _SKL = True
except ImportError:
    _SKL = False
    print("[dl_models] scikit-learn not found — pip install scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class ModelType(Enum):
    LSTM      = "lstm"
    TCN       = "tcn"
    TFT       = "tft"
    ENSEMBLE  = "ensemble"

@dataclass
class DLConfig:
    """Central configuration for all deep-learning models."""

    # ── Data ──────────────────────────────────────────────────────────────────
    seq_len:          int   = 60          # lookback window (bars)
    forecast_horizon: int   = 5           # how many bars ahead to predict
    target:           str   = "returns"   # "returns" | "close" | "log_returns"
    feature_set:      str   = "curated"   # "curated" | "full"
    train_ratio:      float = 0.70
    val_ratio:        float = 0.15        # test = 1 - train - val

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:       int   = 64
    epochs:           int   = 100
    lr:               float = 1e-3
    weight_decay:     float = 1e-4
    patience:         int   = 15          # early stopping
    grad_clip:        float = 1.0
    dropout:          float = 0.2
    model_dir:        str   = "./models"

    # ── LSTM ──────────────────────────────────────────────────────────────────
    lstm_hidden:      int   = 128
    lstm_layers:      int   = 2
    lstm_bidir:       bool  = True

    # ── TCN ───────────────────────────────────────────────────────────────────
    tcn_channels:     List[int] = field(default_factory=lambda: [64, 128, 128, 64])
    tcn_kernel:       int   = 3
    tcn_dilations:    List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])

    # ── TFT ───────────────────────────────────────────────────────────────────
    tft_d_model:      int   = 64
    tft_n_heads:      int   = 4
    tft_n_layers:     int   = 2
    tft_quantiles:    List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


CONFIG = DLConfig()

# ── Curated feature columns (order matters for reproducibility) ───────────────
CURATED_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Returns", "Log_Returns", "HL_Spread",
    "SMA_20", "EMA_21", "MACD", "MACD_Signal", "MACD_Hist",
    "RSI", "ATR", "ADX", "BB_Upper", "BB_Lower", "BB_Width",
    "OBV", "VWAP", "CCI", "%K", "%D",
    "Realized_Vol_21", "Z_Score_20",
]


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEER
# ══════════════════════════════════════════════════════════════════════════════

class DLFeatureEngineer:
    """
    Transforms an indicator-enriched OHLCV DataFrame into scaled numpy arrays
    ready for PyTorch training.

    Usage
    -----
        eng = DLFeatureEngineer(cfg)
        X, y, scaler = eng.build(df)
    """

    def __init__(self, cfg: DLConfig = None):
        self.cfg    = cfg or DLConfig()
        self.scaler: Optional[RobustScaler] = None
        self.feature_cols: List[str] = []
        self.target_idx:   int = 0        # index of target col in feature matrix

    # ── public ────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame,
              fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns
        -------
        X : (n_samples, seq_len, n_features)
        y : (n_samples, forecast_horizon)
        cols : list of feature column names
        """
        if not _SKL:
            raise ImportError("scikit-learn required — pip install scikit-learn")

        data = self._select_features(df)
        data = self._add_time_features(data, df)
        data = data.dropna()

        if len(data) == 0:
            raise ValueError("Not enough valid data points to train the DL model. Try a longer timeframe.")

        self.feature_cols = list(data.columns)

        # identify target column index (used to invert scale later)
        tgt = self._target_col()
        self.target_idx = self.feature_cols.index(tgt) if tgt in self.feature_cols else 3  # Close fallback

        values = data.values.astype(np.float32)

        if fit_scaler:
            self.scaler = RobustScaler()
            values = self.scaler.fit_transform(values)
        elif self.scaler is not None:
            values = self.scaler.transform(values)

        X, y = self._make_windows(values)
        return X, y, self.feature_cols

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Invert scaling on the target column only."""
        if self.scaler is None:
            return y_scaled
        dummy = np.zeros((y_scaled.shape[0], len(self.feature_cols)), dtype=np.float32)
        dummy[:, self.target_idx] = y_scaled[:, 0]
        return self.scaler.inverse_transform(dummy)[:, self.target_idx]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the already-fitted scaler."""
        data = self._select_features(df)
        data = self._add_time_features(data, df)
        data = data.dropna()
        # align to fitted columns
        for c in self.feature_cols:
            if c not in data.columns:
                data[c] = 0.0
        data = data[self.feature_cols]
        return self.scaler.transform(data.values.astype(np.float32))

    # ── private ───────────────────────────────────────────────────────────────

    def _target_col(self) -> str:
        mapping = {"returns": "Returns", "log_returns": "Log_Returns", "close": "Close"}
        return mapping.get(self.cfg.target, "Returns")

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.feature_set == "curated":
            cols = [c for c in CURATED_FEATURES if c in df.columns]
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in num_cols if c != "Ticker"]
        return df[cols].copy()

    @staticmethod
    def _add_time_features(data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        data = data.copy()
        data["DayOfWeek"]   = idx.dayofweek.astype(np.float32) / 4.0
        data["Month"]       = (idx.month.astype(np.float32) - 1) / 11.0
        data["Quarter"]     = (idx.quarter.astype(np.float32) - 1) / 3.0
        data["DayOfYear"]   = idx.dayofyear.astype(np.float32) / 365.0
        return data

    def _make_windows(self,
                      values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding window approach — no data leakage."""
        cfg  = self.cfg
        T    = cfg.seq_len
        H    = cfg.forecast_horizon
        tidx = self.target_idx
        n    = len(values)
        X, y = [], []

        for i in range(T, n - H + 1):
            X.append(values[i - T: i])          # (seq_len, n_features)
            y.append(values[i: i + H, tidx])    # (forecast_horizon,)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    """Minimal wrapper around numpy arrays for DataLoader."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, T, F)
        self.y = torch.from_numpy(y)   # (N, H)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — BIDIRECTIONAL LSTM WITH ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class LSTMAttention(nn.Module if _TORCH else object):
    """
    Bidirectional multi-layer LSTM with a Bahdanau-style additive attention
    mechanism over the temporal dimension, followed by a feedforward head.

    Architecture
    ────────────
    Input  (B, T, F)
    │
    ├─ BiLSTM x n_layers  →  (B, T, 2H)
    ├─ Attention  →  context  (B, 2H)
    ├─ LayerNorm + Dropout
    ├─ FC 2H → H → forecast_horizon
    └─ Output  (B, horizon)
    """

    def __init__(self, n_features: int, cfg: DLConfig):
        super().__init__()
        self.cfg = cfg
        H        = cfg.lstm_hidden
        bidir    = cfg.lstm_bidir
        dirs     = 2 if bidir else 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=H,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=bidir,
        )
        self.attn_w = nn.Linear(dirs * H, dirs * H, bias=False)
        self.attn_v = nn.Linear(dirs * H, 1, bias=False)

        self.norm    = nn.LayerNorm(dirs * H)
        self.drop    = nn.Dropout(cfg.dropout)
        self.head    = nn.Sequential(
            nn.Linear(dirs * H, H),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H, cfg.forecast_horizon),
        )

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        out, _ = self.lstm(x)                       # (B, T, 2H)
        score  = self.attn_v(torch.tanh(self.attn_w(out)))   # (B, T, 1)
        weight = F.softmax(score, dim=1)            # (B, T, 1)
        context = (weight * out).sum(dim=1)         # (B, 2H)
        context = self.drop(self.norm(context))
        return self.head(context), weight.squeeze(-1)   # (B, H), (B, T)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ══════════════════════════════════════════════════════════════════════════════

class _CausalConv1d(nn.Module if _TORCH else object):
    """Causal (left-padded) dilated convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=self.padding)
        )

    def forward(self, x):
        return self.conv(x)[:, :, :-self.padding] if self.padding else self.conv(x)


class _TCNBlock(nn.Module if _TORCH else object):
    """Single TCN residual block with two dilated causal convolutions."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch,  out_ch, kernel, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x) + res
        # LayerNorm over channel dim: transpose → norm → transpose
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class TCNModel(nn.Module if _TORCH else object):
    """
    Temporal Convolutional Network for time-series forecasting.

    Architecture
    ────────────
    Input (B, T, F)  →  transpose  →  (B, F, T)
    │
    ├─ TCNBlock(dilation=1)
    ├─ TCNBlock(dilation=2)
    ├─ TCNBlock(dilation=4)
    ├─ TCNBlock(dilation=8)
    ├─ TCNBlock(dilation=16)
    │
    ├─ Global Average Pool  →  (B, C_last)
    └─ FC head  →  (B, horizon)

    Receptive field = 1 + 2 * (kernel-1) * (2^n_layers - 1)
    For kernel=3, 5 dilations: RF = 97 bars  ✓
    """

    def __init__(self, n_features: int, cfg: DLConfig):
        super().__init__()
        self.cfg = cfg

        # build channel list: input → each TCN level
        channels  = cfg.tcn_channels
        dilations = cfg.tcn_dilations
        kernel    = cfg.tcn_kernel

        layers = []
        in_ch  = n_features
        for i, (out_ch, dil) in enumerate(zip(channels, dilations)):
            layers.append(_TCNBlock(in_ch, out_ch, kernel, dil, cfg.dropout))
            in_ch = out_ch
        # additional dilated layers if more dilations than channels
        for dil in dilations[len(channels):]:
            layers.append(_TCNBlock(in_ch, in_ch, kernel, dil, cfg.dropout))

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(in_ch // 2, cfg.forecast_horizon),
        )

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", None]:
        x = x.transpose(1, 2)           # (B, F, T)
        x = self.network(x)             # (B, C, T)
        x = x.mean(dim=-1)              # (B, C)   — global avg pool
        return self.head(x), None       # (B, horizon), no attention


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 3 — TEMPORAL FUSION TRANSFORMER (TFT)
# ══════════════════════════════════════════════════════════════════════════════

class _GRN(nn.Module if _TORCH else object):
    """Gated Residual Network — core building block of TFT."""

    def __init__(self, d: int, d_out: int = None, dropout: float = 0.1):
        super().__init__()
        d_out = d_out or d
        self.fc1  = nn.Linear(d, d)
        self.fc2  = nn.Linear(d, d_out)
        self.gate = nn.Linear(d, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d, d_out) if d != d_out else nn.Identity()

    def forward(self, x):
        h   = F.gelu(self.fc1(x))
        h   = self.drop(h)
        eta = self.fc2(h)
        sig = torch.sigmoid(self.gate(h))
        out = self.norm(eta * sig + self.skip(x))
        return out


class _VSN(nn.Module if _TORCH else object):
    """Variable Selection Network — learns per-feature importance weights."""

    def __init__(self, n_features: int, d: int, dropout: float):
        super().__init__()
        self.feature_grns = nn.ModuleList([_GRN(1, d, dropout) for _ in range(n_features)])
        self.select_grn   = _GRN(n_features, n_features, dropout)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        # x: (B, T, F)
        B, T, F = x.shape
        feat_tensors = [self.feature_grns[i](x[..., i:i+1]) for i in range(F)]
        feat_stack   = torch.stack(feat_tensors, dim=-2)  # (B, T, F, d)
        weights      = F.softmax(
            self.select_grn(x.reshape(B * T, F)).reshape(B, T, F), dim=-1
        )                                                  # (B, T, F)
        out = (feat_stack * weights.unsqueeze(-1)).sum(dim=-2)  # (B, T, d)
        return out, weights


class TFTModel(nn.Module if _TORCH else object):
    """
    Simplified Temporal Fusion Transformer (Bryan Lim et al., 2021).

    Architecture
    ────────────
    Input (B, T, F)
    │
    ├─ Variable Selection Network (VSN)    →  (B, T, d_model)
    ├─ LSTM encoder                        →  (B, T, d_model)
    ├─ Gated Add & Norm                    →  (B, T, d_model)
    ├─ Multi-Head Self-Attention + GRN     →  (B, T, d_model)
    ├─ Gated Add & Norm                    →  (B, T, d_model)
    ├─ Position-wise Feed-Forward + GRN
    ├─ Pool last H positions or global
    └─ Quantile output head (3 quantiles)  →  (B, H, 3)  or point (B, H)
    """

    def __init__(self, n_features: int, cfg: DLConfig):
        super().__init__()
        self.cfg  = cfg
        d         = cfg.tft_d_model
        nh        = cfg.tft_n_heads
        drop      = cfg.dropout
        H         = cfg.forecast_horizon
        nq        = len(cfg.tft_quantiles)

        self.vsn  = _VSN(n_features, d, drop)
        self.lstm_enc = nn.LSTM(d, d, cfg.tft_n_layers,
                                batch_first=True, dropout=drop if cfg.tft_n_layers > 1 else 0.0)
        self.gate_enc = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.norm_enc = nn.LayerNorm(d)

        attn_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nh, dim_feedforward=d * 4,
            dropout=drop, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(attn_layer, num_layers=cfg.tft_n_layers)
        self.grn_post = _GRN(d, d, drop)
        self.norm_post = nn.LayerNorm(d)

        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d, H * nq),
        )
        self.nq = nq
        self.H  = H

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        vsn_out, vsn_w = self.vsn(x)              # (B, T, d), (B, T, F)
        lstm_out, _    = self.lstm_enc(vsn_out)   # (B, T, d)
        gated  = self.gate_enc(lstm_out)
        enc    = self.norm_enc(lstm_out * gated + vsn_out)

        attn_out = self.attn(enc)                 # (B, T, d)
        grn_out  = self.grn_post(attn_out)
        out      = self.norm_post(grn_out + enc)  # (B, T, d)

        context  = out.mean(dim=1)                # (B, d)  — global mean pool
        preds    = self.head(context)             # (B, H * nq)
        preds    = preds.view(-1, self.H, self.nq)  # (B, H, nq)
        return preds, vsn_w                        # (B, H, nq), (B, T, F)


# ══════════════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def quantile_loss(preds: "torch.Tensor",
                  target: "torch.Tensor",
                  quantiles: List[float]) -> "torch.Tensor":
    """
    Pinball (quantile) loss for TFT.
    preds  : (B, H, nq)
    target : (B, H)
    """
    target = target.unsqueeze(-1).expand_as(preds)
    q      = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype)
    errors = target - preds
    loss   = torch.max((q - 1) * errors, q * errors)
    return loss.mean()


def directional_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """% of time the sign of predicted change matches actual change."""
    if len(pred) < 2:
        return 0.0
    pred_dir   = np.sign(np.diff(pred,   axis=0))
    actual_dir = np.sign(np.diff(actual, axis=0))
    return float(np.mean(pred_dir == actual_dir)) * 100


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    model_type:   str
    train_losses: List[float]
    val_losses:   List[float]
    best_epoch:   int
    metrics:      Dict
    model_path:   str
    duration_sec: float
    n_params:     int


class DLTrainer:
    """
    Unified trainer for LSTM, TCN, and TFT models.

    Usage
    -----
        trainer = DLTrainer(cfg)
        result  = trainer.train(df, ModelType.LSTM)
        preds   = trainer.predict(df)
        signal  = DLSignalGenerator.from_result(result).signal(df, i)
    """

    def __init__(self, cfg: DLConfig = None):
        self.cfg       = cfg or DLConfig()
        self.model:    Optional[nn.Module]   = None
        self.eng:      DLFeatureEngineer     = DLFeatureEngineer(cfg)
        self.model_type: Optional[ModelType] = None
        self._best_state = None

    # ── public ────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, model_type: ModelType,
              progress_cb: Optional[Callable] = None) -> TrainingResult:
        """
        Full train–val–test pipeline.

        Parameters
        ----------
        df          : indicator-enriched OHLCV DataFrame
        model_type  : LSTM | TCN | TFT
        progress_cb : optional callback(epoch, n_epochs, train_loss, val_loss)
                      used by Streamlit progress bars

        Returns
        -------
        TrainingResult with metrics and saved model path
        """
        if not _TORCH:
            raise ImportError("PyTorch required — pip install torch")

        t0 = time.time()
        self.model_type = model_type
        cfg = self.cfg

        # ── 1. Feature engineering ────────────────────────────────────────────
        X, y, cols = self.eng.build(df, fit_scaler=True)
        n_features  = X.shape[2]
        n           = len(X)

        n_train = int(n * cfg.train_ratio)
        n_val   = int(n * cfg.val_ratio)

        X_tr, y_tr = X[:n_train],         y[:n_train]
        X_va, y_va = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_te, y_te = X[n_train+n_val:],   y[n_train+n_val:]

        train_dl = DataLoader(TimeSeriesDataset(X_tr, y_tr),
                              batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
        if len(train_dl) == 0:
            train_dl = DataLoader(TimeSeriesDataset(X_tr, y_tr),
                                  batch_size=cfg.batch_size, shuffle=True,  drop_last=False)
        if len(train_dl) == 0:
            raise ValueError("Not enough data to train. Please decrease sequence length or provide more historical data.")

        val_dl   = DataLoader(TimeSeriesDataset(X_va, y_va),
                              batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        # ── 2. Build model ────────────────────────────────────────────────────
        self.model = self._build_model(model_type, n_features).to(DEVICE)
        n_params   = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n[{model_type.value.upper()}] Parameters: {n_params:,} | Device: {DEVICE}")

        optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=cfg.lr,
                               steps_per_epoch=len(train_dl),
                               epochs=cfg.epochs, pct_start=0.3)
        is_tft = model_type == ModelType.TFT

        # ── 3. Training loop ──────────────────────────────────────────────────
        train_losses, val_losses = [], []
        best_val, patience_ctr   = float("inf"), 0
        best_epoch               = 0

        for epoch in range(1, cfg.epochs + 1):
            tr_loss = self._train_epoch(train_dl, optimizer, scheduler, is_tft)
            vl_loss = self._eval_epoch(val_dl, is_tft)
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            if progress_cb:
                progress_cb(epoch, cfg.epochs, tr_loss, vl_loss)

            if vl_loss < best_val:
                best_val     = vl_loss
                best_epoch   = epoch
                patience_ctr = 0
                self._best_state = {k: v.cpu().clone()
                                    for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.patience:
                    print(f"  Early stop at epoch {epoch} (best={best_epoch})")
                    break

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>4}/{cfg.epochs} | "
                      f"train={tr_loss:.5f}  val={vl_loss:.5f}  "
                      f"best_val={best_val:.5f}")

        # ── 4. Restore best & evaluate ────────────────────────────────────────
        if self._best_state:
            self.model.load_state_dict({k: v.to(DEVICE) for k, v in self._best_state.items()})

        metrics = self._evaluate(X_te, y_te, is_tft)

        # ── 5. Save model ─────────────────────────────────────────────────────
        path = self._save(model_type, n_features)

        return TrainingResult(
            model_type   = model_type.value,
            train_losses = train_losses,
            val_losses   = val_losses,
            best_epoch   = best_epoch,
            metrics      = metrics,
            model_path   = path,
            duration_sec = time.time() - t0,
            n_params     = n_params,
        )

    def predict(self, df: pd.DataFrame,
                return_attention: bool = False) -> Dict:
        """
        Generate multi-step forecasts for the last `seq_len` bars.

        Returns
        -------
        dict with keys:
            'forecast'  : (forecast_horizon,) predicted values (target units)
            'quantiles' : (forecast_horizon, 3) or None — TFT only
            'attention' : attention weights or None
            'dates'     : future date index
        """
        if self.model is None:
            raise RuntimeError("Call train() or load() first")

        self.model.eval()
        cfg     = self.cfg
        is_tft  = self.model_type == ModelType.TFT

        # get last seq_len rows
        arr    = self.eng.transform(df)
        if len(arr) < cfg.seq_len:
            raise ValueError(f"Need at least {cfg.seq_len} rows, got {len(arr)}")
        inp    = torch.from_numpy(arr[-cfg.seq_len:]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out, attn = self.model(inp)

        if is_tft:
            preds_q = out[0].cpu().numpy()     # (H, nq)
            preds   = preds_q[:, 1]            # median quantile
            quants  = preds_q
        else:
            preds  = out[0].cpu().numpy()      # (H,)
            quants = None
            if attn is not None:
                attn = attn[0].cpu().numpy()   # (T,)

        # generate future dates
        last_date = df.index[-1]
        freq      = pd.infer_freq(df.index[-20:]) or "B"
        fut_dates = pd.date_range(last_date, periods=cfg.forecast_horizon + 1, freq=freq)[1:]

        return {
            "forecast":  preds,
            "quantiles": quants,
            "attention": attn if return_attention else None,
            "dates":     fut_dates,
        }

    def load(self, path: str, n_features: int,
             model_type: Optional[ModelType] = None) -> None:
        """Load a previously saved model."""
        if not _TORCH:
            raise ImportError("PyTorch required")
        if model_type is None:
            model_type = self.model_type or ModelType.LSTM
        self.model_type = model_type
        self.model = self._build_model(model_type, n_features).to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[load] {model_type.value} loaded from {path}")

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_model(self, model_type: ModelType, n_features: int) -> "nn.Module":
        cfg = self.cfg
        if model_type == ModelType.LSTM:
            return LSTMAttention(n_features, cfg)
        elif model_type == ModelType.TCN:
            return TCNModel(n_features, cfg)
        elif model_type == ModelType.TFT:
            return TFTModel(n_features, cfg)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _train_epoch(self, dl, optimizer, scheduler, is_tft: bool) -> float:
        self.model.train()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out, _ = self.model(xb)
            if is_tft:
                loss = quantile_loss(out, yb, self.cfg.tft_quantiles)
            else:
                loss = F.mse_loss(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(xb)
        return total / len(dl.dataset)

    @torch.no_grad()
    def _eval_epoch(self, dl, is_tft: bool) -> float:
        self.model.eval()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out, _ = self.model(xb)
            if is_tft:
                loss = quantile_loss(out, yb, self.cfg.tft_quantiles)
            else:
                loss = F.mse_loss(out, yb)
            total += loss.item() * len(xb)
        return total / len(dl.dataset)

    @torch.no_grad()
    def _evaluate(self, X_te: np.ndarray, y_te: np.ndarray,
                  is_tft: bool) -> Dict:
        self.model.eval()
        ds    = TimeSeriesDataset(X_te, y_te)
        dl    = DataLoader(ds, batch_size=128, shuffle=False)
        preds, actuals = [], []

        for xb, yb in dl:
            xb = xb.to(DEVICE)
            out, _ = self.model(xb)
            if is_tft:
                out = out[:, :, 1]   # median
            preds.append(out.cpu().numpy())
            actuals.append(yb.numpy())

        preds   = np.concatenate(preds,   axis=0)   # (N, H)
        actuals = np.concatenate(actuals, axis=0)   # (N, H)

        flat_p = preds.flatten()
        flat_a = actuals.flatten()
        mse    = mean_squared_error(flat_a, flat_p)
        mae    = mean_absolute_error(flat_a, flat_p)
        r2     = r2_score(flat_a, flat_p)
        mape   = float(np.mean(np.abs((flat_a - flat_p) / (np.abs(flat_a) + 1e-8)))) * 100
        da     = directional_accuracy(flat_p, flat_a)

        metrics = {
            "mse":  round(mse,  6),
            "rmse": round(float(np.sqrt(mse)), 6),
            "mae":  round(mae,  6),
            "mape": round(mape, 4),
            "r2":   round(r2,   4),
            "directional_accuracy": round(da, 2),
            "n_test_samples": len(preds),
        }
        print(f"\n-- Test Metrics ({self.model_type.value.upper()}) ------------------")
        for k, v in metrics.items():
            print(f"  {k:<26}: {v}")
        return metrics

    def _save(self, model_type: ModelType, n_features: int) -> str:
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        fname = os.path.join(self.cfg.model_dir,
                             f"{model_type.value}_f{n_features}_h{self.cfg.forecast_horizon}.pt")
        torch.save(self.model.state_dict(), fname)
        print(f"\n[save] Model saved -> {fname}")
        return fname


# ══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleTrainer:
    """
    Trains all three base models and combines their predictions via
    weighted averaging (weights learned on the validation set).

    Usage
    -----
        ens = EnsembleTrainer(cfg)
        results = ens.train_all(df, progress_cb=None)
        preds   = ens.predict(df)
    """

    def __init__(self, cfg: DLConfig = None):
        self.cfg      = cfg or DLConfig()
        self.trainers = {
            ModelType.LSTM: DLTrainer(cfg),
            ModelType.TCN:  DLTrainer(cfg),
            ModelType.TFT:  DLTrainer(cfg),
        }
        self.weights  = {ModelType.LSTM: 1/3, ModelType.TCN: 1/3, ModelType.TFT: 1/3}
        self.results: Dict[ModelType, TrainingResult] = {}

    def train_all(self, df: pd.DataFrame,
                  progress_cb: Optional[Callable] = None) -> Dict:
        """Train all three models sequentially."""
        all_results = {}
        for mt, trainer in self.trainers.items():
            print(f"\n{'='*60}")
            print(f"  Training {mt.value.upper()}")
            print(f"{'='*60}")
            result = trainer.train(df, mt, progress_cb=progress_cb)
            self.results[mt] = result
            all_results[mt.value] = result
        self._learn_weights(df)
        return all_results

    def predict(self, df: pd.DataFrame) -> Dict:
        """Ensemble prediction: weighted average of all three models."""
        all_preds = {}
        for mt, trainer in self.trainers.items():
            try:
                p = trainer.predict(df)
                all_preds[mt] = p
            except Exception as e:
                print(f"[Ensemble] {mt.value} predict failed: {e}")

        if not all_preds:
            raise RuntimeError("No models could predict")

        # weighted combination
        ref    = next(iter(all_preds.values()))
        total  = sum(self.weights[mt] for mt in all_preds)
        combo  = np.zeros_like(ref["forecast"])
        for mt, p in all_preds.items():
            combo += (self.weights[mt] / total) * p["forecast"]

        return {
            "forecast":   combo,
            "quantiles":  None,
            "attention":  None,
            "per_model":  {mt.value: p["forecast"] for mt, p in all_preds.items()},
            "weights":    {mt.value: self.weights[mt] for mt in all_preds},
            "dates":      ref["dates"],
        }

    def _learn_weights(self, df: pd.DataFrame) -> None:
        """Weight each model by its validation-set directional accuracy."""
        da_scores = {}
        for mt, res in self.results.items():
            da = res.metrics.get("directional_accuracy", 50.0)
            da_scores[mt] = max(da, 0.01)
        total = sum(da_scores.values())
        self.weights = {mt: s / total for mt, s in da_scores.items()}
        print("\n[Ensemble] Learned weights:")
        for mt, w in self.weights.items():
            print(f"  {mt.value:<8}: {w:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class DLSignalGenerator:
    """
    Converts DL model forecasts into backtester-compatible signals.

    Integrates with the existing Backtester via a dynamic signal function.

    Signal logic (return-based target)
    ───────────────────────────────────
    - Predicted cumulative return over horizon > +threshold  → LONG
    - Predicted cumulative return over horizon < -threshold  → SHORT
    - Otherwise                                              → FLAT (hold)
    """

    def __init__(self, trainer: "DLTrainer | EnsembleTrainer",
                 threshold: float = 0.005,
                 use_quantile_filter: bool = True):
        self.trainer    = trainer
        self.threshold  = threshold
        self.use_qf     = use_quantile_filter
        self._last_pred: Optional[Dict] = None
        self._last_bar:  int = -1

    def make_signal_func(self, df: pd.DataFrame) -> Callable:
        """
        Returns a callable compatible with Backtester.run(df, signal_func).

        The function caches predictions and re-infers every forecast_horizon bars
        to avoid re-running the model on every single bar.
        """
        cfg    = self.trainer.cfg if hasattr(self.trainer, "cfg") else DLConfig()
        H      = cfg.forecast_horizon
        engine = self

        def signal_func(df_bt: pd.DataFrame, i: int):
            # lazy import to avoid circular
            try:
                from backtester import PositionSide
            except ImportError:
                return None

            if i < cfg.seq_len:
                return None

            # re-predict every H bars
            if engine._last_bar == -1 or (i - engine._last_bar) >= H:
                try:
                    sub = df_bt.iloc[:i + 1]
                    engine._last_pred = engine.trainer.predict(sub)
                    engine._last_bar  = i
                except Exception:
                    return None

            pred = engine._last_pred
            if pred is None:
                return None

            fcast = pred["forecast"]
            cum_ret = float(np.sum(fcast))

            # TFT quantile filter: only trade if downside q10 confirms direction
            if engine.use_qf and pred.get("quantiles") is not None:
                q_lo = float(pred["quantiles"][:, 0].sum())
                q_hi = float(pred["quantiles"][:, 2].sum())
                if cum_ret > engine.threshold and q_lo < 0:
                    return None       # uncertain — skip long
                if cum_ret < -engine.threshold and q_hi > 0:
                    return None       # uncertain — skip short

            if cum_ret > engine.threshold:
                return PositionSide.LONG
            elif cum_ret < -engine.threshold:
                return PositionSide.SHORT
            return PositionSide.FLAT

        return signal_func


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY  — Streamlit helper
# ══════════════════════════════════════════════════════════════════════════════

class DLModelRegistry:
    """
    Keeps track of trained models within a Streamlit session.
    Store in st.session_state["dl_registry"].
    """

    def __init__(self):
        self.trainers: Dict[str, "DLTrainer | EnsembleTrainer"] = {}
        self.results:  Dict[str, TrainingResult] = {}

    def register(self, key: str,
                 trainer: "DLTrainer | EnsembleTrainer",
                 result: TrainingResult):
        self.trainers[key] = trainer
        self.results[key]  = result

    def get(self, key: str) -> Optional["DLTrainer | EnsembleTrainer"]:
        return self.trainers.get(key)

    def list_models(self) -> List[str]:
        return list(self.trainers.keys())

    def is_trained(self, key: str) -> bool:
        return key in self.trainers


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTICS  — standalone test
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_df(n: int = 500) -> pd.DataFrame:
    """Generate a minimal synthetic OHLCV + indicator DataFrame for testing."""
    np.random.seed(42)
    dates  = pd.date_range("2021-01-01", periods=n, freq="B")
    close  = 1000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    df = pd.DataFrame({
        "Open":         close * (1 + np.random.normal(0, 0.002, n)),
        "High":         close * (1 + np.abs(np.random.normal(0, 0.008, n))),
        "Low":          close * (1 - np.abs(np.random.normal(0, 0.008, n))),
        "Close":        close,
        "Volume":       np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=dates)
    ret = df["Close"].pct_change()
    df["Returns"]       = ret
    df["Log_Returns"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Spread"]     = (df["High"] - df["Low"]) / df["Close"]
    df["SMA_20"]        = df["Close"].rolling(20).mean()
    df["EMA_21"]        = df["Close"].ewm(span=21).mean()
    e12 = df["Close"].ewm(span=12).mean()
    e26 = df["Close"].ewm(span=26).mean()
    df["MACD"]          = e12 - e26
    df["MACD_Signal"]   = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"]     = df["MACD"] - df["MACD_Signal"]
    d = df["Close"].diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["RSI"]           = 100 - (100 / (1 + gain / loss))
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"]           = tr.rolling(14).mean()
    up = df["High"].diff().clip(lower=0)
    dn = (-df["Low"].diff()).clip(lower=0)
    a14 = tr.rolling(14).mean()
    dip = 100 * up.rolling(14).mean() / a14
    din = 100 * dn.rolling(14).mean() / a14
    df["DI_Pos"]        = dip
    df["DI_Neg"]        = din
    df["ADX"]           = (100*(dip-din).abs()/(dip+din+1e-9)).rolling(14).mean()
    mid = df["Close"].rolling(20).mean(); std = df["Close"].rolling(20).std()
    df["BB_Upper"]      = mid + 2 * std
    df["BB_Lower"]      = mid - 2 * std
    df["BB_Width"]      = (df["BB_Upper"] - df["BB_Lower"]) / mid
    direction           = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"]           = (direction * df["Volume"]).cumsum()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"]          = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    ma = tp.rolling(20).mean(); md_ = (tp - ma).abs().rolling(20).mean()
    df["%K"]            = 100 * (df["Close"] - df["Low"].rolling(14).min()) / \
                          (df["High"].rolling(14).max() - df["Low"].rolling(14).min() + 1e-9)
    df["%D"]            = df["%K"].rolling(3).mean()
    df["CCI"]           = (tp - ma) / (0.015 * md_ + 1e-9)
    df["Realized_Vol_21"] = df["Log_Returns"].rolling(21).std() * np.sqrt(252)
    df["Z_Score_20"]    = (df["Close"] - df["Close"].rolling(20).mean()) / \
                          (df["Close"].rolling(20).std() + 1e-9)
    return df.dropna()


if __name__ == "__main__":
    if not _TORCH:
        print("Install PyTorch first:  pip install torch")
        raise SystemExit(1)
    if not _SKL:
        print("Install scikit-learn:   pip install scikit-learn")
        raise SystemExit(1)

    print("\n" + "=" * 65)
    print("  dl_models.py — Deep Learning Engine Self-Test")
    print("=" * 65)

    df = _generate_synthetic_df(600)
    print(f"\n  Synthetic df: {len(df)} rows × {len(df.columns)} cols")
    print(f"  Date range  : {df.index[0].date()} -> {df.index[-1].date()}")

    cfg = DLConfig(epochs=30, seq_len=40, forecast_horizon=3, batch_size=32, patience=8)

    for mt in [ModelType.LSTM, ModelType.TCN, ModelType.TFT]:
        print(f"\n{'─'*65}")
        print(f"  Testing {mt.value.upper()}")
        print(f"{'─'*65}")
        trainer = DLTrainer(cfg)
        result  = trainer.train(df, mt)
        pred    = trainer.predict(df)
        print(f"\n  Forecast (next {cfg.forecast_horizon} bars): {pred['forecast'].round(5)}")
        print(f"  Future dates: {list(pred['dates'].strftime('%Y-%m-%d'))}")
        if pred.get("quantiles") is not None:
            print(f"  Quantiles q10/q50/q90 (bar 1): {pred['quantiles'][0].round(5)}")
        if pred.get("attention") is not None:
            print(f"  Attention shape: {pred['attention'].shape}")
        print(f"\n  Training time: {result.duration_sec:.1f}s  |  "
              f"Best epoch: {result.best_epoch}  |  "
              f"Params: {result.n_params:,}")

    print("\n" + "=" * 65)
    print("  Testing Ensemble")
    print("=" * 65)
    ens     = EnsembleTrainer(cfg)
    all_res = ens.train_all(df)
    ep      = ens.predict(df)
    print(f"\n  Ensemble forecast: {ep['forecast'].round(5)}")
    print(f"  Per-model forecasts:")
    for k, v in ep["per_model"].items():
        print(f"    {k:<8}: {v.round(5)}")
    print(f"  Weights: {ep['weights']}")
    print("\n  OK  All models passed self-test.\n")
