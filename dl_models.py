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
try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    _SKL = True
except ImportError:
    _SKL = False
    print("[dl_models] scikit-learn not found — pip install scikit-learn")

# CONFIG

class ModelType(Enum):
    LSTM      = "lstm"
    TCN       = "tcn"
    TFT       = "tft"
    ENSEMBLE  = "ensemble"

@dataclass
class DLConfig:
    """Central configuration for all deep-learning models."""
    # Data 
    seq_len:          int   = 60          # lookback window (bars)
    forecast_horizon: int   = 5           # how many bars ahead to predict
    target:           str   = "returns"   # "returns" | "close" | "log_returns"
    feature_set:      str   = "curated"   # "curated" | "full"
    train_ratio:      float = 0.70
    val_ratio:        float = 0.15        # test = 1 - train - val

    #Training 
    batch_size:       int   = 64
    epochs:           int   = 100
    lr:               float = 1e-3
    weight_decay:     float = 1e-4
    patience:         int   = 15          # early stopping
    grad_clip:        float = 1.0
    dropout:          float = 0.2
    model_dir:        str   = "./models"

    # LSTM 
    lstm_hidden:      int   = 128
    lstm_layers:      int   = 2
    lstm_bidir:       bool  = True

    #TCN 
    tcn_channels:     List[int] = field(default_factory=lambda: [64, 128, 128, 64])
    tcn_kernel:       int   = 3
    tcn_dilations:    List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])

    # TFT 
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

#  FEATURE ENGINEER

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

    # public 

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

    # private 

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

#  PYTORCH DATASET

class TimeSeriesDataset(Dataset):
    """Minimal wrapper around numpy arrays for DataLoader."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, T, F)
        self.y = torch.from_numpy(y)   # (N, H)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#  MODEL 1 — BIDIRECTIONAL LSTM WITH ATTENTION

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

#  MODEL 2 — TEMPORAL CONVOLUTIONAL NETWORK (TCN)

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

#  MODEL 3 — TEMPORAL FUSION TRANSFORMER (TFT)

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

#  LOSSES
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

#  TRAINER
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

        # Feature engineering 
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
        val_dl   = DataLoader(TimeSeriesDataset(X_va, y_va),
                              batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        # Build model 
        self.model = self._build_model(model_type, n_features).to(DEVICE)
        n_params   = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n[{model_type.value.upper()}] Parameters: {n_params:,} | Device: {DEVICE}")

        optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=cfg.lr,
                               steps_per_epoch=len(train_dl),
                               epochs=cfg.epochs, pct_start=0.3)
        is_tft = model_type == ModelType.TFT

        # Training loop
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
