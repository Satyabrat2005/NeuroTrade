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
