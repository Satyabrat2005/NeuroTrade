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
