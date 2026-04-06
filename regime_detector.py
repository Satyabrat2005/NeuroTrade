from __future__ import annotations

import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Optional HMM support — gracefully degrade if hmmlearn is not installed
try:
    from hmmlearn import hmm as _hmm_module
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn(
        "hmmlearn is not installed. HMM-based regime detection will be unavailable. "
        "Install it with: pip install hmmlearn",
        ImportWarning,
        stacklevel=2,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
class Regime(str, Enum):
    """Enumeration of supported market regimes."""
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"   # emitted when a row cannot be classified
# Features consumed from the upstream indicators.py pipeline
REQUIRED_FEATURES: list[str] = ["returns", "rsi", "volatility", "ma_diff"]

# Default model backend
DEFAULT_BACKEND: str = "kmeans"  # or "hmm"

# KMeans default config
KMEANS_N_CLUSTERS: int = 3
RANDOM_STATE: int = 42
# Helper utilities


def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that *df* contains all required feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame produced by indicators.py.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = set(REQUIRED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_FEATURES}"
        )
