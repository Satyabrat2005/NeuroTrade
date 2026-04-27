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

# Mapping from capitalized indicator columns to regime detector features
COLUMN_MAP = {
    "Returns": "returns",
    "RSI": "rsi",
    "Realized_Vol_21": "volatility",
    "ATR": "volatility",
}


def prepare_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an indicator-enriched OHLCV DataFrame into the feature
    format expected by RegimeDetector.

    Maps: Returns → returns, RSI → rsi,
          Realized_Vol_21 or ATR → volatility,
          SMA_20 diff → ma_diff.
    """
    out = pd.DataFrame(index=df.index)
    if "Returns" in df.columns:
        out["returns"] = df["Returns"]
    elif "Close" in df.columns:
        out["returns"] = df["Close"].pct_change()
    else:
        out["returns"] = 0.0

    out["rsi"] = df["RSI"] if "RSI" in df.columns else 50.0

    if "Realized_Vol_21" in df.columns:
        out["volatility"] = df["Realized_Vol_21"]
    elif "ATR" in df.columns:
        out["volatility"] = df["ATR"] / (df["Close"] + 1e-9)
    else:
        out["volatility"] = out["returns"].rolling(21).std()

    if "SMA_20" in df.columns:
        out["ma_diff"] = (df["Close"] - df["SMA_20"]) / (df["SMA_20"] + 1e-9)
    elif "EMA_20" in df.columns:
        out["ma_diff"] = (df["Close"] - df["EMA_20"]) / (df["EMA_20"] + 1e-9)
    else:
        out["ma_diff"] = 0.0

    return out

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
def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values without introducing look-ahead bias.

    Strategy:
      - Forward-fill (propagates last known value forward in time)
      - Backward-fill  (handles NaNs at the very start of the series)
      - Any remaining NaNs are filled with column medians computed on
        the *already-available* data (no future information is used).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (may contain NaNs from indicator warm-up).

    Returns
    -------
    pd.DataFrame
        DataFrame with NaNs resolved.
    """
    df = df.copy()
    null_count = df[REQUIRED_FEATURES].isnull().sum().sum()
    if null_count:
        logger.info("Imputing %d missing value(s) across feature columns.", null_count)

    # Time-safe fill: forward then backward
    df[REQUIRED_FEATURES] = (
        df[REQUIRED_FEATURES]
        .ffill()
        .bfill()
    )

    # Fallback: column medians for any surviving NaNs (e.g. all-NaN column)
    for col in REQUIRED_FEATURES:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.warning(
                "Column '%s' still contained NaNs after ffill/bfill; "
                "filled with median (%f).",
                col,
                median_val,
            )

    return df


def _map_clusters_to_regimes(
    labels: np.ndarray,
    returns: pd.Series,
) -> dict[int, Regime]:
    """
    Assign regime labels to raw cluster IDs based on mean returns.

    Rules
    -----
    - Cluster with the **highest** mean return  → BULL
    - Cluster with the **lowest**  mean return  → BEAR
    - Remaining cluster                         → VOLATILE

    Parameters
    ----------
    labels : np.ndarray
        Integer cluster assignments aligned with *returns*.
    returns : pd.Series
        The ``returns`` column from the feature DataFrame.

    Returns
    -------
    dict[int, Regime]
        Mapping from cluster ID to Regime enum value.
    """
    unique_clusters = np.unique(labels)
    mean_returns: dict[int, float] = {
        c: returns.values[labels == c].mean() for c in unique_clusters
    }

    sorted_clusters = sorted(mean_returns, key=mean_returns.__getitem__)

    # sorted_clusters: [lowest_return, ..., highest_return]
    bear_cluster = sorted_clusters[0]
    bull_cluster = sorted_clusters[-1]
    volatile_clusters = [c for c in sorted_clusters if c not in (bear_cluster, bull_cluster)]

    mapping: dict[int, Regime] = {
        bear_cluster: Regime.BEAR,
        bull_cluster: Regime.BULL,
    }
    for c in volatile_clusters:
        mapping[c] = Regime.VOLATILE

    logger.info(
        "Cluster → Regime mapping: %s  |  Mean returns: %s",
        {k: v.value for k, v in mapping.items()},
        {k: f"{v:.6f}" for k, v in mean_returns.items()},
    )
    return mapping


# KMeans backend


class _KMeansBackend:
    """
    Internal KMeans-based clustering backend.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (default: 3).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_clusters: int = KMEANS_N_CLUSTERS, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        self._cluster_regime_map: dict[int, Regime] = {}
        self._is_fitted: bool = False

   
    def fit(self, X: pd.DataFrame, returns: pd.Series) -> "_KMeansBackend":
        """
        Fit the scaler and KMeans model, then derive the regime mapping.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (already imputed, shape [n_samples, n_features]).
        returns : pd.Series
            The ``returns`` column used for cluster labelling.

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        labels: np.ndarray = self.model.labels_
        self._cluster_regime_map = _map_clusters_to_regimes(labels, returns)
        self._is_fitted = True
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Assign regimes to new observations using the fitted model.

        The scaler uses **training statistics only** — no data leakage.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (already imputed).

        Returns
        -------
        np.ndarray of str
            Regime string labels aligned with the input rows.
        """
        if not self._is_fitted:
            raise RuntimeError("Backend must be fitted before calling predict().")

        # Transform with training-time statistics (no leakage)
        X_scaled = self.scaler.transform(X)
        cluster_ids: np.ndarray = self.model.predict(X_scaled)
        regimes = np.array(
            [self._cluster_regime_map.get(c, Regime.UNKNOWN).value for c in cluster_ids]
        )
        return regimes


# HMM backend (optional)


class _HMMBackend:
    """
    Internal Gaussian HMM-based backend (requires hmmlearn).

    Parameters
    ----------
    n_components : int
        Number of hidden states (regimes).
    random_state : int
        Seed for reproducibility.
    covariance_type : str
        HMM covariance type ('full', 'diag', 'tied', 'spherical').
    n_iter : int
        Maximum EM iterations.
    """

    def __init__(
        self,
        n_components: int = KMEANS_N_CLUSTERS,
        random_state: int = RANDOM_STATE,
        covariance_type: str = "full",
        n_iter: int = 100,
    ):
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HMM-based regime detection. "
                "Install it with: pip install hmmlearn"
            )
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = _hmm_module.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self._state_regime_map: dict[int, Regime] = {}
        self._is_fitted: bool = False

  
    def fit(self, X: pd.DataFrame, returns: pd.Series) -> "_HMMBackend":
        """
        Fit the HMM and derive the state → regime mapping.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (already imputed).
        returns : pd.Series
            Used to assign semantic regime labels to states.

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        states: np.ndarray = self.model.predict(X_scaled)
        self._state_regime_map = _map_clusters_to_regimes(states, returns)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Decode the most-likely hidden state sequence for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (already imputed).

        Returns
        -------
        np.ndarray of str
            Regime string labels aligned with the input rows.
        """
        if not self._is_fitted:
            raise RuntimeError("Backend must be fitted before calling predict().")

        X_scaled = self.scaler.transform(X)
        states: np.ndarray = self.model.predict(X_scaled)
        regimes = np.array(
            [self._state_regime_map.get(s, Regime.UNKNOWN).value for s in states]
        )
        return regimes



# Public API: RegimeDetector


class RegimeDetector:
    """
    Production-grade market regime detector.

    Detects one of three regimes for each bar in a financial time series:
    - **BULL**     — trending upward (highest mean returns)
    - **BEAR**     — trending downward (lowest mean returns)
    - **VOLATILE** — high uncertainty / sideways movement

    Parameters
    ----------
    backend : str
        Model backend: ``"kmeans"`` (default) or ``"hmm"``.
    n_clusters : int
        Number of regimes / clusters (default: 3).
    random_state : int
        Random seed for reproducibility (default: 42).
    hmm_covariance_type : str
        Covariance structure for HMM (only used when backend="hmm").
    hmm_n_iter : int
        Maximum EM iterations for HMM fitting.

    Examples
    --------
    >>> from indicators import build_features          # your pipeline
    >>> from regime_detector import RegimeDetector
    >>>
    >>> df = build_features(price_data)
    >>> detector = RegimeDetector(backend="kmeans")
    >>> regimes = detector.fit_predict(df)             # pd.Series of "BULL"/"BEAR"/"VOLATILE"
    >>>
    >>> detector.save("models/regime_detector.joblib")
    >>> loaded = RegimeDetector.load("models/regime_detector.joblib")
    >>> live_regimes = loaded.predict(live_df)
    """

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        n_clusters: int = KMEANS_N_CLUSTERS,
        random_state: int = RANDOM_STATE,
        hmm_covariance_type: str = "full",
        hmm_n_iter: int = 100,
    ) -> None:
        backend = backend.lower()
        if backend not in ("kmeans", "hmm"):
            raise ValueError(f"Unsupported backend '{backend}'. Choose 'kmeans' or 'hmm'.")

        self.backend_name = backend
        self.n_clusters = n_clusters
        self.random_state = random_state

        if backend == "kmeans":
            self._backend: _KMeansBackend | _HMMBackend = _KMeansBackend(
                n_clusters=n_clusters,
                random_state=random_state,
            )
        else:
            self._backend = _HMMBackend(
                n_components=n_clusters,
                random_state=random_state,
                covariance_type=hmm_covariance_type,
                n_iter=hmm_n_iter,
            )

        self._is_fitted: bool = False

    # Core API
  

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Fit the regime detector on historical feature data.

        This method:
          1. Validates required columns are present.
          2. Imputes missing values without look-ahead bias.
          3. Fits the chosen backend (KMeans or HMM) on the features.
          4. Derives the cluster → regime label mapping from mean returns.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame from ``indicators.py`` with columns:
            ``returns``, ``rsi``, ``volatility``, ``ma_diff``.

        Returns
        -------
        self
            Fitted RegimeDetector instance (enables method chaining).

        Raises
        ------
        ValueError
            If required feature columns are missing.
        """
        _validate_dataframe(df)
        clean_df = _impute_missing(df)

        X = clean_df[REQUIRED_FEATURES]
        returns = clean_df["returns"]

        logger.info(
            "Fitting RegimeDetector [backend=%s] on %d rows.", self.backend_name, len(X)
        )
        self._backend.fit(X, returns)
        self._is_fitted = True
        logger.info("RegimeDetector fitting complete.")
        return self


    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regimes for new (unseen) data using the fitted model.

        Missing values are imputed using the same forward-fill / backward-fill
        strategy as during training; the feature scaler uses **training-time
        statistics only** to prevent data leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with the same schema as the training data.

        Returns
        -------
        pd.Series
            Regime labels (``"BULL"``, ``"BEAR"``, ``"VOLATILE"``) aligned to
            the input DataFrame's index.

        Raises
        ------
        RuntimeError
            If called before ``fit()`` or ``fit_predict()``.
        ValueError
            If required feature columns are missing.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RegimeDetector must be fitted before calling predict(). "
                "Call fit() or fit_predict() first."
            )

        _validate_dataframe(df)
        clean_df = _impute_missing(df)
        X = clean_df[REQUIRED_FEATURES]

        logger.info("Predicting regimes for %d row(s).", len(X))
        regime_array = self._backend.predict(X)

        return pd.Series(
            data=regime_array,
            index=df.index,
            name="regime",
            dtype="object",
        )

  
    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Convenience method: fit the model and return in-sample regime labels.

        Equivalent to calling ``fit(df).predict(df)`` but more efficient
        because the backend can reuse internal state computed during fitting.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with columns: ``returns``, ``rsi``,
            ``volatility``, ``ma_diff``.

        Returns
        -------
        pd.Series
            Regime labels aligned to the input DataFrame's index.
        """
        return self.fit(df).predict(df)


    # Persistence helpers
    

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted RegimeDetector to disk using joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"models/regime_detector.joblib"``).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("RegimeDetector saved to '%s'.", path)


    @classmethod
    def load(cls, path: str | Path) -> "RegimeDetector":
        """
        Load a previously saved RegimeDetector from disk.

        Parameters
        ----------
        path : str or Path
            Path to a ``.joblib`` file created by :meth:`save`.

        Returns
        -------
        RegimeDetector
            A fully fitted RegimeDetector instance ready for inference.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{path}'.")

        detector: RegimeDetector = joblib.load(path)
        logger.info("RegimeDetector loaded from '%s'.", path)
        return detector

    # Diagnostics


    def regime_summary(self, regimes: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute descriptive statistics for each detected regime.

        Useful for validation and reporting after ``fit_predict``.

        Parameters
        ----------
        regimes : pd.Series
            Output of :meth:`fit_predict` or :meth:`predict`.
        df : pd.DataFrame
            The feature DataFrame used to generate *regimes*.

        Returns
        -------
        pd.DataFrame
            A table of mean feature values and observation counts per regime.
        """
        _validate_dataframe(df)
        combined = df[REQUIRED_FEATURES].copy()
        combined["regime"] = regimes.values

        summary = (
            combined.groupby("regime")[REQUIRED_FEATURES]
            .agg(["mean", "std", "count"])
        )
        return summary



    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"RegimeDetector(backend='{self.backend_name}', "
            f"n_clusters={self.n_clusters}, "
            f"random_state={self.random_state}, "
            f"status='{status}')"
        )



# Quick smoke-test (run directly: python regime_detector.py)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(RANDOM_STATE)
    n = 500

    mock_df = pd.DataFrame(
        {
            "returns":    rng.normal(0.0, 0.01, n),
            "rsi":        rng.uniform(20, 80, n),
            "volatility": rng.uniform(0.005, 0.03, n),
            "ma_diff":    rng.normal(0.0, 0.5, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )

    # Inject a small number of NaNs to exercise imputation
    mock_df.iloc[0:5, 0] = np.nan
    mock_df.iloc[10, 2] = np.nan

    # --- KMeans ---
    detector_km = RegimeDetector(backend="kmeans", n_clusters=3, random_state=RANDOM_STATE)
    regimes_km = detector_km.fit_predict(mock_df)

    print("\n=== KMeans Regime Counts ===")
    print(regimes_km.value_counts())

    print("\n=== Regime Summary ===")
    print(detector_km.regime_summary(regimes_km, mock_df))

    # Save / load round-trip
    detector_km.save("/tmp/regime_detector_test.joblib")
    loaded = RegimeDetector.load("/tmp/regime_detector_test.joblib")
    assert loaded.predict(mock_df).equals(regimes_km), "Round-trip prediction mismatch!"
    print("\nSave/load round-trip: OK")

    # --- HMM (if available) ---
    if HMM_AVAILABLE:
        detector_hmm = RegimeDetector(backend="hmm", n_clusters=3, random_state=RANDOM_STATE)
        regimes_hmm = detector_hmm.fit_predict(mock_df)
        print("\n=== HMM Regime Counts ===")
        print(regimes_hmm.value_counts())
    else:
        print("\nSkipping HMM smoke-test (hmmlearn not installed).")

    print("\nAll smoke-tests passed.")
