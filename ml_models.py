"""
NeuroTrade — Classical ML Models
ml_models.py — XGBoost, Random Forest, SVM for time-series price prediction.

Each model:
  - Accepts OHLCV + indicator DataFrames
  - Predicts next-bar direction or returns
  - Exposes unified train / predict / signal interface
  - Generates signals compatible with backtester.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ── Optional deps (graceful fallback) ────────────────────────────────────────
try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                                 mean_absolute_error, r2_score)
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.pipeline import Pipeline
    _SKL = True
except ImportError:
    _SKL = False
    print("[ml_models] scikit-learn not found — pip install scikit-learn")

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False
    print("[ml_models] xgboost not found — pip install xgboost")

try:
    from backtester import PositionSide
except ImportError:
    from enum import Enum
    class PositionSide(Enum):
        LONG = "long"
        SHORT = "short"
        FLAT = "flat"


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLConfig:
    """Configuration for all classical ML models."""
    # data
    lookback:         int   = 60
    forecast_horizon: int   = 5
    target_type:      str   = "direction"   # "direction" | "returns"
    train_ratio:      float = 0.70
    val_ratio:        float = 0.15

    # XGBoost
    xgb_n_estimators:   int   = 300
    xgb_max_depth:      int   = 6
    xgb_learning_rate:  float = 0.05
    xgb_subsample:      float = 0.8
    xgb_colsample:      float = 0.8
    xgb_early_stopping: int   = 20

    # Random Forest
    rf_n_estimators:  int   = 200
    rf_max_depth:     int   = 12
    rf_min_samples:   int   = 5

    # SVM
    svm_C:            float = 10.0
    svm_gamma:        str   = "scale"
    svm_kernel:       str   = "rbf"

    # signal
    long_threshold:   float = 0.55
    short_threshold:  float = 0.45
    model_dir:        str   = "./models"


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

class MLFeatureEngineer:
    """
    Builds a flat feature matrix from OHLCV+indicator DataFrame.
    Adds lag features, rolling stats, and cross features for ML models.
    """

    def __init__(self, cfg: MLConfig = None):
        self.cfg = cfg or MLConfig()
        self.scaler: Optional[RobustScaler] = None
        self.feature_cols: List[str] = []

    def build(self, df: pd.DataFrame,
              fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns
        -------
        X : (n_samples, n_features)
        y : (n_samples,)  — direction (0/1) or returns (float)
        cols : feature column names
        """
        data = self._select_features(df)
        data = self._add_lag_features(data)
        data = self._add_rolling_stats(data)
        data = self._add_cross_features(data)
        data = self._add_time_features(data, df)

        # target
        if self.cfg.target_type == "direction":
            future_ret = df["Close"].pct_change(self.cfg.forecast_horizon).shift(
                -self.cfg.forecast_horizon)
            target = (future_ret > 0).astype(int)
        else:
            target = df["Close"].pct_change(self.cfg.forecast_horizon).shift(
                -self.cfg.forecast_horizon)

        target.name = "target"
        valid = data.notna().all(axis=1) & target.notna()
        data = data.loc[valid]
        target = target.loc[valid]

        if len(data) == 0:
            raise ValueError("Not enough valid data points to train the model. Try a longer timeframe.")

        self.feature_cols = list(data.columns)
        values = data.values.astype(np.float32)

        if fit_scaler:
            self.scaler = RobustScaler()
            values = self.scaler.fit_transform(values)
        elif self.scaler is not None:
            values = self.scaler.transform(values)

        return values, target.values.astype(np.float32), self.feature_cols

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        data = self._select_features(df)
        data = self._add_lag_features(data)
        data = self._add_rolling_stats(data)
        data = self._add_cross_features(data)
        data = self._add_time_features(data, df)
        for c in self.feature_cols:
            if c not in data.columns:
                data[c] = 0.0
        data = data[self.feature_cols].dropna()
        return self.scaler.transform(data.values.astype(np.float32))

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in CURATED_FEATURES if c in df.columns]
        return df[cols].copy()

    def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if "Close" in data.columns:
            ret = data["Close"].pct_change()
            for lag in [1, 3, 5, 10]:
                data[f"Ret_Lag_{lag}"] = ret.shift(lag)
        return data

    def _add_rolling_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        if "Close" in data.columns:
            ret = data["Close"].pct_change()
            for w in [5, 10, 21]:
                data[f"Vol_{w}d"] = ret.rolling(w).std()
                data[f"Mean_{w}d"] = ret.rolling(w).mean()
            data["Skew_21d"] = ret.rolling(21).skew()
        return data

    def _add_cross_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if "RSI" in data.columns and "ATR" in data.columns:
            data["RSI_x_ATR"] = data["RSI"] * data["ATR"]
        if "MACD" in data.columns and "Volume" in data.columns:
            data["MACD_x_Vol"] = data["MACD"] * np.log1p(data["Volume"])
        return data

    @staticmethod
    def _add_time_features(data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        data = data.copy()
        data["DayOfWeek"] = idx.dayofweek.astype(np.float32) / 4.0
        data["Month"] = (idx.month.astype(np.float32) - 1) / 11.0
        data["Quarter"] = (idx.quarter.astype(np.float32) - 1) / 3.0
        return data


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostModel:
    """XGBoost gradient-boosted trees wrapper."""

    def __init__(self, cfg: MLConfig, task: str = "direction"):
        self.cfg = cfg
        self.task = task
        self.model = None

    def fit(self, X_tr, y_tr, X_va, y_va):
        if not _XGB:
            raise ImportError("xgboost required — pip install xgboost")
        cfg = self.cfg
        if self.task == "direction":
            self.model = xgb.XGBClassifier(
                n_estimators=cfg.xgb_n_estimators,
                max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_learning_rate,
                subsample=cfg.xgb_subsample,
                colsample_bytree=cfg.xgb_colsample,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=cfg.xgb_n_estimators,
                max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_learning_rate,
                subsample=cfg.xgb_subsample,
                colsample_bytree=cfg.xgb_colsample,
                eval_metric="rmse",
                random_state=42,
                n_jobs=-1,
            )
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        return self

    def predict(self, X):
        if self.task == "direction":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def feature_importance(self, cols):
        imp = self.model.feature_importances_
        return dict(zip(cols, imp.tolist()))


class RandomForestModel:
    """Random Forest wrapper."""

    def __init__(self, cfg: MLConfig, task: str = "direction"):
        self.cfg = cfg
        self.task = task
        self.model = None

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        cfg = self.cfg
        if self.task == "direction":
            self.model = RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_leaf=cfg.rf_min_samples,
                random_state=42, n_jobs=-1,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_leaf=cfg.rf_min_samples,
                random_state=42, n_jobs=-1,
            )
        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, X):
        if self.task == "direction":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def feature_importance(self, cols):
        imp = self.model.feature_importances_
        return dict(zip(cols, imp.tolist()))


class SVMModel:
    """Support Vector Machine wrapper (RBF kernel)."""

    def __init__(self, cfg: MLConfig, task: str = "direction"):
        self.cfg = cfg
        self.task = task
        self.model = None

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        cfg = self.cfg
        if self.task == "direction":
            self.model = SVC(
                C=cfg.svm_C, gamma=cfg.svm_gamma, kernel=cfg.svm_kernel,
                probability=True, random_state=42,
            )
        else:
            self.model = SVR(
                C=cfg.svm_C, gamma=cfg.svm_gamma, kernel=cfg.svm_kernel,
            )
        # SVM is slow on large data — sample if needed
        max_n = 5000
        if len(X_tr) > max_n:
            idx = np.random.choice(len(X_tr), max_n, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, X):
        if self.task == "direction":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED TRAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLTrainingResult:
    model_name: str
    metrics:    Dict
    duration:   float
    feature_importance: Optional[Dict] = None


class MLTrainer:
    """
    Unified trainer for XGBoost, RF, and SVM.

    Usage
    -----
        trainer = MLTrainer(cfg)
        results = trainer.train_all(df)
        pred    = trainer.predict(df)
        signal  = trainer.get_signal(df, i)
    """

    MODEL_MAP = {"xgboost": XGBoostModel, "rf": RandomForestModel, "svm": SVMModel}

    def __init__(self, cfg: MLConfig = None):
        self.cfg = cfg or MLConfig()
        self.eng = MLFeatureEngineer(self.cfg)
        self.models: Dict[str, object] = {}
        self.results: Dict[str, MLTrainingResult] = {}
        self.weights: Dict[str, float] = {}
        self._trained = False

    def train_all(self, df: pd.DataFrame,
                  progress_cb: Optional[Callable] = None) -> Dict[str, MLTrainingResult]:
        if not _SKL:
            raise ImportError("scikit-learn required — pip install scikit-learn")

        cfg = self.cfg
        task = cfg.target_type

        # feature engineering
        X, y, cols = self.eng.build(df, fit_scaler=True)
        n = len(X)
        n_tr = int(n * cfg.train_ratio)
        n_va = int(n * cfg.val_ratio)

        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_va, y_va = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
        X_te, y_te = X[n_tr+n_va:], y[n_tr+n_va:]

        if task == "direction":
            y_tr = y_tr.astype(int)
            y_va = y_va.astype(int)
            y_te = y_te.astype(int)

        all_results = {}
        scores = {}

        for name, ModelCls in self.MODEL_MAP.items():
            if name == "xgboost" and not _XGB:
                print(f"  [SKIP] XGBoost not installed")
                continue

            t0 = time.time()
            print(f"\n  Training {name.upper()}...")
            model = ModelCls(cfg, task)
            model.fit(X_tr, y_tr, X_va, y_va)
            dur = time.time() - t0

            # evaluate on test set
            preds = model.predict(X_te)
            if task == "direction":
                pred_labels = (preds > 0.5).astype(int)
                acc = accuracy_score(y_te, pred_labels) * 100
                f1 = f1_score(y_te, pred_labels, zero_division=0) * 100
                metrics = {"accuracy": round(acc, 2), "f1": round(f1, 2),
                           "n_test": len(y_te)}
                scores[name] = acc
                print(f"    Accuracy: {acc:.2f}%  F1: {f1:.2f}%  ({dur:.1f}s)")
            else:
                mse = mean_squared_error(y_te, preds)
                mae = mean_absolute_error(y_te, preds)
                r2 = r2_score(y_te, preds)
                metrics = {"mse": round(mse, 6), "mae": round(mae, 6),
                           "r2": round(r2, 4), "n_test": len(y_te)}
                scores[name] = max(r2, 0.01) * 100
                print(f"    MSE: {mse:.6f}  R²: {r2:.4f}  ({dur:.1f}s)")

            fi = model.feature_importance(cols) if hasattr(model, 'feature_importance') else None
            result = MLTrainingResult(name, metrics, dur, fi)

            self.models[name] = model
            self.results[name] = result
            all_results[name] = result

            if progress_cb:
                progress_cb(name, metrics)

        # learn ensemble weights by score
        total = sum(scores.values()) or 1.0
        self.weights = {k: v / total for k, v in scores.items()}
        self._trained = True

        print(f"\n  Ensemble weights: {self.weights}")
        return all_results

    def predict(self, df: pd.DataFrame) -> Dict:
        if not self._trained:
            raise RuntimeError("Call train_all() first")

        X = self.eng.transform(df)
        if len(X) == 0:
            return {"ensemble": np.array([]), "per_model": {}}

        per_model = {}
        for name, model in self.models.items():
            per_model[name] = model.predict(X)

        # weighted ensemble
        combo = np.zeros(len(X))
        total_w = sum(self.weights[k] for k in per_model)
        for name, preds in per_model.items():
            combo += (self.weights[name] / total_w) * preds

        return {"ensemble": combo, "per_model": per_model, "weights": self.weights}

    def predict_latest(self, df: pd.DataFrame) -> float:
        result = self.predict(df)
        if len(result["ensemble"]) == 0:
            return 0.5
        return float(result["ensemble"][-1])

    def get_signal(self, df: pd.DataFrame, i: int = -1) -> Optional[PositionSide]:
        if not self._trained:
            return None
        end = i if i >= 0 else len(df)
        start = max(0, end - self.cfg.lookback - 100)
        subset = df.iloc[start:end].copy()
        if len(subset) < self.cfg.lookback + 10:
            return None

        pred = self.predict_latest(subset)
        cfg = self.cfg

        if cfg.target_type == "direction":
            if pred > cfg.long_threshold:
                return PositionSide.LONG
            elif pred < cfg.short_threshold:
                return PositionSide.SHORT
        else:
            if pred > 0.001:
                return PositionSide.LONG
            elif pred < -0.001:
                return PositionSide.SHORT
        return PositionSide.FLAT


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATOR (backtester compatible)
# ══════════════════════════════════════════════════════════════════════════════

class MLSignalGenerator:
    """Converts ML predictions into backtester-compatible signal functions."""

    def __init__(self, trainer: MLTrainer, threshold: float = 0.55):
        self.trainer = trainer
        self.threshold = threshold
        self._cache_bar = -1
        self._cache_pred = None

    def make_signal_func(self, df: pd.DataFrame) -> Callable:
        engine = self
        cfg = self.trainer.cfg

        def signal_func(df_bt: pd.DataFrame, i: int, **kwargs):
            if i < cfg.lookback + 30:
                return None
            # re-predict every 5 bars
            if engine._cache_bar == -1 or (i - engine._cache_bar) >= 5:
                try:
                    sub = df_bt.iloc[:i + 1]
                    engine._cache_pred = engine.trainer.predict_latest(sub)
                    engine._cache_bar = i
                except Exception:
                    return None

            pred = engine._cache_pred
            if pred is None:
                return None

            if cfg.target_type == "direction":
                if pred > engine.threshold:
                    return PositionSide.LONG
                elif pred < (1.0 - engine.threshold):
                    return PositionSide.SHORT
            else:
                if pred > 0.001:
                    return PositionSide.LONG
                elif pred < -0.001:
                    return PositionSide.SHORT
            return PositionSide.FLAT

        return signal_func


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_df(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    close = 1000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    df = pd.DataFrame({
        "Open": close * (1 + np.random.normal(0, 0.002, n)),
        "High": close * (1 + np.abs(np.random.normal(0, 0.008, n))),
        "Low":  close * (1 - np.abs(np.random.normal(0, 0.008, n))),
        "Close": close,
        "Volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=dates)
    ret = df["Close"].pct_change()
    df["Returns"] = ret
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_21"] = df["Close"].ewm(span=21).mean()
    e12 = df["Close"].ewm(span=12).mean()
    e26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = e12 - e26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    d = df["Close"].diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ADX"] = 25 + np.random.normal(0, 8, n)
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = mid + 2*std
    df["BB_Lower"] = mid - 2*std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / mid
    df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    tp = (df["High"]+df["Low"]+df["Close"])/3
    df["VWAP"] = (tp*df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015*tp.rolling(20).std()+1e-9)
    df["%K"] = 50 + np.random.normal(0, 20, n)
    df["%D"] = df["%K"].rolling(3).mean()
    df["Realized_Vol_21"] = df["Log_Returns"].rolling(21).std() * np.sqrt(252)
    df["Z_Score_20"] = (df["Close"] - df["Close"].rolling(20).mean()) / (df["Close"].rolling(20).std()+1e-9)
    return df.dropna()


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  ml_models.py — Classical ML Engine Self-Test")
    print("=" * 65)

    df = _generate_synthetic_df(600)
    print(f"\n  Synthetic df: {len(df)} rows × {len(df.columns)} cols")

    cfg = MLConfig(forecast_horizon=5, xgb_n_estimators=50, rf_n_estimators=50)
    trainer = MLTrainer(cfg)
    results = trainer.train_all(df)

    for name, res in results.items():
        print(f"\n  {name.upper()}: {res.metrics}  ({res.duration:.1f}s)")

    pred = trainer.predict_latest(df)
    sig = trainer.get_signal(df)
    print(f"\n  Latest prediction: {pred:.4f}")
    print(f"  Signal: {sig}")
    print("\n  OK  All ML models passed self-test.\n")
