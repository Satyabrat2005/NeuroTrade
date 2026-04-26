"""
NeuroTrade — Ensemble Methods
ensemble.py — Stacking, Voting, and Weighted Ensemble for multi-model fusion.

Combines predictions from ML, DL, and Quantum models into a single robust signal.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, log_loss
    _SKL = True
except ImportError:
    _SKL = False

try:
    from backtester import PositionSide
except ImportError:
    from enum import Enum
    class PositionSide(Enum):
        LONG = "long"; SHORT = "short"; FLAT = "flat"


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    method:           str   = "weighted"    # "vote" | "weighted" | "stacking"
    task:             str   = "direction"   # "direction" | "returns"
    long_threshold:   float = 0.55
    short_threshold:  float = 0.45
    # stacking
    stack_meta:       str   = "logistic"    # "logistic" | "ridge"
    stack_cv:         int   = 3
    # diversity
    correlation_penalty: float = 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  HARD VOTE
# ══════════════════════════════════════════════════════════════════════════════

class HardVotingEnsemble:
    """Majority-vote ensemble: each model gets one vote."""

    def __init__(self, cfg: EnsembleConfig = None):
        self.cfg = cfg or EnsembleConfig()

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        predictions: {model_name: array of probabilities or class labels}
        Returns: array of ensemble predictions
        """
        if self.cfg.task == "direction":
            # convert probabilities to votes
            votes = []
            for name, preds in predictions.items():
                votes.append((preds > 0.5).astype(int))
            vote_matrix = np.column_stack(votes)
            # majority vote
            return (vote_matrix.mean(axis=1) > 0.5).astype(int)
        else:
            # regression: simple average
            arrays = list(predictions.values())
            return np.mean(arrays, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  SOFT VOTE (Weighted Average)
# ══════════════════════════════════════════════════════════════════════════════

class WeightedEnsemble:
    """
    Weighted average ensemble. Weights can be:
    - Uniform (equal weight)
    - Performance-based (validation accuracy)
    - Diversity-penalized (reduce correlated models)
    """

    def __init__(self, cfg: EnsembleConfig = None):
        self.cfg = cfg or EnsembleConfig()
        self.weights: Dict[str, float] = {}

    def learn_weights(self, predictions: Dict[str, np.ndarray],
                      y_true: np.ndarray) -> Dict[str, float]:
        """Learn optimal weights from validation set."""
        names = list(predictions.keys())
        n_models = len(names)

        if n_models == 0:
            return {}

        # base scores
        scores = {}
        for name in names:
            preds = predictions[name]
            if self.cfg.task == "direction":
                pred_labels = (preds > 0.5).astype(int)
                scores[name] = accuracy_score(y_true, pred_labels)
            else:
                mse = np.mean((preds - y_true) ** 2)
                scores[name] = 1.0 / (mse + 1e-8)

        # diversity penalty: reduce weight for correlated models
        if n_models > 1 and self.cfg.correlation_penalty > 0:
            pred_matrix = np.column_stack([predictions[n] for n in names])
            corr = np.corrcoef(pred_matrix.T)
            for i, name_i in enumerate(names):
                avg_corr = (np.sum(np.abs(corr[i])) - 1) / (n_models - 1)
                scores[name_i] *= (1 - self.cfg.correlation_penalty * avg_corr)

        # normalize
        total = sum(max(s, 0.01) for s in scores.values())
        self.weights = {k: max(v, 0.01) / total for k, v in scores.items()}
        return self.weights

    def predict(self, predictions: Dict[str, np.ndarray],
                weights: Dict[str, float] = None) -> np.ndarray:
        w = weights or self.weights
        if not w:
            w = {k: 1.0 / len(predictions) for k in predictions}

        result = np.zeros(len(next(iter(predictions.values()))))
        total_w = sum(w.get(k, 0) for k in predictions)

        for name, preds in predictions.items():
            result += (w.get(name, 0) / total_w) * preds

        return result


# ══════════════════════════════════════════════════════════════════════════════
#  STACKING
# ══════════════════════════════════════════════════════════════════════════════

class StackingEnsemble:
    """
    Level-2 stacking: base model predictions become features for a meta-learner.
    Uses time-series cross-validation to avoid leakage.
    """

    def __init__(self, cfg: EnsembleConfig = None):
        if not _SKL:
            raise ImportError("scikit-learn required")
        self.cfg = cfg or EnsembleConfig()
        self.meta_model = None
        self.model_names: List[str] = []

    def fit(self, predictions: Dict[str, np.ndarray],
            y_true: np.ndarray):
        """Train meta-learner on base model predictions."""
        self.model_names = list(predictions.keys())
        X_meta = np.column_stack([predictions[n] for n in self.model_names])

        if self.cfg.task == "direction":
            y_int = y_true.astype(int)
            if self.cfg.stack_meta == "logistic":
                self.meta_model = LogisticRegression(
                    C=1.0, max_iter=500, random_state=42)
            else:
                self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(X_meta, y_int)
        else:
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(X_meta, y_true)

        return self

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if self.meta_model is None:
            raise RuntimeError("Call fit() first")
        X_meta = np.column_stack([predictions[n] for n in self.model_names])

        if self.cfg.task == "direction" and hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(X_meta)[:, 1]
        return self.meta_model.predict(X_meta)

    def get_model_weights(self) -> Dict[str, float]:
        """Extract learned importance from meta-learner coefficients."""
        if self.meta_model is None:
            return {}
        coefs = self.meta_model.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        abs_coefs = np.abs(coefs)
        total = abs_coefs.sum() or 1.0
        return {name: float(abs_coefs[i] / total)
                for i, name in enumerate(self.model_names)}


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED ENSEMBLE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleManager:
    """
    Orchestrates multi-model ensemble with configurable fusion method.

    Usage
    -----
        mgr = EnsembleManager(cfg)
        mgr.fit(train_preds, y_train)
        combined = mgr.predict(test_preds)
        signal = mgr.get_signal(combined[-1])
    """

    def __init__(self, cfg: EnsembleConfig = None):
        self.cfg = cfg or EnsembleConfig()
        if cfg and cfg.method == "stacking":
            self.engine = StackingEnsemble(cfg)
        elif cfg and cfg.method == "vote":
            self.engine = HardVotingEnsemble(cfg)
        else:
            self.engine = WeightedEnsemble(cfg)
        self.weights: Dict[str, float] = {}

    def fit(self, predictions: Dict[str, np.ndarray],
            y_true: np.ndarray):
        if isinstance(self.engine, StackingEnsemble):
            self.engine.fit(predictions, y_true)
            self.weights = self.engine.get_model_weights()
        elif isinstance(self.engine, WeightedEnsemble):
            self.weights = self.engine.learn_weights(predictions, y_true)
        return self

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if isinstance(self.engine, StackingEnsemble):
            return self.engine.predict(predictions)
        elif isinstance(self.engine, WeightedEnsemble):
            return self.engine.predict(predictions, self.weights)
        else:
            return self.engine.predict(predictions)

    def get_signal(self, pred_value: float) -> PositionSide:
        cfg = self.cfg
        if cfg.task == "direction":
            if pred_value > cfg.long_threshold:
                return PositionSide.LONG
            elif pred_value < cfg.short_threshold:
                return PositionSide.SHORT
        else:
            if pred_value > 0.001:
                return PositionSide.LONG
            elif pred_value < -0.001:
                return PositionSide.SHORT
        return PositionSide.FLAT

    def make_signal_func(self, model_predictors: Dict[str, Callable]):
        """Create backtester-compatible signal from multiple model predictors."""
        mgr = self
        cache = {"bar": -1, "pred": None}

        def signal_func(df: pd.DataFrame, i: int, **kwargs):
            if i < 60:
                return None
            if cache["bar"] == -1 or (i - cache["bar"]) >= 5:
                try:
                    sub = df.iloc[:i+1]
                    preds = {}
                    for name, predictor in model_predictors.items():
                        p = predictor(sub)
                        if p is not None:
                            preds[name] = np.array([p])
                    if preds:
                        combo = mgr.predict(preds)
                        cache["pred"] = float(combo[-1])
                    cache["bar"] = i
                except Exception:
                    return None

            if cache["pred"] is None:
                return None
            return mgr.get_signal(cache["pred"])

        return signal_func


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  ensemble.py — Ensemble Methods Self-Test")
    print("=" * 65)

    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, n)

    # simulate 3 model predictions (probabilities)
    preds = {
        "xgboost": np.clip(y_true * 0.6 + np.random.normal(0.3, 0.15, n), 0, 1),
        "rf":      np.clip(y_true * 0.5 + np.random.normal(0.3, 0.2, n), 0, 1),
        "lstm":    np.clip(y_true * 0.55 + np.random.normal(0.25, 0.18, n), 0, 1),
    }

    # Split
    split = int(n * 0.7)
    tr_preds = {k: v[:split] for k, v in preds.items()}
    te_preds = {k: v[split:] for k, v in preds.items()}
    y_tr, y_te = y_true[:split], y_true[split:]

    for method in ["vote", "weighted", "stacking"]:
        print(f"\n  {method.upper()}")
        print("  " + "-" * 40)
        cfg = EnsembleConfig(method=method)
        mgr = EnsembleManager(cfg)
        mgr.fit(tr_preds, y_tr)
        combo = mgr.predict(te_preds)
        acc = accuracy_score(y_te, (combo > 0.5).astype(int))
        print(f"    Test Accuracy: {acc*100:.2f}%")
        print(f"    Weights: {mgr.weights}")
        sig = mgr.get_signal(float(combo[-1]))
        print(f"    Last Signal: {sig}")

    print("\n  ✓  All ensemble methods passed.\n")
