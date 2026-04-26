"""
NeuroTrade — Explainability Module
explainability.py — SHAP values, Attention Maps, Feature Importance visualization.

Provides interpretability for ML/DL model predictions.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

warnings.filterwarnings("ignore")

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False
    print("[explainability] shap not found — pip install shap")

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    SHAP-based feature importance for tree and kernel models.

    Supports:
    - TreeExplainer (XGBoost, RF) — exact, fast
    - KernelExplainer (SVM, any model) — model-agnostic, approximate
    """

    def __init__(self, model, model_type: str = "tree"):
        """
        model_type: "tree" | "kernel" | "linear"
        """
        if not _SHAP:
            raise ImportError("shap required — pip install shap")
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None

    def fit(self, X_background: np.ndarray, max_samples: int = 100):
        """Initialize explainer with background data."""
        bg = X_background
        if len(bg) > max_samples:
            idx = np.random.choice(len(bg), max_samples, replace=False)
            bg = bg[idx]

        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, bg)
        else:
            predict_fn = (self.model.predict_proba if hasattr(self.model, 'predict_proba')
                         else self.model.predict)
            self.explainer = shap.KernelExplainer(predict_fn, bg)
        return self

    def explain(self, X: np.ndarray, feature_names: List[str] = None,
                max_samples: int = 500) -> Dict:
        """Compute SHAP values."""
        if self.explainer is None:
            raise RuntimeError("Call fit() first")

        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]

        sv = self.explainer.shap_values(X)

        # handle multi-class (take class 1 for binary)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]

        self.shap_values = sv
        names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # global importance: mean |SHAP|
        mean_abs = np.abs(sv).mean(axis=0)
        importance = dict(zip(names, mean_abs.tolist()))
        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "shap_values": sv,
            "feature_names": names,
            "global_importance": sorted_imp,
            "top_10": dict(list(sorted_imp.items())[:10]),
            "expected_value": (self.explainer.expected_value[1]
                              if isinstance(self.explainer.expected_value, (list, np.ndarray))
                              else self.explainer.expected_value),
        }

    def explain_single(self, x: np.ndarray, feature_names: List[str] = None) -> Dict:
        """Explain a single prediction."""
        if self.explainer is None:
            raise RuntimeError("Call fit() first")

        x_2d = x.reshape(1, -1) if x.ndim == 1 else x
        sv = self.explainer.shap_values(x_2d)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        sv = sv.flatten()

        names = feature_names or [f"f{i}" for i in range(len(sv))]
        contributions = dict(zip(names, sv.tolist()))
        sorted_c = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

        return {
            "contributions": sorted_c,
            "top_5_positive": {k: v for k, v in sorted_c.items() if v > 0}.__class__(
                list({k: v for k, v in sorted_c.items() if v > 0}.items())[:5]),
            "top_5_negative": {k: v for k, v in sorted_c.items() if v < 0}.__class__(
                list({k: v for k, v in sorted_c.items() if v < 0}.items())[:5]),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENTION MAP EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class AttentionMapExtractor:
    """
    Extracts and interprets attention weights from LSTM/TFT models.
    """

    @staticmethod
    def extract_temporal_attention(model, X: np.ndarray,
                                   feature_names: List[str] = None) -> Dict:
        """
        Extract temporal attention weights (which time steps matter most).
        Works with LSTMAttention from dl_models.py.
        """
        if not _TORCH:
            raise ImportError("PyTorch required")

        model.eval()
        inp = torch.from_numpy(X).float()
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)

        with torch.no_grad():
            out, attn = model(inp)

        if attn is None:
            return {"error": "Model doesn't produce attention weights"}

        attn_np = attn.cpu().numpy()

        # average across batch
        if attn_np.ndim == 2:
            avg_attn = attn_np.mean(axis=0)
        elif attn_np.ndim == 3:
            avg_attn = attn_np.mean(axis=(0, 1))
        else:
            avg_attn = attn_np.flatten()

        # normalize
        avg_attn = avg_attn / (avg_attn.sum() + 1e-8)

        return {
            "temporal_weights": avg_attn,
            "peak_timestep": int(np.argmax(avg_attn)),
            "attention_entropy": float(-np.sum(avg_attn * np.log(avg_attn + 1e-8))),
            "concentration": float(np.max(avg_attn)),
            "raw_attention": attn_np,
        }

    @staticmethod
    def extract_feature_attention(model, X: np.ndarray,
                                  feature_names: List[str] = None) -> Dict:
        """
        Extract variable selection weights (which features matter most).
        Works with TFTModel from dl_models.py (VSN weights).
        """
        if not _TORCH:
            raise ImportError("PyTorch required")

        model.eval()
        inp = torch.from_numpy(X).float()
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)

        with torch.no_grad():
            out, vsn_weights = model(inp)

        if vsn_weights is None:
            return {"error": "Model doesn't produce feature weights"}

        w_np = vsn_weights.cpu().numpy()

        # average across batch and time
        if w_np.ndim == 3:
            avg_w = w_np.mean(axis=(0, 1))
        elif w_np.ndim == 2:
            avg_w = w_np.mean(axis=0)
        else:
            avg_w = w_np.flatten()

        names = feature_names or [f"feature_{i}" for i in range(len(avg_w))]
        names = names[:len(avg_w)]

        importance = dict(zip(names, avg_w.tolist()))
        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "feature_weights": sorted_imp,
            "top_10": dict(list(sorted_imp.items())[:10]),
            "raw_weights": w_np,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PERMUTATION IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

class PermutationImportance:
    """
    Model-agnostic feature importance via permutation.
    Works with any model that has a predict() method.
    """

    @staticmethod
    def compute(model, X: np.ndarray, y: np.ndarray,
                feature_names: List[str] = None,
                n_repeats: int = 5, metric: str = "accuracy") -> Dict:

        names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # baseline score
        if metric == "accuracy":
            from sklearn.metrics import accuracy_score
            preds = model.predict(X)
            if hasattr(preds, 'shape') and preds.ndim > 0:
                pred_labels = (preds > 0.5).astype(int) if preds.max() <= 1 else preds
            else:
                pred_labels = preds
            baseline = accuracy_score(y, pred_labels)
        else:
            baseline = -np.mean((model.predict(X) - y) ** 2)  # neg MSE

        importances = {}
        for j in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, j] = np.random.permutation(X_perm[:, j])
                preds = model.predict(X_perm)
                if metric == "accuracy":
                    pred_labels = (preds > 0.5).astype(int) if preds.max() <= 1 else preds
                    score = accuracy_score(y, pred_labels)
                else:
                    score = -np.mean((preds - y) ** 2)
                scores.append(baseline - score)
            importances[names[j]] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }

        sorted_imp = dict(sorted(importances.items(),
                                  key=lambda x: x[1]["mean"], reverse=True))
        return {
            "importance": sorted_imp,
            "baseline_score": baseline,
            "top_10": dict(list(sorted_imp.items())[:10]),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

class PredictionDecomposer:
    """
    Decomposes a prediction into contributions by feature group.
    Groups: Trend, Momentum, Volatility, Volume, Time.
    """

    GROUPS = {
        "Trend": ["SMA_20", "EMA_21", "ADX", "DI_Pos", "DI_Neg", "Close"],
        "Momentum": ["RSI", "MACD", "MACD_Signal", "MACD_Hist", "%K", "%D", "CCI"],
        "Volatility": ["ATR", "BB_Upper", "BB_Lower", "BB_Width",
                        "Realized_Vol_21", "Z_Score_20", "HL_Spread"],
        "Volume": ["Volume", "OBV", "VWAP"],
        "Time": ["DayOfWeek", "Month", "Quarter", "DayOfYear"],
    }

    @staticmethod
    def decompose(shap_result: Dict) -> Dict[str, float]:
        """Group SHAP contributions by feature category."""
        contributions = shap_result.get("global_importance", {})
        group_scores = {}

        for group_name, group_features in PredictionDecomposer.GROUPS.items():
            score = sum(contributions.get(f, 0) for f in group_features)
            group_scores[group_name] = round(score, 6)

        # unmatched features
        all_grouped = set(f for feats in PredictionDecomposer.GROUPS.values() for f in feats)
        other = sum(v for k, v in contributions.items() if k not in all_grouped)
        if other > 0:
            group_scores["Other"] = round(other, 6)

        return dict(sorted(group_scores.items(), key=lambda x: x[1], reverse=True))


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  explainability.py — Explainability Module Self-Test")
    print("=" * 65)

    np.random.seed(42)
    n, f = 200, 10
    X = np.random.randn(n, f).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n) * 0.2 > 0).astype(int)
    names = [f"feature_{i}" for i in range(f)]

    # test permutation importance with a simple sklearn model
    try:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X[:150], y[:150])

        print("\n  Permutation Importance:")
        perm = PermutationImportance.compute(rf, X[150:], y[150:], names, n_repeats=3)
        for k, v in list(perm["top_10"].items())[:5]:
            print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

        if _SHAP:
            print("\n  SHAP TreeExplainer:")
            explainer = SHAPExplainer(rf, "tree")
            explainer.fit(X[:100])
            result = explainer.explain(X[150:], names)
            for k, v in list(result["top_10"].items())[:5]:
                print(f"    {k}: {v:.4f}")

            print("\n  Single Prediction Explanation:")
            single = explainer.explain_single(X[150], names)
            for k, v in list(single["contributions"].items())[:3]:
                print(f"    {k}: {v:+.4f}")
        else:
            print("\n  [SKIP] SHAP not installed")

        print("\n  Prediction Decomposition:")
        mock_shap = {"global_importance": {
            "RSI": 0.15, "MACD": 0.12, "ATR": 0.08, "Close": 0.20,
            "Volume": 0.05, "SMA_20": 0.10, "BB_Width": 0.07, "OBV": 0.03,
        }}
        decomp = PredictionDecomposer.decompose(mock_shap)
        for group, score in decomp.items():
            print(f"    {group}: {score:.4f}")

    except ImportError as e:
        print(f"  [SKIP] {e}")

    print("\n  ✓  Explainability self-test complete.\n")
