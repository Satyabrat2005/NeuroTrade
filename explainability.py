"""
explainability.py — SHAP explanations & attention-style heatmaps for NeuroTrade.

Provides two main capabilities:
  1. SHAP analysis   — per-sample and global feature contributions
  2. Attention maps  — time × feature importance heatmaps that mimic
                       attention visualisations from deep learning

Works with any trained BaseModel from ml_models.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  1. SHAP Explainer
# ══════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """Compute SHAP values for a trained model.

    Supports tree-based (RF, XGBoost) and kernel-based (SVM) models
    by automatically selecting the right SHAP explainer.
    """

    def __init__(self):
        if not SHAP_AVAILABLE:
            raise ImportError("shap is not installed. Run: pip install shap")
        self._explainer = None
        self._shap_values = None
        self._base_value = None
        self._feature_names: List[str] = []

    # ── fit ───────────────────────────────────────────────────────────────
    def fit(
        self,
        model,
        X_background: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_type: str = "auto",
    ) -> "SHAPExplainer":
        """Create the SHAP explainer from a background dataset.

        Parameters
        ----------
        model : BaseModel wrapper or raw sklearn / xgb estimator
            If it has a ``.model`` attribute the underlying estimator is used.
        X_background : ndarray
            A representative sample (≤200 rows recommended) for KernelExplainer.
        feature_names : list[str], optional
            Column names for display.
        model_type : str
            ``'tree'``, ``'kernel'``, or ``'auto'`` (auto-detects).
        """
        estimator = getattr(model, "model", model)

        # If it's a Pipeline (SVM), pull the raw estimator for type-check
        raw_est = estimator
        if hasattr(estimator, "named_steps"):
            raw_est = list(estimator.named_steps.values())[-1]

        self._feature_names = list(feature_names) if feature_names else [
            f"f{i}" for i in range(X_background.shape[1])
        ]

        if model_type == "auto":
            tree_types = (
                "RandomForestClassifier", "XGBClassifier",
                "GradientBoostingClassifier", "DecisionTreeClassifier",
            )
            model_type = (
                "tree" if type(raw_est).__name__ in tree_types else "kernel"
            )

        if model_type == "tree":
            self._explainer = shap.TreeExplainer(estimator)
        else:
            # Use a subsample for speed
            bg = X_background
            if len(bg) > 100:
                idx = np.random.choice(len(bg), 100, replace=False)
                bg = bg[idx]

            predict_fn = (
                estimator.predict_proba
                if hasattr(estimator, "predict_proba")
                else estimator.predict
            )
            self._explainer = shap.KernelExplainer(predict_fn, bg)

        return self

    # ── explain ──────────────────────────────────────────────────────────
    def explain(
        self, X: np.ndarray, max_samples: int = 200,
    ) -> Dict:
        """Compute SHAP values for the given samples.

        Returns
        -------
        dict with keys:
            shap_values    — (n_samples, n_features) array for class 1
            base_value     — expected model output
            feature_names  — column labels
            X              — the input samples used
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit() first.")

        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]

        raw_shap = self._explainer.shap_values(X)

        # Normalise shape: we want values for class 1 (positive / BUY)
        if isinstance(raw_shap, list):
            sv = raw_shap[1]   # binary: [class0, class1]
        elif raw_shap.ndim == 3:
            sv = raw_shap[:, :, 1]
        else:
            sv = raw_shap

        # Base value
        bv = self._explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            bv = bv[1] if len(bv) > 1 else bv[0]

        self._shap_values = sv
        self._base_value = float(bv)

        return {
            "shap_values": sv,
            "base_value": self._base_value,
            "feature_names": self._feature_names,
            "X": X,
        }

    # ── global_importance ────────────────────────────────────────────────
    def global_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Mean |SHAP| per feature, sorted descending."""
        if self._shap_values is None:
            raise RuntimeError("Call .explain() first.")

        mean_abs = np.abs(self._shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self._feature_names,
            "mean_shap": mean_abs,
        }).sort_values("mean_shap", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    # ── single_explanation ───────────────────────────────────────────────
    def single_explanation(self, idx: int = -1) -> pd.DataFrame:
        """SHAP breakdown for one sample (default: last / latest)."""
        if self._shap_values is None:
            raise RuntimeError("Call .explain() first.")

        sv = self._shap_values[idx]
        df = pd.DataFrame({
            "feature": self._feature_names,
            "shap_value": sv,
            "abs_shap": np.abs(sv),
        }).sort_values("abs_shap", ascending=False)
        return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  2. Attention Maps  (time × feature importance heatmap)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionMapper:
    """Generate time-series 'attention' heatmaps using rolling SHAP or
    permutation importance, analogous to transformer attention maps.

    Each cell (t, f) shows how much feature *f* contributed to the
    prediction at time step *t*.
    """

    def __init__(self):
        self._attention_matrix: Optional[pd.DataFrame] = None

    # ── from_shap ────────────────────────────────────────────────────────
    def from_shap(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        window: int = 20,
        model_type: str = "auto",
    ) -> pd.DataFrame:
        """Build attention matrix using rolling-window SHAP values.

        Parameters
        ----------
        model : trained BaseModel
        X : (n_samples, n_features) full feature matrix (time-ordered)
        feature_names : column names
        window : rolling window size for SHAP computation
        model_type : passed to SHAPExplainer.fit

        Returns
        -------
        DataFrame  (n_windows, n_features) with normalised attention scores.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap is required for SHAP-based attention maps.")

        n = len(X)
        if n < window + 10:
            window = max(n // 3, 5)

        attention_rows = []
        step = max(1, window // 4)

        for start in range(0, n - window + 1, step):
            chunk = X[start : start + window]
            bg = X[max(0, start - 50) : start] if start > 50 else X[:50]
            if len(bg) < 5:
                bg = chunk[:5]

            try:
                exp = SHAPExplainer()
                exp.fit(model, bg, feature_names=feature_names, model_type=model_type)
                result = exp.explain(chunk, max_samples=window)
                sv = result["shap_values"]
                # Average |SHAP| over the window → one row
                avg_abs = np.abs(sv).mean(axis=0)
                attention_rows.append(avg_abs)
            except Exception:
                attention_rows.append(np.zeros(len(feature_names)))

        mat = np.array(attention_rows)

        # Row-normalise to [0, 1]
        row_max = mat.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        mat = mat / row_max

        self._attention_matrix = pd.DataFrame(
            mat, columns=feature_names
        )
        return self._attention_matrix

    # ── from_permutation ─────────────────────────────────────────────────
    def from_permutation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        window: int = 20,
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """Build attention matrix using rolling permutation importance.

        Faster than SHAP but less granular.  Does NOT require the shap
        package.
        """
        from sklearn.metrics import accuracy_score

        n = len(X)
        if n < window + 10:
            window = max(n // 3, 5)

        attention_rows = []
        step = max(1, window // 4)

        for start in range(0, n - window + 1, step):
            Xw = X[start : start + window]
            yw = y[start : start + window]

            base_acc = accuracy_score(yw, model.predict(Xw))
            importances = np.zeros(Xw.shape[1])

            for f_idx in range(Xw.shape[1]):
                drops = []
                for _ in range(n_repeats):
                    Xp = Xw.copy()
                    Xp[:, f_idx] = np.random.permutation(Xp[:, f_idx])
                    perm_acc = accuracy_score(yw, model.predict(Xp))
                    drops.append(base_acc - perm_acc)
                importances[f_idx] = max(np.mean(drops), 0)

            attention_rows.append(importances)

        mat = np.array(attention_rows)
        row_max = mat.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        mat = mat / row_max

        self._attention_matrix = pd.DataFrame(
            mat, columns=feature_names
        )
        return self._attention_matrix

    @property
    def matrix(self) -> Optional[pd.DataFrame]:
        return self._attention_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  3. Convenience wrappers for app.py
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_for_model(
    model,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    model_type: str = "auto",
    max_samples: int = 100,
) -> Optional[Dict]:
    """One-shot SHAP computation. Returns None on failure."""
    if not SHAP_AVAILABLE:
        return None
    try:
        exp = SHAPExplainer()
        exp.fit(model, X_train, feature_names=feature_names, model_type=model_type)
        result = exp.explain(X_explain, max_samples=max_samples)
        result["global_importance"] = exp.global_importance()
        result["latest_explanation"] = exp.single_explanation()
        return result
    except Exception as e:
        import warnings
        warnings.warn(f"SHAP computation failed: {e}")
        return None


def compute_attention_map(
    model,
    X: np.ndarray,
    feature_names: List[str],
    y: Optional[np.ndarray] = None,
    method: str = "permutation",
    window: int = 20,
    model_type: str = "auto",
) -> Optional[pd.DataFrame]:
    """One-shot attention map. Returns None on failure."""
    try:
        mapper = AttentionMapper()
        if method == "shap" and SHAP_AVAILABLE:
            return mapper.from_shap(
                model, X, feature_names, window=window, model_type=model_type
            )
        elif y is not None:
            return mapper.from_permutation(
                model, X, y, feature_names, window=window
            )
        else:
            return None
    except Exception as e:
        import warnings
        warnings.warn(f"Attention map failed: {e}")
        return None
