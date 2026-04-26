"""
NeuroTrade — Probability Calibration
calibration.py — Platt Scaling, Isotonic Regression, and calibration diagnostics.

Converts raw model scores into well-calibrated probabilities.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

warnings.filterwarnings("ignore")

try:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    _SKL = True
except ImportError:
    _SKL = False
    print("[calibration] scikit-learn not found — pip install scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
#  PLATT SCALING
# ══════════════════════════════════════════════════════════════════════════════

class PlattScaling:
    """
    Platt Scaling: fits a logistic regression on raw model scores
    to produce calibrated probabilities. P(y=1|s) = 1/(1+exp(As+B))

    Best for: SVM, neural networks, any model with uncalibrated scores.
    """

    def __init__(self):
        self.model = None
        self._fitted = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray):
        """
        scores: raw model outputs (probabilities or decision function)
        y_true: binary labels (0/1)
        """
        if not _SKL:
            raise ImportError("scikit-learn required")

        self.model = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
        X = scores.reshape(-1, 1) if scores.ndim == 1 else scores
        self.model.fit(X, y_true.astype(int))
        self._fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Transform raw scores to calibrated probabilities."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        X = scores.reshape(-1, 1) if scores.ndim == 1 else scores
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> Dict:
        """Return fitted Platt parameters A and B."""
        if not self._fitted:
            return {}
        return {
            "A": float(self.model.coef_[0][0]),
            "B": float(self.model.intercept_[0]),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ISOTONIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

class IsotonicCalibration:
    """
    Isotonic Regression calibration: non-parametric, monotonic mapping.

    Better than Platt when the calibration curve is non-sigmoid.
    Needs more data (~1000+ samples) to avoid overfitting.
    """

    def __init__(self):
        self.model = None
        self._fitted = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray):
        if not _SKL:
            raise ImportError("scikit-learn required")

        self.model = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self.model.fit(scores.flatten(), y_true.astype(float))
        self._fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return self.model.predict(scores.flatten())


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE SCALING (for neural networks)
# ══════════════════════════════════════════════════════════════════════════════

class TemperatureScaling:
    """
    Temperature Scaling: divides logits by learned temperature T.
    P(y|x) = softmax(z/T). Simple, single-parameter calibration.

    Best for: Deep learning models (LSTM, TCN, TFT).
    """

    def __init__(self):
        self.temperature = 1.0
        self._fitted = False

    def fit(self, logits: np.ndarray, y_true: np.ndarray,
            lr: float = 0.01, max_iter: int = 200):
        """Find optimal temperature by minimizing NLL."""
        best_t = 1.0
        best_loss = float('inf')

        for t in np.linspace(0.1, 5.0, 100):
            calibrated = self._apply_temp(logits, t)
            loss = log_loss(y_true, calibrated, labels=[0, 1])
            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = best_t
        self._fitted = True
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return self._apply_temp(logits, self.temperature)

    @staticmethod
    def _apply_temp(logits: np.ndarray, t: float) -> np.ndarray:
        scaled = logits / t
        # sigmoid for binary
        return 1.0 / (1.0 + np.exp(-scaled))


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

class CalibrationDiagnostics:
    """
    Computes calibration metrics and reliability diagram data.
    """

    @staticmethod
    def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10) -> Dict:
        """Compute reliability diagram data."""
        if not _SKL:
            raise ImportError("scikit-learn required")

        fraction_pos, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform')

        return {
            "fraction_positive": fraction_pos.tolist(),
            "mean_predicted": mean_predicted.tolist(),
            "n_bins": n_bins,
            "perfect_line": list(np.linspace(0, 1, n_bins)),
        }

    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                    n_bins: int = 10) -> float:
        """ECE: weighted average of per-bin calibration error."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
        return float(ece)

    @staticmethod
    def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                   n_bins: int = 10) -> float:
        """MCE: maximum per-bin calibration error."""
        bins = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            err = abs(y_true[mask].mean() - y_prob[mask].mean())
            mce = max(mce, err)
        return float(mce)

    @staticmethod
    def full_report(y_true: np.ndarray, y_prob: np.ndarray,
                    label: str = "Model") -> Dict:
        """Complete calibration report."""
        brier = brier_score_loss(y_true, y_prob) if _SKL else 0
        ece = CalibrationDiagnostics.expected_calibration_error(y_true, y_prob)
        mce = CalibrationDiagnostics.maximum_calibration_error(y_true, y_prob)
        curve = CalibrationDiagnostics.reliability_curve(y_true, y_prob) if _SKL else {}

        return {
            "label": label,
            "brier_score": round(float(brier), 6),
            "ece": round(ece, 6),
            "mce": round(mce, 6),
            "reliability_curve": curve,
            "mean_predicted": round(float(y_prob.mean()), 4),
            "mean_actual": round(float(y_true.mean()), 4),
            "calibration_gap": round(abs(float(y_prob.mean() - y_true.mean())), 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED CALIBRATOR
# ══════════════════════════════════════════════════════════════════════════════

class Calibrator:
    """
    Auto-selects best calibration method by comparing ECE on validation set.

    Usage
    -----
        cal = Calibrator()
        cal.fit(val_scores, val_labels)
        calibrated = cal.predict(test_scores)
        report = cal.report(test_labels, calibrated)
    """

    def __init__(self, method: str = "auto"):
        """method: "platt" | "isotonic" | "temperature" | "auto" """
        self.method = method
        self.calibrators = {}
        self.best_method = None
        self.diagnostics = CalibrationDiagnostics()

    def fit(self, scores: np.ndarray, y_true: np.ndarray):
        methods = {
            "platt": PlattScaling(),
            "isotonic": IsotonicCalibration(),
            "temperature": TemperatureScaling(),
        }

        if self.method != "auto":
            cal = methods[self.method]
            cal.fit(scores, y_true)
            self.calibrators[self.method] = cal
            self.best_method = self.method
            return self

        # auto: try all, pick lowest ECE
        best_ece = float('inf')
        for name, cal in methods.items():
            try:
                cal.fit(scores, y_true)
                calibrated = cal.predict(scores)
                ece = self.diagnostics.expected_calibration_error(y_true, calibrated)
                self.calibrators[name] = cal
                if ece < best_ece:
                    best_ece = ece
                    self.best_method = name
            except Exception:
                continue

        if self.best_method is None:
            raise RuntimeError("All calibration methods failed")

        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.calibrators[self.best_method].predict(scores)

    def report(self, y_true: np.ndarray, y_prob: np.ndarray = None,
               scores: np.ndarray = None) -> Dict:
        """Generate calibration report before/after calibration."""
        results = {}

        if scores is not None:
            results["before"] = self.diagnostics.full_report(y_true, scores, "Uncalibrated")
            calibrated = self.predict(scores)
            results["after"] = self.diagnostics.full_report(y_true, calibrated, "Calibrated")
        elif y_prob is not None:
            results["calibrated"] = self.diagnostics.full_report(y_true, y_prob, "Calibrated")

        results["best_method"] = self.best_method
        results["available_methods"] = list(self.calibrators.keys())
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  calibration.py — Probability Calibration Self-Test")
    print("=" * 65)

    if not _SKL:
        print("  scikit-learn required"); raise SystemExit(1)

    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 2, n)
    # simulate poorly calibrated scores
    raw_scores = np.clip(y_true * 0.7 + np.random.normal(0.15, 0.25, n), 0, 1)

    split = int(n * 0.6)
    s_tr, y_tr = raw_scores[:split], y_true[:split]
    s_te, y_te = raw_scores[split:], y_true[split:]

    print(f"\n  Train: {len(s_tr)}  Test: {len(s_te)}")

    # Before calibration
    before = CalibrationDiagnostics.full_report(y_te, s_te, "Raw")
    print(f"\n  Before Calibration:")
    print(f"    Brier: {before['brier_score']:.4f}  ECE: {before['ece']:.4f}")

    # Platt
    print("\n  Platt Scaling:")
    platt = PlattScaling()
    platt.fit(s_tr, y_tr)
    cal_platt = platt.predict(s_te)
    r_platt = CalibrationDiagnostics.full_report(y_te, cal_platt, "Platt")
    print(f"    Brier: {r_platt['brier_score']:.4f}  ECE: {r_platt['ece']:.4f}")
    print(f"    Params: {platt.get_params()}")

    # Isotonic
    print("\n  Isotonic Regression:")
    iso = IsotonicCalibration()
    iso.fit(s_tr, y_tr)
    cal_iso = iso.predict(s_te)
    r_iso = CalibrationDiagnostics.full_report(y_te, cal_iso, "Isotonic")
    print(f"    Brier: {r_iso['brier_score']:.4f}  ECE: {r_iso['ece']:.4f}")

    # Temperature
    print("\n  Temperature Scaling:")
    temp = TemperatureScaling()
    logits = np.log(s_tr / (1 - s_tr + 1e-8))
    temp.fit(logits, y_tr)
    logits_te = np.log(s_te / (1 - s_te + 1e-8))
    cal_temp = temp.predict(logits_te)
    print(f"    Temperature: {temp.temperature:.3f}")

    # Auto calibrator
    print("\n  Auto Calibrator:")
    cal = Calibrator(method="auto")
    cal.fit(s_tr, y_tr)
    report = cal.report(y_te, scores=s_te)
    print(f"    Best method: {report['best_method']}")
    print(f"    Before ECE: {report['before']['ece']:.4f}")
    print(f"    After ECE:  {report['after']['ece']:.4f}")

    print("\n  ✓  Calibration self-test complete.\n")
