"""
NeuroTrade — Quantum-Enhanced Models
quantum_model.py — VQC and QCNN quantum circuits for market prediction.

Algorithms:
  - VQC  (Variational Quantum Classifier): direction prediction (up/down)
  - QCNN (Quantum Convolutional Neural Network): hierarchical feature extraction

Backend: PennyLane default.qubit simulator (CPU, no hardware needed).
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ── PennyLane ────────────────────────────────────────────────────────────────
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    _QML = True
except ImportError:
    _QML = False
    print("[quantum_model] PennyLane not found — pip install pennylane")

# ── Scikit-learn ─────────────────────────────────────────────────────────────
try:
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score
    _SKL = True
except ImportError:
    _SKL = False
    print("[quantum_model] scikit-learn not found — pip install scikit-learn")

# ── Backtester ───────────────────────────────────────────────────────────────
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
class QuantumConfig:
    """Configuration for quantum models."""
    n_qubits:         int   = 8
    n_layers:         int   = 4
    backend:          str   = "default.qubit"
    learning_rate:    float = 0.01
    epochs:           int   = 60
    batch_size:       int   = 32
    patience:         int   = 10
    train_ratio:      float = 0.70
    val_ratio:        float = 0.15
    forecast_horizon: int   = 5
    # QCNN specific
    qcnn_conv_layers: int   = 2
    # signal
    long_threshold:   float = 0.55
    short_threshold:  float = 0.45

CURATED_FEATURES = [
    "Returns", "Log_Returns", "HL_Spread", "RSI", "MACD", "MACD_Hist",
    "ATR", "ADX", "BB_Width", "CCI", "%K", "Z_Score_20",
    "Realized_Vol_21", "OBV", "VWAP",
]


# ══════════════════════════════════════════════════════════════════════════════
#  QUANTUM FEATURE ENGINEER
# ══════════════════════════════════════════════════════════════════════════════

class QuantumFeatureEngineer:
    """
    Prepares features for quantum circuits:
    1. Select + clean numeric features
    2. Scale to [0, 1] range
    3. PCA reduce to n_qubits dimensions
    4. Map to [0, π] for angle encoding
    """

    def __init__(self, cfg: QuantumConfig = None):
        self.cfg = cfg or QuantumConfig()
        self.scaler = None
        self.pca = None
        self.feature_cols = []

    def build(self, df: pd.DataFrame, fit: bool = True):
        """Returns X (n, n_qubits), y (n,) direction labels 0/1."""
        data = self._select(df)

        # target: future direction
        future_ret = df["Close"].pct_change(self.cfg.forecast_horizon).shift(
            -self.cfg.forecast_horizon)
        target = (future_ret > 0).astype(int)

        valid = data.notna().all(axis=1) & target.notna()
        data = data.loc[valid]
        target = target.loc[valid]

        if len(data) == 0:
            raise ValueError("Not enough valid data points to train the model. Try a longer timeframe.")

        self.feature_cols = list(data.columns)
        values = data.values.astype(np.float64)

        if fit:
            self.scaler = StandardScaler()
            values = self.scaler.fit_transform(values)
            n_comp = min(self.cfg.n_qubits, values.shape[1])
            self.pca = PCA(n_components=n_comp)
            values = self.pca.fit_transform(values)
        else:
            values = self.scaler.transform(values)
            values = self.pca.transform(values)

        # normalize to [0, π] for angle encoding
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-8] = 1.0
        values = (values - mins) / ranges * np.pi

        return values.astype(np.float64), target.values.astype(np.int64)

    def transform(self, df: pd.DataFrame):
        data = self._select(df)
        data = data.dropna()
        for c in self.feature_cols:
            if c not in data.columns:
                data[c] = 0.0
        data = data[self.feature_cols]
        values = self.scaler.transform(data.values.astype(np.float64))
        values = self.pca.transform(values)
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-8] = 1.0
        values = (values - mins) / ranges * np.pi
        return values.astype(np.float64)

    def _select(self, df):
        cols = [c for c in CURATED_FEATURES if c in df.columns]
        if not cols:
            cols = df.select_dtypes(include=[np.number]).columns[:15].tolist()
        return df[cols].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  VQC — VARIATIONAL QUANTUM CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class VQCModel:
    """
    Variational Quantum Classifier for market direction prediction.

    Circuit:
        Angle Encoding → [RY + RZ rotations + CNOT entanglement] × n_layers → Measurement

    Trained with parameter-shift gradient descent.
    """

    def __init__(self, cfg: QuantumConfig = None):
        if not _QML:
            raise ImportError("PennyLane required — pip install pennylane")
        self.cfg = cfg or QuantumConfig()
        self.n_qubits = cfg.n_qubits if cfg else 8
        self.n_layers = cfg.n_layers if cfg else 4
        self.dev = qml.device(self.cfg.backend, wires=self.n_qubits)
        self.weights = None
        self.bias = None
        self._circuit = None
        self._build()

    def _build(self):
        n_q = self.n_qubits
        n_l = self.n_layers
        dev = self.dev

        @qml.qnode(dev, interface="autograd")
        def circuit(inputs, weights):
            # Angle encoding
            for i in range(n_q):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for layer in range(n_l):
                for i in range(n_q):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # entanglement
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_q - 1, 0])  # ring connectivity

            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit
        # init weights
        self.weights = pnp.random.uniform(
            -np.pi, np.pi, (n_l, n_q, 2), requires_grad=True)
        self.bias = pnp.array(0.0, requires_grad=True)

    def _predict_single(self, x):
        raw = self._circuit(x, self.weights) + self.bias
        return (pnp.tanh(raw) + 1) / 2  # map to [0, 1]

    def predict_proba(self, X):
        probs = []
        for x in X:
            p = float(self._predict_single(x))
            probs.append(p)
        return np.array(probs)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, verbose=True):
        cfg = self.cfg
        opt = qml.AdamOptimizer(stepsize=cfg.learning_rate)

        def cost_fn(weights, bias, X_batch, y_batch):
            loss = 0.0
            for x, y in zip(X_batch, y_batch):
                raw = self._circuit(x, weights) + bias
                pred = (pnp.tanh(raw) + 1) / 2
                loss += (pred - y) ** 2
            return loss / len(X_batch)

        best_val_loss = float("inf")
        best_weights = None
        best_bias = None
        patience_ctr = 0

        n = len(X_tr)
        bs = min(cfg.batch_size, n)

        for epoch in range(1, cfg.epochs + 1):
            # mini-batch
            idx = np.random.permutation(n)[:bs]
            X_b = X_tr[idx]
            y_b = y_tr[idx].astype(np.float64)

            self.weights, self.bias, _, _ = opt.step(
                cost_fn, self.weights, self.bias, X_b, y_b)

            if epoch % 5 == 0 or epoch == 1:
                # validation
                if X_va is not None:
                    val_preds = self.predict(X_va)
                    val_acc = accuracy_score(y_va, val_preds)
                    val_loss = 1.0 - val_acc

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self.weights.copy()
                        best_bias = float(self.bias)
                        patience_ctr = 0
                    else:
                        patience_ctr += 1

                    if verbose and (epoch % 10 == 0 or epoch == 1):
                        tr_acc = accuracy_score(y_tr[:bs], self.predict(X_tr[:bs]))
                        print(f"    Epoch {epoch:3d}/{cfg.epochs}  "
                              f"train_acc={tr_acc:.3f}  val_acc={val_acc:.3f}")

                    if patience_ctr >= cfg.patience:
                        if verbose:
                            print(f"    Early stop at epoch {epoch}")
                        break

        if best_weights is not None:
            self.weights = best_weights
            self.bias = pnp.array(best_bias, requires_grad=True)

        return self


# ══════════════════════════════════════════════════════════════════════════════
#  QCNN — QUANTUM CONVOLUTIONAL NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class QCNNModel:
    """
    Quantum CNN with convolution + pooling layers.

    Circuit:
        Encoding → [Conv(2-qubit unitaries) + Pool(measure+conditional)] × layers
        → Measurement of remaining qubits → Classical FC head

    Requires n_qubits to be a power of 2 for clean pooling.
    """

    def __init__(self, cfg: QuantumConfig = None):
        if not _QML:
            raise ImportError("PennyLane required — pip install pennylane")
        self.cfg = cfg or QuantumConfig()
        # ensure power of 2
        self.n_qubits = 2 ** int(np.ceil(np.log2(cfg.n_qubits if cfg else 8)))
        self.dev = qml.device(self.cfg.backend, wires=self.n_qubits)
        self.weights = None
        self.classical_weights = None
        self.classical_bias = None
        self._circuit = None
        self._n_conv_layers = min(cfg.qcnn_conv_layers if cfg else 2,
                                   int(np.log2(self.n_qubits)))
        self._build()

    def _build(self):
        n_q = self.n_qubits
        n_cl = self._n_conv_layers
        dev = self.dev

        @qml.qnode(dev, interface="autograd")
        def circuit(inputs, conv_weights):
            # Angle encoding (pad if needed)
            for i in range(n_q):
                idx = i % len(inputs)
                qml.RY(inputs[idx], wires=i)
                qml.RZ(inputs[idx] * 0.5, wires=i)

            active_wires = list(range(n_q))
            w_idx = 0

            for layer in range(n_cl):
                n_active = len(active_wires)
                if n_active < 2:
                    break

                # Convolution: parameterized 2-qubit gates on adjacent pairs
                for j in range(0, n_active - 1, 2):
                    w0 = active_wires[j]
                    w1 = active_wires[j + 1]
                    qml.RY(conv_weights[w_idx], wires=w0)
                    qml.RY(conv_weights[w_idx + 1], wires=w1)
                    qml.CNOT(wires=[w0, w1])
                    qml.RZ(conv_weights[w_idx + 2], wires=w1)
                    qml.CNOT(wires=[w1, w0])
                    w_idx += 3

                # Pooling: keep every other qubit
                active_wires = active_wires[::2]

            # Measure all remaining active qubits
            return [qml.expval(qml.PauliZ(w)) for w in active_wires]

        self._circuit = circuit
        self._active_final = max(1, n_q // (2 ** n_cl))

        # count total conv weights needed
        total_w = 0
        n_active = n_q
        for _ in range(n_cl):
            n_pairs = n_active // 2
            total_w += n_pairs * 3
            n_active = n_active // 2

        self.weights = pnp.random.uniform(
            -np.pi, np.pi, (total_w,), requires_grad=True)

        # classical head
        self.classical_weights = pnp.random.uniform(
            -0.5, 0.5, (self._active_final,), requires_grad=True)
        self.classical_bias = pnp.array(0.0, requires_grad=True)

    def _predict_single(self, x):
        measurements = self._circuit(x, self.weights)
        if isinstance(measurements, (list, tuple)):
            measurements = pnp.array(measurements)
        else:
            measurements = pnp.array([measurements])
        raw = pnp.dot(measurements[:len(self.classical_weights)],
                       self.classical_weights) + self.classical_bias
        return (pnp.tanh(raw) + 1) / 2

    def predict_proba(self, X):
        probs = []
        for x in X:
            p = float(self._predict_single(x))
            probs.append(p)
        return np.array(probs)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, verbose=True):
        cfg = self.cfg
        opt = qml.AdamOptimizer(stepsize=cfg.learning_rate)

        def cost_fn(weights, c_weights, c_bias, X_batch, y_batch):
            self.weights = weights
            self.classical_weights = c_weights
            self.classical_bias = c_bias
            loss = 0.0
            for x, y in zip(X_batch, y_batch):
                pred = self._predict_single(x)
                loss += (pred - y) ** 2
            return loss / len(X_batch)

        best_val_loss = float("inf")
        patience_ctr = 0
        best_state = None
        n = len(X_tr)
        bs = min(cfg.batch_size, n)

        for epoch in range(1, cfg.epochs + 1):
            idx = np.random.permutation(n)[:bs]
            X_b = X_tr[idx]
            y_b = y_tr[idx].astype(np.float64)

            self.weights, self.classical_weights, self.classical_bias, _, _ = opt.step(
                cost_fn, self.weights, self.classical_weights,
                self.classical_bias, X_b, y_b)

            if epoch % 5 == 0 or epoch == 1:
                if X_va is not None:
                    val_preds = self.predict(X_va)
                    val_acc = accuracy_score(y_va, val_preds)
                    val_loss = 1.0 - val_acc

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = (self.weights.copy(),
                                     self.classical_weights.copy(),
                                     float(self.classical_bias))
                        patience_ctr = 0
                    else:
                        patience_ctr += 1

                    if verbose and (epoch % 10 == 0 or epoch == 1):
                        tr_acc = accuracy_score(y_tr[:bs], self.predict(X_tr[:bs]))
                        print(f"    Epoch {epoch:3d}/{cfg.epochs}  "
                              f"train_acc={tr_acc:.3f}  val_acc={val_acc:.3f}")

                    if patience_ctr >= cfg.patience:
                        if verbose:
                            print(f"    Early stop at epoch {epoch}")
                        break

        if best_state:
            self.weights = best_state[0]
            self.classical_weights = best_state[1]
            self.classical_bias = pnp.array(best_state[2], requires_grad=True)

        return self


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED QUANTUM TRAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumTrainingResult:
    model_name: str
    metrics:    Dict
    duration:   float
    n_qubits:   int
    n_layers:   int


class QuantumTrainer:
    """
    Unified trainer for VQC and QCNN.

    Usage
    -----
        trainer = QuantumTrainer(cfg)
        results = trainer.train_all(df)
        pred    = trainer.predict(df)
    """

    def __init__(self, cfg: QuantumConfig = None):
        self.cfg = cfg or QuantumConfig()
        self.eng = QuantumFeatureEngineer(self.cfg)
        self.models: Dict[str, object] = {}
        self.results: Dict[str, QuantumTrainingResult] = {}
        self.weights: Dict[str, float] = {}
        self._trained = False

    def train_all(self, df: pd.DataFrame,
                  progress_cb: Optional[Callable] = None) -> Dict:
        if not _QML:
            raise ImportError("PennyLane required — pip install pennylane")
        if not _SKL:
            raise ImportError("scikit-learn required — pip install scikit-learn")

        cfg = self.cfg
        X, y = self.eng.build(df, fit=True)
        n = len(X)
        n_tr = int(n * cfg.train_ratio)
        n_va = int(n * cfg.val_ratio)

        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_va, y_va = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
        X_te, y_te = X[n_tr+n_va:], y[n_tr+n_va:]

        all_results = {}
        scores = {}

        for name, ModelCls in [("vqc", VQCModel), ("qcnn", QCNNModel)]:
            print(f"\n  Training {name.upper()} ({cfg.n_qubits} qubits, "
                  f"{cfg.n_layers} layers)...")
            t0 = time.time()

            model = ModelCls(cfg)
            model.fit(X_tr, y_tr, X_va, y_va, verbose=True)
            dur = time.time() - t0

            # evaluate
            te_preds = model.predict(X_te)
            acc = accuracy_score(y_te, te_preds) * 100
            f1 = f1_score(y_te, te_preds, zero_division=0) * 100

            metrics = {"accuracy": round(acc, 2), "f1": round(f1, 2),
                       "n_test": len(y_te)}
            scores[name] = acc
            print(f"    Test Accuracy: {acc:.2f}%  F1: {f1:.2f}%  ({dur:.1f}s)")

            result = QuantumTrainingResult(name, metrics, dur,
                                            cfg.n_qubits, cfg.n_layers)
            self.models[name] = model
            self.results[name] = result
            all_results[name] = result

            if progress_cb:
                progress_cb(name, metrics)

        # weights
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
            per_model[name] = model.predict_proba(X)

        combo = np.zeros(len(X))
        total_w = sum(self.weights[k] for k in per_model)
        for name, probs in per_model.items():
            combo += (self.weights[name] / total_w) * probs

        return {"ensemble": combo, "per_model": per_model, "weights": self.weights}

    def predict_latest(self, df: pd.DataFrame) -> float:
        result = self.predict(df)
        if len(result["ensemble"]) == 0:
            return 0.5
        return float(result["ensemble"][-1])

    def get_signal(self, df: pd.DataFrame, i: int = -1):
        if not self._trained:
            return None
        end = i if i >= 0 else len(df)
        start = max(0, end - 120)
        subset = df.iloc[start:end].copy()
        if len(subset) < 50:
            return None
        pred = self.predict_latest(subset)
        if pred > self.cfg.long_threshold:
            return PositionSide.LONG
        elif pred < self.cfg.short_threshold:
            return PositionSide.SHORT
        return PositionSide.FLAT


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class QuantumSignalGenerator:
    """Backtester-compatible signal wrapper for quantum models."""

    def __init__(self, trainer: QuantumTrainer, threshold: float = 0.55):
        self.trainer = trainer
        self.threshold = threshold
        self._cache_bar = -1
        self._cache_pred = None

    def make_signal_func(self, df: pd.DataFrame):
        engine = self

        def signal_func(df_bt: pd.DataFrame, i: int, **kwargs):
            if i < 60:
                return None
            if engine._cache_bar == -1 or (i - engine._cache_bar) >= 5:
                try:
                    sub = df_bt.iloc[:i+1]
                    engine._cache_pred = engine.trainer.predict_latest(sub)
                    engine._cache_bar = i
                except Exception:
                    return None

            pred = engine._cache_pred
            if pred is None:
                return None
            if pred > engine.threshold:
                return PositionSide.LONG
            elif pred < (1.0 - engine.threshold):
                return PositionSide.SHORT
            return PositionSide.FLAT

        return signal_func


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    df = pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": np.random.randint(1e5, 1e6, n).astype(float),
    }, index=dates)
    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["RSI"] = 50 + np.cumsum(np.random.normal(0, 1, n))
    df["RSI"] = df["RSI"].clip(10, 90)
    e12 = df["Close"].ewm(span=12).mean()
    e26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = e12 - e26
    df["MACD_Hist"] = df["MACD"] - df["MACD"].ewm(span=9).mean()
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ADX"] = 25 + np.random.normal(0, 8, n)
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Width"] = (4 * std) / (mid + 1e-9)
    df["CCI"] = np.random.normal(0, 50, n)
    df["%K"] = 50 + np.random.normal(0, 20, n)
    df["Z_Score_20"] = (df["Close"] - mid) / (std + 1e-9)
    df["Realized_Vol_21"] = df["Log_Returns"].rolling(21).std() * np.sqrt(252)
    df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df.dropna()


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  quantum_model.py — Quantum ML Engine Self-Test")
    print("=" * 65)

    if not _QML:
        print("\n  PennyLane not installed. Run: pip install pennylane")
        raise SystemExit(1)

    df = _generate_synthetic_df(300)
    print(f"\n  Synthetic df: {len(df)} rows × {len(df.columns)} cols")

    cfg = QuantumConfig(n_qubits=4, n_layers=2, epochs=20,
                         batch_size=16, patience=8)

    trainer = QuantumTrainer(cfg)
    results = trainer.train_all(df)

    for name, res in results.items():
        print(f"\n  {name.upper()}: {res.metrics}  ({res.duration:.1f}s)")

    pred = trainer.predict_latest(df)
    sig = trainer.get_signal(df)
    print(f"\n  Latest prediction: {pred:.4f}")
    print(f"  Signal: {sig}")
    print("\n  OK  All quantum models passed self-test.\n")
