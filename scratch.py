import pandas as pd
from dl_models import DLConfig, DLTrainer, ModelType, _generate_synthetic_df

df = _generate_synthetic_df(100)
cfg = DLConfig(epochs=1, seq_len=10, forecast_horizon=1)
trainer = DLTrainer(cfg)

try:
    print("Testing DL (LSTM)...")
    trainer.train(df, ModelType.LSTM)
    print("DL Success")
except Exception as e:
    import traceback
    traceback.print_exc()

from ml_models import MLConfig, MLTrainer
try:
    print("Testing ML...")
    cfg2 = MLConfig(forecast_horizon=1)
    trainer2 = MLTrainer(cfg2)
    trainer2.train_all(df)
    print("ML Success")
except Exception as e:
    import traceback
    traceback.print_exc()

from quantum_models import QuantumConfig, QuantumTrainer
try:
    print("Testing Quantum...")
    cfg3 = QuantumConfig(n_qubits=2, n_layers=1)
    trainer3 = QuantumTrainer(cfg3)
    trainer3.train_all(df)
    print("Quantum Success")
except Exception as e:
    import traceback
    traceback.print_exc()
