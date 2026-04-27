"""
NeuroTrade — Utility Indicators
utils.py — Technical indicator functions (mirrors indicators.py contract).

Column contract expected by app.py:
    SMA_{period}, EMA_{period}
    MACD, MACD_Signal, MACD_Hist
    ADX, DI_Pos, DI_Neg
    RSI
    Stoch_K, Stoch_D
    CCI
    BB_Upper, BB_Mid, BB_Lower, BB_Width
    ATR
    OBV, VWAP
    Returns, Log_Returns, HL_Spread
    Realized_Vol_21, Z_Score_20
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  TREND INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_sma(df, period=20):
    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    return df


def add_ema(df, period=20):
    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df


def add_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def add_adx(df, period=14):
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['+DM'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0), 0
    )
    df['-DM'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0), 0
    )

    TRn = df['TR'].rolling(period).mean()
    plusDMn = df['+DM'].rolling(period).mean()
    minusDMn = df['-DM'].rolling(period).mean()

    df['DI_Pos'] = (plusDMn / TRn) * 100
    df['DI_Neg'] = (minusDMn / TRn) * 100
    df['+DI'] = df['DI_Pos']
    df['-DI'] = df['DI_Neg']

    df['DX'] = (abs(df['DI_Pos'] - df['DI_Neg']) /
                (df['DI_Pos'] + df['DI_Neg'] + 1e-9)) * 100
    df['ADX'] = df['DX'].rolling(period).mean()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MOMENTUM INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_stochastic(df, period=14):
    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-9))
    df['Stoch_K'] = k
    df['Stoch_D'] = k.rolling(3).mean()
    df['%K'] = df['Stoch_K']
    df['%D'] = df['Stoch_D']
    return df


def add_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    df['CCI'] = (tp - ma) / (0.015 * md + 1e-9)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  VOLATILITY INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_bollinger_bands(df, period=20):
    df['BB_Mid'] = df['Close'].rolling(period).mean()
    df['BB_Middle'] = df['BB_Mid']
    std = df['Close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * std)
    df['BB_Lower'] = df['BB_Mid'] - (2 * std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Mid'] + 1e-9)
    return df


def add_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(period).mean()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  VOLUME INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    df['OBV'] = (direction * df['Volume']).cumsum()
    return df


def add_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED FEATURES (for ML/DL/Quantum models)
# ══════════════════════════════════════════════════════════════════════════════

def add_derived_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['Realized_Vol_21'] = df['Returns'].rolling(21).std() * np.sqrt(252)
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['Z_Score_20'] = (df['Close'] - sma_20) / (std_20 + 1e-9)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION — called as add_all_indicators(df) from app.py
# ══════════════════════════════════════════════════════════════════════════════

def add_all_indicators(df):
    """
    Enrich an OHLCV DataFrame with every supported indicator.
    Column contract expected by app.py:
        SMA_20, EMA_20
        MACD, MACD_Signal, MACD_Hist
        ADX, DI_Pos, DI_Neg
        RSI
        Stoch_K, Stoch_D
        CCI
        BB_Upper, BB_Mid, BB_Lower, BB_Width
        ATR
        OBV, VWAP
        Returns, Log_Returns, HL_Spread
        Realized_Vol_21, Z_Score_20
    """
    df = df.copy()
    df = add_sma(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_adx(df)
    df = add_rsi(df)
    df = add_stochastic(df)
    df = add_cci(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_derived_features(df)
    return df
