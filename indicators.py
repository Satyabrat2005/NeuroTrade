import pandas as pd
import numpy as np



#  TREND INDICATORS


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
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def add_adx(df, period=14):
    df['TR'] = np.maximum(df['High'] - df['Low'],
                np.maximum(abs(df['High'] - df['Close'].shift(1)),
                           abs(df['Low'] - df['Close'].shift(1))))

    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > 
                        (df['Low'].shift(1) - df['Low']),
                        np.maximum(df['High'] - df['High'].shift(1), 0), 0)

    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > 
                        (df['High'] - df['High'].shift(1)),
                        np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    TRn = df['TR'].rolling(period).mean()
    plusDMn = df['+DM'].rolling(period).mean()
    minusDMn = df['-DM'].rolling(period).mean()

    df['+DI'] = (plusDMn / TRn) * 100
    df['-DI'] = (minusDMn / TRn) * 100

    df['DX'] = (abs(df['+DI'] - df['-DI']) / 
                (df['+DI'] + df['-DI'])) * 100

    df['ADX'] = df['DX'].rolling(period).mean()

    return df


#  MOMENTUM INDICATORS


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

    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(3).mean()

    return df


def add_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()

    df['CCI'] = (tp - ma) / (0.015 * md)
    return df


# VOLATILITY INDICATORS


def add_bollinger_bands(df, period=20):
    df['BB_Middle'] = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()

    df['BB_Upper'] = df['BB_Middle'] + (2 * std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * std)

    return df


def add_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    df['ATR'] = true_range.rolling(period).mean()
    return df


#  VOLUME INDICATORS


def add_obv(df):
    obv = [0]

    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])

    df['OBV'] = obv
    return df


def add_vwap(df):
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df


# MASTER FUNCTION


def add_all_indicators(df):
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

    return df
