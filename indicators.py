"""
╔══════════════════════════════════════════════════════════════════════╗
║                    indicators.py  — Feature Engine                  ║
║                                                                      ║
║  Column contract (matches data.py OHLCVCleaner + app.py exactly):  ║
║                                                                      ║
║  Trend      : SMA_N, EMA_N, MACD, MACD_Signal, MACD_Hist,          ║
║               ADX, DI_Pos, DI_Neg, Ichimoku_*,                      ║
║               DEMA_N, TEMA_N, HMA_N, KAMA, ALMA,                   ║
║               Parabolic_SAR, SuperTrend, VWMA_N                     ║
║                                                                      ║
║  Momentum   : RSI, %K, %D, CCI, Williams_%R, ROC_N,                ║
║               MFI, CMF, TRIX_N, DPO_N, UO, Aroon_Up, Aroon_Down,  ║
║               Aroon_Osc, TSI, KST, PPO, PVO                         ║
║                                                                      ║
║  Volatility : BB_Upper, BB_Mid, BB_Lower, BB_Width, BB_Pct,        ║
║               ATR, Keltner_Upper, Keltner_Lower,                    ║
║               Donchian_Upper, Donchian_Lower,                        ║
║               Historical_Vol, Chaikin_Vol, UI                       ║
║                                                                      ║
║  Volume     : OBV, VWAP, CMF, MFI, AD_Line, ADOSC,                 ║
║               Force_Index, EOM, NVI, PVI, VWMA_20                  ║
║                                                                      ║
║  Pattern    : Doji, Hammer, Shooting_Star, Engulf_Bull,             ║
║               Engulf_Bear, Morning_Star, Evening_Star                ║
║                                                                      ║
║  Stats      : Returns, Log_Returns, HL_Spread (from data.py),       ║
║               Realized_Vol_21, Z_Score_20, Price_Percentile_52w,   ║
║               Beta_SPY (if multi-ticker df), Skew_63, Kurt_63      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def _tr(df: pd.DataFrame) -> pd.Series:
    """True Range."""
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low']  - df['Close'].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Smoothed Moving Average (used in RSI, ATR, ADX)."""
    result = series.copy().astype(float)
    result.iloc[:period] = series.iloc[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(series)):
        result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — TREND
# ══════════════════════════════════════════════════════════════════════════════

def add_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df[f'SMA_{period}'] = _sma(df['Close'], period)
    return df


def add_ema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df[f'EMA_{period}'] = _ema(df['Close'], period)
    return df


def add_dema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Double EMA — faster trend with less lag."""
    e1 = _ema(df['Close'], period)
    df[f'DEMA_{period}'] = 2 * e1 - _ema(e1, period)
    return df


def add_tema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Triple EMA."""
    e1 = _ema(df['Close'], period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    df[f'TEMA_{period}'] = 3 * e1 - 3 * e2 + e3
    return df


def add_hma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Hull Moving Average — responsive to price changes."""
    half  = _sma(df['Close'], period // 2)
    full  = _sma(df['Close'], period)
    raw   = 2 * half - full
    df[f'HMA_{period}'] = _sma(raw, int(np.sqrt(period)))
    return df


def add_kama(df: pd.DataFrame, period: int = 10,
             fast: int = 2, slow: int = 30) -> pd.DataFrame:
    """Kaufman Adaptive Moving Average."""
    close   = df['Close'].values.astype(float)
    kama    = np.full(len(close), np.nan)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    kama[period - 1] = close[period - 1]
    for i in range(period, len(close)):
        direction = abs(close[i] - close[i - period])
        volatility = sum(abs(close[j] - close[j - 1])
                        for j in range(i - period + 1, i + 1))
        er  = direction / volatility if volatility != 0 else 0
        sc  = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    df['KAMA'] = kama
    return df


def add_vwma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume-Weighted Moving Average."""
    df[f'VWMA_{period}'] = (
        (df['Close'] * df['Volume']).rolling(period).sum()
        / df['Volume'].rolling(period).sum()
    )
    return df


def add_macd(df: pd.DataFrame,
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD — columns: MACD, MACD_Signal, MACD_Hist (app.py expects these exact names)."""
    ema_fast = _ema(df['Close'], fast)
    ema_slow = _ema(df['Close'], slow)
    df['MACD']        = ema_fast - ema_slow
    df['MACD_Signal'] = _ema(df['MACD'], signal)   # Note: Signal not signal
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    return df


def add_ppo(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    """Percentage Price Oscillator — normalised MACD."""
    ema_fast = _ema(df['Close'], fast)
    ema_slow = _ema(df['Close'], slow)
    df['PPO'] = (ema_fast - ema_slow) / ema_slow * 100
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX with Wilder smoothing.
    app.py expects: ADX, DI_Pos, DI_Neg (not +DI / -DI)."""
    tr    = _tr(df)
    up    = df['High'].diff()
    down  = -df['Low'].diff()

    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr14    = _rma(tr,                              period)
    plus_di  = 100 * _rma(pd.Series(plus_dm,  index=df.index), period) / atr14
    minus_di = 100 * _rma(pd.Series(minus_dm, index=df.index), period) / atr14

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))

    df['DI_Pos'] = plus_di     # app.py key
    df['DI_Neg'] = minus_di    # app.py key
    df['ADX']    = _rma(dx, period)
    return df


def add_ichimoku(df: pd.DataFrame,
                 tenkan: int = 9, kijun: int = 26,
                 senkou_b: int = 52, displacement: int = 26) -> pd.DataFrame:
    """Ichimoku Cloud — all 5 lines."""
    def mid(h, l, n): return (h.rolling(n).max() + l.rolling(n).min()) / 2
    t  = mid(df['High'], df['Low'], tenkan)
    k  = mid(df['High'], df['Low'], kijun)
    df['Ichimoku_Tenkan']   = t
    df['Ichimoku_Kijun']    = k
    df['Ichimoku_SpanA']    = ((t + k) / 2).shift(displacement)
    df['Ichimoku_SpanB']    = mid(df['High'], df['Low'], senkou_b).shift(displacement)
    df['Ichimoku_Chikou']   = df['Close'].shift(-displacement)
    return df


def add_parabolic_sar(df: pd.DataFrame,
                      af_start: float = 0.02,
                      af_max:   float = 0.20) -> pd.DataFrame:
    """Parabolic SAR."""
    high  = df['High'].values;  low  = df['Low'].values
    close = df['Close'].values
    n     = len(close)
    sar   = np.full(n, np.nan)
    ep    = close[0]; af = af_start
    bull  = True
    sar[0] = low[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[max(0, i - 2)])
            if low[i] < sar[i]:
                bull = False; sar[i] = ep; ep = low[i]; af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]; af = min(af + af_start, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[max(0, i - 2)])
            if high[i] > sar[i]:
                bull = True; sar[i] = ep; ep = high[i]; af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]; af = min(af + af_start, af_max)

    df['Parabolic_SAR'] = sar
    return df


def add_supertrend(df: pd.DataFrame,
                   period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend indicator."""
    hl2 = (df['High'] + df['Low']) / 2
    atr = _tr(df).rolling(period).mean()
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=float)
    supertrend.iloc[0] = lower.iloc[0]
    direction.iloc[0]  = 1

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(lower.iloc[i], supertrend.iloc[i - 1]) \
                                  if direction.iloc[i - 1] == 1 else lower.iloc[i]
            direction.iloc[i]  = 1
        else:
            supertrend.iloc[i] = min(upper.iloc[i], supertrend.iloc[i - 1]) \
                                  if direction.iloc[i - 1] == -1 else upper.iloc[i]
            direction.iloc[i]  = -1

    df['SuperTrend']     = supertrend
    df['SuperTrend_Dir'] = direction
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder RSI — app.py expects 'RSI' column."""
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_stochastic(df: pd.DataFrame,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator — app.py expects %K, %D."""
    low_min  = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    df['%D'] = df['%K'].rolling(d_period).mean()
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """CCI — app.py expects 'CCI' column."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    df['CCI'] = (tp - ma) / (0.015 * md + 1e-9)
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_max = df['High'].rolling(period).max()
    low_min  = df['Low'].rolling(period).min()
    df['Williams_%R'] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-9)
    return df


def add_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Rate of Change."""
    df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100
    return df


def add_trix(df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
    """TRIX — triple-smoothed EMA momentum."""
    e1 = _ema(df['Close'], period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    df[f'TRIX_{period}'] = e3.pct_change(1) * 100
    return df


def add_dpo(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Detrended Price Oscillator."""
    shift = period // 2 + 1
    df[f'DPO_{period}'] = (
        df['Close'].shift(shift) - _sma(df['Close'], period)
    )
    return df


def add_aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    roll_high = df['High'].rolling(period + 1)
    roll_low  = df['Low'].rolling(period + 1)
    df['Aroon_Up']   = roll_high.apply(lambda x: (np.argmax(x) / period) * 100, raw=True)
    df['Aroon_Down'] = roll_low.apply( lambda x: (np.argmin(x) / period) * 100, raw=True)
    df['Aroon_Osc']  = df['Aroon_Up'] - df['Aroon_Down']
    return df


def add_ultimate_oscillator(df: pd.DataFrame,
                             p1: int = 7, p2: int = 14, p3: int = 28) -> pd.DataFrame:
    """Ultimate Oscillator."""
    prev_close = df['Close'].shift(1)
    bp  = df['Close'] - pd.concat([df['Low'], prev_close], axis=1).min(axis=1)
    tr  = pd.concat([df['High'], prev_close], axis=1).max(axis=1) - \
          pd.concat([df['Low'],  prev_close], axis=1).min(axis=1)

    def _avg(bp, tr, n):
        return bp.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)

    a1 = _avg(bp, tr, p1); a2 = _avg(bp, tr, p2); a3 = _avg(bp, tr, p3)
    df['UO'] = 100 * (4 * a1 + 2 * a2 + a3) / 7
    return df


def add_tsi(df: pd.DataFrame,
            long: int = 25, short: int = 13) -> pd.DataFrame:
    """True Strength Index."""
    pc  = df['Close'].diff()
    ds  = _ema(_ema(pc, long), short)
    apc = _ema(_ema(pc.abs(), long), short)
    df['TSI'] = 100 * ds / apc.replace(0, np.nan)
    return df


def add_kst(df: pd.DataFrame) -> pd.DataFrame:
    """Know Sure Thing."""
    def roc(n): return df['Close'].pct_change(n) * 100
    df['KST'] = (
        _sma(roc(10), 10) * 1 + _sma(roc(13), 13) * 2 +
        _sma(roc(14), 14) * 3 + _sma(roc(15), 15) * 4
    )
    df['KST_Signal'] = _sma(df['KST'], 9)
    return df


def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Money Flow Index — volume-weighted RSI."""
    tp  = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    mfr = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    df['MFI'] = 100 - (100 / (1 + mfr))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def add_bollinger_bands(df: pd.DataFrame, period: int = 20,
                        std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands.
    app.py expects: BB_Upper, BB_Mid, BB_Lower, (optional) BB_Width, BB_Pct."""
    mid = _sma(df['Close'], period)
    std = df['Close'].rolling(period).std()
    df['BB_Upper'] = mid + std_dev * std
    df['BB_Mid']   = mid                    # app.py uses BB_Mid
    df['BB_Lower'] = mid - std_dev * std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / mid.replace(0, np.nan)
    df['BB_Pct']   = (df['Close'] - df['BB_Lower']) / \
                     (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range using Wilder smoothing — app.py expects 'ATR'."""
    df['ATR'] = _rma(_tr(df), period)
    return df


def add_keltner_channel(df: pd.DataFrame,
                        ema_period: int = 20,
                        atr_period: int = 10,
                        multiplier: float = 2.0) -> pd.DataFrame:
    """Keltner Channel."""
    basis = _ema(df['Close'], ema_period)
    atr   = _rma(_tr(df), atr_period)
    df['Keltner_Upper']  = basis + multiplier * atr
    df['Keltner_Middle'] = basis
    df['Keltner_Lower']  = basis - multiplier * atr
    return df


def add_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channel."""
    df['Donchian_Upper']  = df['High'].rolling(period).max()
    df['Donchian_Lower']  = df['Low'].rolling(period).min()
    df['Donchian_Middle'] = (df['Donchian_Upper'] + df['Donchian_Lower']) / 2
    return df


def add_historical_volatility(df: pd.DataFrame,
                               period: int = 21) -> pd.DataFrame:
    """Annualised close-to-close historical volatility."""
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    df['Historical_Vol'] = log_ret.rolling(period).std() * np.sqrt(252) * 100
    return df


def add_ulcer_index(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Ulcer Index — downside-only volatility."""
    roll_max = df['Close'].rolling(period).max()
    dd_pct   = ((df['Close'] - roll_max) / roll_max) * 100
    df['UI'] = np.sqrt((dd_pct ** 2).rolling(period).mean())
    return df


def add_chaikin_volatility(df: pd.DataFrame,
                           ema_period: int = 10,
                           roc_period: int = 10) -> pd.DataFrame:
    """Chaikin Volatility."""
    hl_ema = _ema(df['High'] - df['Low'], ema_period)
    df['Chaikin_Vol'] = hl_ema.pct_change(roc_period) * 100
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — VOLUME
# ══════════════════════════════════════════════════════════════════════════════

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume — vectorised (no loop). app.py expects 'OBV'."""
    direction = np.sign(df['Close'].diff()).fillna(0)
    df['OBV'] = (direction * df['Volume']).cumsum()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP (session cumulative). app.py expects 'VWAP'."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df


def add_ad_line(df: pd.DataFrame) -> pd.DataFrame:
    """Accumulation / Distribution Line."""
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
          (df['High'] - df['Low'] + 1e-9)
    df['AD_Line'] = (clv * df['Volume']).cumsum()
    return df


def add_adosc(df: pd.DataFrame,
              fast: int = 3, slow: int = 10) -> pd.DataFrame:
    """Chaikin A/D Oscillator."""
    clv   = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
            (df['High'] - df['Low'] + 1e-9)
    ad    = (clv * df['Volume']).cumsum()
    df['ADOSC'] = _ema(ad, fast) - _ema(ad, slow)
    return df


def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Chaikin Money Flow."""
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
          (df['High'] - df['Low'] + 1e-9)
    df['CMF'] = (clv * df['Volume']).rolling(period).sum() / \
                df['Volume'].rolling(period).sum().replace(0, np.nan)
    return df


def add_force_index(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """Force Index."""
    fi = df['Close'].diff() * df['Volume']
    df['Force_Index'] = _ema(fi, period)
    return df


def add_ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Ease of Movement."""
    mid_diff = ((df['High'] + df['Low']) / 2).diff()
    box_ratio = (df['Volume'] / 1e6) / (df['High'] - df['Low'] + 1e-9)
    eom = mid_diff / box_ratio
    df['EOM'] = eom.rolling(period).mean()
    return df


def add_nvi_pvi(df: pd.DataFrame) -> pd.DataFrame:
    """Negative / Positive Volume Index."""
    pct = df['Close'].pct_change().fillna(0)
    vol_change = df['Volume'].diff()
    nvi = [1000.0]; pvi = [1000.0]
    for i in range(1, len(df)):
        if vol_change.iloc[i] < 0:
            nvi.append(nvi[-1] + nvi[-1] * pct.iloc[i])
            pvi.append(pvi[-1])
        else:
            pvi.append(pvi[-1] + pvi[-1] * pct.iloc[i])
            nvi.append(nvi[-1])
    df['NVI'] = nvi
    df['PVI'] = pvi
    return df


def add_pvo(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    """Percentage Volume Oscillator."""
    ema_fast = _ema(df['Volume'].astype(float), fast)
    ema_slow = _ema(df['Volume'].astype(float), slow)
    df['PVO'] = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 100
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — CANDLESTICK PATTERNS  (binary 0/1 flags)
# ══════════════════════════════════════════════════════════════════════════════

def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    op = df['Open']; hi = df['High']; lo = df['Low']; cl = df['Close']
    body   = (cl - op).abs()
    candle = hi - lo + 1e-9

    # Doji
    df['Pat_Doji'] = (body / candle < 0.1).astype(int)

    # Hammer (bullish, in downtrend — simplified)
    lower_shadow = op.where(cl >= op, cl) - lo
    upper_shadow = hi - cl.where(cl >= op, op)
    df['Pat_Hammer'] = (
        (lower_shadow > 2 * body) & (upper_shadow < 0.2 * body) & (cl > op)
    ).astype(int)

    # Shooting Star (bearish)
    df['Pat_Shooting_Star'] = (
        (upper_shadow > 2 * body) & (lower_shadow < 0.2 * body) & (cl < op)
    ).astype(int)

    # Bullish Engulfing
    df['Pat_Engulf_Bull'] = (
        (cl.shift(1) < op.shift(1)) &
        (op < cl.shift(1)) & (cl > op.shift(1))
    ).astype(int)

    # Bearish Engulfing
    df['Pat_Engulf_Bear'] = (
        (cl.shift(1) > op.shift(1)) &
        (op > cl.shift(1)) & (cl < op.shift(1))
    ).astype(int)

    # Morning Star (simplified 3-bar)
    df['Pat_Morning_Star'] = (
        (cl.shift(2) < op.shift(2)) &
        (body.shift(1) < body.shift(2) * 0.3) &
        (cl > op) & (cl > (op.shift(2) + cl.shift(2)) / 2)
    ).astype(int)

    # Evening Star (simplified 3-bar)
    df['Pat_Evening_Star'] = (
        (cl.shift(2) > op.shift(2)) &
        (body.shift(1) < body.shift(2) * 0.3) &
        (cl < op) & (cl < (op.shift(2) + cl.shift(2)) / 2)
    ).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 6 — STATISTICAL / DERIVED FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived statistical features.
    data.py OHLCVCleaner already adds Returns, Log_Returns, HL_Spread —
    this function adds higher-order stats.
    """
    log_ret = np.log(df['Close'] / df['Close'].shift(1))

    # Realised volatility (21-day annualised)
    df['Realized_Vol_21'] = log_ret.rolling(21).std() * np.sqrt(252)

    # Z-score of close vs 20-day mean
    mu = df['Close'].rolling(20).mean()
    sg = df['Close'].rolling(20).std()
    df['Z_Score_20'] = (df['Close'] - mu) / sg.replace(0, np.nan)

    # 52-week price percentile
    df['Price_Pct_52w'] = df['Close'].rolling(252).rank(pct=True)

    # Rolling skew / kurtosis of returns (63-day)
    df['Skew_63']  = log_ret.rolling(63).skew()
    df['Kurt_63']  = log_ret.rolling(63).kurt()

    # Gap (overnight jump)
    df['Gap_Pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100

    # Price distance from VWAP (if already computed)
    if 'VWAP' in df.columns:
        df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP'] * 100

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION  — called by app.py and data.py
# ══════════════════════════════════════════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame,
                       sma_periods:  list = None,
                       ema_periods:  list = None,
                       include_slow: bool = True) -> pd.DataFrame:
    """
    Full indicator pipeline.

    Parameters
    ----------
    df            : OHLCV DataFrame from DataPipeline / OHLCVCleaner
    sma_periods   : list of SMA periods (default: [20, 50, 200])
    ema_periods   : list of EMA periods (default: [9, 21, 50])
    include_slow  : if False, skip Ichimoku, KAMA, Parabolic SAR, patterns
                    (useful for fast intraday DataFrames)

    Returns
    -------
    df with all indicator columns appended in-place.
    """
    df = df.copy()

    sma_periods = sma_periods or [20, 50, 200]
    ema_periods = ema_periods or [9, 21, 50]

    # ── TREND ────────────────────────────────────────────────────────────────
    for p in sma_periods:
        df = add_sma(df, p)
    for p in ema_periods:
        df = add_ema(df, p)

    df = add_macd(df)          # MACD, MACD_Signal, MACD_Hist
    df = add_adx(df)           # ADX, DI_Pos, DI_Neg
    df = add_ppo(df)
    df = add_dema(df, 20)
    df = add_tema(df, 20)
    df = add_hma(df, 20)
    df = add_vwma(df, 20)

    if include_slow:
        df = add_kama(df)
        df = add_ichimoku(df)
        df = add_parabolic_sar(df)
        df = add_supertrend(df)

    # ── MOMENTUM ─────────────────────────────────────────────────────────────
    df = add_rsi(df)           # RSI
    df = add_stochastic(df)    # %K, %D
    df = add_cci(df)           # CCI
    df = add_williams_r(df)
    df = add_roc(df, 10)
    df = add_roc(df, 20)
    df = add_trix(df, 15)
    df = add_dpo(df, 20)
    df = add_aroon(df, 25)
    df = add_ultimate_oscillator(df)
    df = add_tsi(df)
    df = add_kst(df)
    df = add_mfi(df)           # MFI

    # ── VOLATILITY ───────────────────────────────────────────────────────────
    df = add_bollinger_bands(df)   # BB_Upper, BB_Mid, BB_Lower, BB_Width, BB_Pct
    df = add_atr(df)               # ATR
    df = add_keltner_channel(df)
    df = add_donchian_channel(df)
    df = add_historical_volatility(df)
    df = add_ulcer_index(df)
    df = add_chaikin_volatility(df)

    # ── VOLUME ───────────────────────────────────────────────────────────────
    df = add_obv(df)           # OBV
    df = add_vwap(df)          # VWAP
    df = add_ad_line(df)
    df = add_adosc(df)
    df = add_cmf(df)           # CMF
    df = add_force_index(df)
    df = add_ease_of_movement(df)
    df = add_nvi_pvi(df)
    df = add_pvo(df)

    # ── PATTERNS ─────────────────────────────────────────────────────────────
    if include_slow:
        df = add_candlestick_patterns(df)

    # ── STATISTICAL ──────────────────────────────────────────────────────────
    df = add_statistical_features(df)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN REGISTRY  — useful for dashboard feature-selector dropdowns
# ══════════════════════════════════════════════════════════════════════════════

INDICATOR_GROUPS = {
    "Trend": [
        "SMA_20","SMA_50","SMA_200","EMA_9","EMA_21","EMA_50",
        "MACD","MACD_Signal","MACD_Hist","ADX","DI_Pos","DI_Neg",
        "DEMA_20","TEMA_20","HMA_20","KAMA","VWMA_20","PPO",
        "Parabolic_SAR","SuperTrend","SuperTrend_Dir",
        "Ichimoku_Tenkan","Ichimoku_Kijun","Ichimoku_SpanA","Ichimoku_SpanB",
    ],
    "Momentum": [
        "RSI","%K","%D","CCI","Williams_%R",
        "ROC_10","ROC_20","TRIX_15","DPO_20",
        "Aroon_Up","Aroon_Down","Aroon_Osc",
        "UO","TSI","KST","KST_Signal","MFI",
    ],
    "Volatility": [
        "BB_Upper","BB_Mid","BB_Lower","BB_Width","BB_Pct","ATR",
        "Keltner_Upper","Keltner_Middle","Keltner_Lower",
        "Donchian_Upper","Donchian_Middle","Donchian_Lower",
        "Historical_Vol","UI","Chaikin_Vol",
    ],
    "Volume": [
        "OBV","VWAP","AD_Line","ADOSC","CMF",
        "Force_Index","EOM","NVI","PVI","PVO","MFI",
    ],
    "Patterns": [
        "Pat_Doji","Pat_Hammer","Pat_Shooting_Star",
        "Pat_Engulf_Bull","Pat_Engulf_Bear",
        "Pat_Morning_Star","Pat_Evening_Star",
    ],
    "Statistical": [
        "Returns","Log_Returns","HL_Spread",
        "Realized_Vol_21","Z_Score_20","Price_Pct_52w",
        "Skew_63","Kurt_63","Gap_Pct","Price_vs_VWAP",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Works standalone — generates synthetic data + runs full pipeline
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 1000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    df_test = pd.DataFrame({
        "Open":   close * (1 + np.random.normal(0, 0.002, n)),
        "High":   close * (1 + np.abs(np.random.normal(0, 0.008, n))),
        "Low":    close * (1 - np.abs(np.random.normal(0, 0.008, n))),
        "Close":  close,
        "Volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=idx)

    print("Running add_all_indicators …")
    out = add_all_indicators(df_test)
    print(f"Input cols  : 5")
    print(f"Output cols : {len(out.columns)}")
    print(f"NaN rows    : {out.isna().any(axis=1).sum()}")
    print("\nColumn groups:")
    for grp, cols in INDICATOR_GROUPS.items():
        present = [c for c in cols if c in out.columns]
        print(f"  {grp:<14}: {len(present)}/{len(cols)} present")
    print("\nLast row sample:")
    sample_cols = ["Close","RSI","MACD","MACD_Signal","ADX","DI_Pos","DI_Neg",
                   "BB_Upper","BB_Mid","BB_Lower","ATR","OBV","VWAP","CCI"]
    print(out[sample_cols].tail(1).T.to_string())