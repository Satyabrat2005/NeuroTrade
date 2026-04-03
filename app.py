"""
NeuroTrade — AI-Powered Trading Assistant
app.py — Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Try importing custom utils, fall back to built-in 
try:
    from utils import add_all_indicators
    CUSTOM_UTILS = True
except ImportError:
    CUSTOM_UTILS = False


#  INDICATOR ENGINE  (self-contained fallback — mirrors utils.py contract)


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_bollinger(series: pd.Series, period=20, std_dev=2):
    mid = compute_sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d

def compute_atr(high, low, close, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_adx(high, low, close, period=14):
    tr = compute_atr(high, low, close, period)
    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr = tr.rolling(period).mean()
    pos_di = 100 * pd.Series(pos_dm, index=close.index).rolling(period).mean() / atr
    neg_di = 100 * pd.Series(neg_dm, index=close.index).rolling(period).mean() / atr
    dx = (100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-9))
    adx = dx.rolling(period).mean()
    return adx, pos_di, neg_di

def compute_cci(high, low, close, period=20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad + 1e-9)

def compute_obv(close, volume) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def compute_vwap(high, low, close, volume) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()

def add_all_indicators_builtin(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df[f"SMA_{params['sma']}"] = compute_sma(c, params["sma"])
    df[f"EMA_{params['ema']}"] = compute_ema(c, params["ema"])

    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(c)

    df["RSI"] = compute_rsi(c, params["rsi"])

    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(c, params["bb"])

    df["ATR"] = compute_atr(h, l, c)

    df["ADX"], df["DI_Pos"], df["DI_Neg"] = compute_adx(h, l, c)

    df["CCI"] = compute_cci(h, l, c)

    stoch_k, stoch_d = compute_stochastic(h, l, c)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d

    df["OBV"] = compute_obv(c, v)
    df["VWAP"] = compute_vwap(h, l, c, v)

    return df


#  STREAMLIT CONFIG


st.set_page_config(
    page_title="NeuroTrade",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  /* ── Root palette ── */
  :root {
    --bg:        #080b12;
    --surface:   #0d1117;
    --surface2:  #131a24;
    --border:    rgba(0,255,200,0.12);
    --accent:    #00ffc8;
    --accent2:   #7b61ff;
    --red:       #ff4566;
    --gold:      #ffd166;
    --text:      #e0e8f0;
    --muted:     #5a6a7a;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
  }

  /* ── Global reset ── */
  html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-head);
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] * { color: var(--text) !important; }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    backdrop-filter: blur(8px);
  }
  div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono);
    font-size: 1.3rem;
    color: var(--accent) !important;
  }

  /* ── Select / Input ── */
  div[data-baseweb="select"] > div,
  input[type="text"], input[type="number"] {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: var(--font-mono);
    font-weight: 700;
    letter-spacing: 0.08em;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; }

  /* ── Insight cards ── */
  .insight-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    backdrop-filter: blur(12px);
  }
  .insight-card.bull  { border-left: 3px solid var(--accent); }
  .insight-card.bear  { border-left: 3px solid var(--red); }
  .insight-card.neut  { border-left: 3px solid var(--gold); }
  .insight-icon { font-size: 1.3rem; line-height: 1; margin-top: 2px; }
  .insight-title {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 2px;
  }
  .insight-body {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
  }

  /* ── Section headers ── */
  .section-header {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 16px 0 8px;
  }

  /* ── Logo ── */
  .logo-text {
    font-family: var(--font-head);
    font-weight: 800;
    font-size: 1.5rem;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.04em;
  }
  .logo-sub {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
  }

  /* ── Signal badge ── */
  .signal-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-buy  { background: rgba(0,255,200,0.15); color: #00ffc8; border: 1px solid rgba(0,255,200,0.4); }
  .badge-sell { background: rgba(255,69,102,0.15); color: #ff4566; border: 1px solid rgba(255,69,102,0.4); }
  .badge-hold { background: rgba(255,209,102,0.15); color: #ffd166; border: 1px solid rgba(255,209,102,0.4); }

  /* ── Plotly container ── */
  .stPlotlyChart { border-radius: 16px; overflow: hidden; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.2rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a0e16",
    font=dict(family="Space Mono, monospace", color="#8899aa", size=11),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,255,200,0.15)",
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        rangeslider=dict(visible=False),
        showspikes=True, spikecolor="#00ffc8", spikethickness=1,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        showspikes=True, spikecolor="#00ffc8", spikethickness=1,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0d1117",
        bordercolor="rgba(0,255,200,0.3)",
        font=dict(family="Space Mono", color="#e0e8f0", size=11),
    ),
)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "NFLX", "AMD", "INTC", "BTC-USD", "ETH-USD", "SOL-USD",
    "SPY", "QQQ", "GLD", "RELIANCE.NS", "TCS.NS", "INFY.NS",
]

TIMEFRAMES = {
    "1 Week":    ("7d",  "15m"),
    "1 Month":   ("1mo", "1h"),
    "3 Months":  ("3mo", "1d"),
    "6 Months":  ("6mo", "1d"),
    "1 Year":    ("1y",  "1d"),
    "2 Years":   ("2y",  "1wk"),
    "5 Years":   ("5y",  "1wk"),
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def detect_signals(df: pd.DataFrame, sma_p, ema_p):
    """Returns (signal_str, color) based on last-row indicators."""
    last = df.iloc[-1]
    score = 0

    rsi = last.get("RSI", 50)
    if rsi > 70:   score -= 2
    elif rsi < 30: score += 2
    elif rsi > 55: score += 1
    elif rsi < 45: score -= 1

    macd = last.get("MACD", 0)
    sig  = last.get("MACD_Signal", 0)
    if macd > sig:  score += 2
    else:           score -= 2

    adx = last.get("ADX", 0)
    if adx > 25:
        di_pos = last.get("DI_Pos", 0)
        di_neg = last.get("DI_Neg", 0)
        score += 1 if di_pos > di_neg else -1

    close = last["Close"]
    bb_upper = last.get("BB_Upper", close * 1.1)
    bb_lower = last.get("BB_Lower", close * 0.9)
    if close > bb_upper: score -= 1
    elif close < bb_lower: score += 1

    sma = last.get(f"SMA_{sma_p}", close)
    ema = last.get(f"EMA_{ema_p}", close)
    if close > sma: score += 1
    else:           score -= 1
    if close > ema: score += 1
    else:           score -= 1

    if score >= 3:   return "BUY",  "#00ffc8"
    elif score <= -3: return "SELL", "#ff4566"
    else:            return "HOLD", "#ffd166"


def generate_insights(df: pd.DataFrame) -> list[dict]:
    last  = df.iloc[-1]
    prev  = df.iloc[-2] if len(df) > 1 else last
    insights = []

    # RSI
    rsi = last.get("RSI", np.nan)
    if not np.isnan(rsi):
        if rsi > 70:
            insights.append(dict(kind="bear", icon="🔴", title="RSI · Momentum",
                body=f"RSI at {rsi:.1f} — market is <b>overbought</b>. Watch for reversal."))
        elif rsi < 30:
            insights.append(dict(kind="bull", icon="🟢", title="RSI · Momentum",
                body=f"RSI at {rsi:.1f} — market is <b>oversold</b>. Potential bounce."))
        else:
            insights.append(dict(kind="neut", icon="🟡", title="RSI · Momentum",
                body=f"RSI at {rsi:.1f} — neutral zone. No extreme pressure."))

    # MACD crossover
    macd      = last.get("MACD", np.nan)
    macd_sig  = last.get("MACD_Signal", np.nan)
    p_macd    = prev.get("MACD", np.nan)
    p_sig     = prev.get("MACD_Signal", np.nan)
    if not any(np.isnan(x) for x in [macd, macd_sig, p_macd, p_sig]):
        if p_macd <= p_sig and macd > macd_sig:
            insights.append(dict(kind="bull", icon="⚡", title="MACD · Crossover",
                body="Bullish crossover detected — MACD crossed <b>above</b> Signal Line."))
        elif p_macd >= p_sig and macd < macd_sig:
            insights.append(dict(kind="bear", icon="⚠️", title="MACD · Crossover",
                body="Bearish crossover detected — MACD crossed <b>below</b> Signal Line."))
        elif macd > 0:
            insights.append(dict(kind="bull", icon="📈", title="MACD · Trend",
                body="MACD positive — <b>bullish momentum</b> persisting."))
        else:
            insights.append(dict(kind="bear", icon="📉", title="MACD · Trend",
                body="MACD negative — <b>bearish momentum</b> persisting."))

    # ADX
    adx    = last.get("ADX", np.nan)
    di_pos = last.get("DI_Pos", np.nan)
    di_neg = last.get("DI_Neg", np.nan)
    if not np.isnan(adx):
        if adx > 25:
            direction = "bullish" if di_pos > di_neg else "bearish"
            insights.append(dict(kind="bull" if direction=="bullish" else "bear",
                icon="💪", title="ADX · Trend Strength",
                body=f"ADX at {adx:.1f} — <b>strong {direction} trend</b>. Ride with the momentum."))
        else:
            insights.append(dict(kind="neut", icon="〰️", title="ADX · Trend Strength",
                body=f"ADX at {adx:.1f} — <b>weak trend / ranging</b>. Avoid trend-following."))

    # Bollinger
    close    = last.get("Close", np.nan)
    bb_upper = last.get("BB_Upper", np.nan)
    bb_lower = last.get("BB_Lower", np.nan)
    bb_mid   = last.get("BB_Mid", np.nan)
    if not any(np.isnan(x) for x in [close, bb_upper, bb_lower]):
        band_width = bb_upper - bb_lower
        pct_b = (close - bb_lower) / (band_width + 1e-9)
        if close > bb_upper:
            insights.append(dict(kind="bear", icon="🔺", title="Bollinger Bands",
                body="Price above upper band — potential <b>mean reversion sell</b> setup."))
        elif close < bb_lower:
            insights.append(dict(kind="bull", icon="🔻", title="Bollinger Bands",
                body="Price below lower band — potential <b>mean reversion buy</b> setup."))
        else:
            insights.append(dict(kind="neut", icon="📊", title="Bollinger Bands",
                body=f"%B at {pct_b:.0%} — price within bands. Squeeze watch: {'⚡ narrow' if band_width < (bb_mid * 0.03) else 'normal width'}."))

    # CCI
    cci = last.get("CCI", np.nan)
    if not np.isnan(cci):
        if cci > 100:
            insights.append(dict(kind="bear", icon="🌡️", title="CCI · Overbought",
                body=f"CCI at {cci:.1f} — commodity channel signals <b>overbought</b> conditions."))
        elif cci < -100:
            insights.append(dict(kind="bull", icon="❄️", title="CCI · Oversold",
                body=f"CCI at {cci:.1f} — commodity channel signals <b>oversold</b> conditions."))

    return insights


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_chart(df, cfg):
    show_rsi  = cfg["show_rsi"]
    show_macd = cfg["show_macd"]
    show_obv  = cfg["show_obv"]
    sma_p     = cfg["sma_p"]
    ema_p     = cfg["ema_p"]

    n_rows  = 1 + show_rsi + show_macd + show_obv
    row_h   = [0.55]
    specs   = [[ {"secondary_y": False} ]]
    sub_titles = [""]

    if show_rsi:
        row_h.append(0.18)
        specs.append([{"secondary_y": False}])
        sub_titles.append("RSI")
    if show_macd:
        row_h.append(0.18)
        specs.append([{"secondary_y": False}])
        sub_titles.append("MACD")
    if show_obv:
        row_h.append(0.12)
        specs.append([{"secondary_y": False}])
        sub_titles.append("OBV")

    # Normalise heights
    total = sum(row_h)
    row_h = [r / total for r in row_h]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=row_h,
        specs=specs,
        subplot_titles=sub_titles,
    )

    idx = df.index

    # ── Candlestick ─────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=idx,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#00ffc8", width=1), fillcolor="rgba(0,255,200,0.75)"),
        decreasing=dict(line=dict(color="#ff4566", width=1), fillcolor="rgba(255,69,102,0.75)"),
        whiskerwidth=0.3,
    ), row=1, col=1)

    # ── Volume bars ──────────────────────────────────────────────────────────
    colors = ["rgba(0,255,200,0.25)" if c >= o else "rgba(255,69,102,0.25)"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=idx, y=df["Volume"],
        name="Volume",
        marker_color=colors,
        yaxis="y2",
        showlegend=False,
    ), row=1, col=1)
    fig.update_layout(yaxis2=dict(
        overlaying="y", side="right",
        showgrid=False, showticklabels=False,
        range=[0, df["Volume"].max() * 5],
    ))

    # ── SMA ──────────────────────────────────────────────────────────────────
    sma_col = f"SMA_{sma_p}"
    if cfg["show_sma"] and sma_col in df:
        fig.add_trace(go.Scatter(
            x=idx, y=df[sma_col], name=f"SMA {sma_p}",
            line=dict(color="#ffd166", width=1.5, dash="dot"),
        ), row=1, col=1)

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema_col = f"EMA_{ema_p}"
    if cfg["show_ema"] and ema_col in df:
        fig.add_trace(go.Scatter(
            x=idx, y=df[ema_col], name=f"EMA {ema_p}",
            line=dict(color="#7b61ff", width=1.5),
        ), row=1, col=1)

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    if cfg["show_bb"] and "BB_Upper" in df:
        fig.add_trace(go.Scatter(
            x=idx, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(123,97,255,0.5)", width=1),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=df["BB_Lower"], name="BB Bands",
            fill="tonexty",
            fillcolor="rgba(123,97,255,0.06)",
            line=dict(color="rgba(123,97,255,0.5)", width=1),
        ), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────────────────────
    if cfg["show_vwap"] and "VWAP" in df:
        fig.add_trace(go.Scatter(
            x=idx, y=df["VWAP"], name="VWAP",
            line=dict(color="#ff6b9d", width=1.2, dash="dashdot"),
        ), row=1, col=1)

    # ── Buy/Sell signal markers ──────────────────────────────────────────────
    if cfg["show_signals"] and "RSI" in df and "MACD" in df:
        buy_mask  = (df["RSI"] < 35) & (df["MACD"] > df["MACD_Signal"])
        sell_mask = (df["RSI"] > 65) & (df["MACD"] < df["MACD_Signal"])

        if buy_mask.any():
            fig.add_trace(go.Scatter(
                x=idx[buy_mask], y=df["Low"][buy_mask] * 0.997,
                mode="markers", name="BUY Signal",
                marker=dict(symbol="triangle-up", size=9,
                            color="#00ffc8", line=dict(color="#000", width=1)),
            ), row=1, col=1)
        if sell_mask.any():
            fig.add_trace(go.Scatter(
                x=idx[sell_mask], y=df["High"][sell_mask] * 1.003,
                mode="markers", name="SELL Signal",
                marker=dict(symbol="triangle-down", size=9,
                            color="#ff4566", line=dict(color="#000", width=1)),
            ), row=1, col=1)

    # ── Sub-panels ───────────────────────────────────────────────────────────
    sub_row = 2

    if show_rsi and "RSI" in df:
        fig.add_trace(go.Scatter(
            x=idx, y=df["RSI"], name="RSI",
            line=dict(color="#00ffc8", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,255,200,0.04)",
        ), row=sub_row, col=1)
        for level, col in [(70, "rgba(255,69,102,0.3)"), (30, "rgba(0,255,200,0.3)")]:
            fig.add_hline(y=level, line=dict(color=col, width=1, dash="dot"),
                          row=sub_row, col=1)
        fig.update_yaxes(range=[0, 100], row=sub_row, col=1)
        sub_row += 1

    if show_macd and "MACD" in df:
        hist_colors = ["rgba(0,255,200,0.7)" if v >= 0 else "rgba(255,69,102,0.7)"
                       for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=idx, y=df["MACD_Hist"], name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=sub_row, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=df["MACD"], name="MACD",
            line=dict(color="#00ffc8", width=1.5),
        ), row=sub_row, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#ff6b9d", width=1.2, dash="dot"),
        ), row=sub_row, col=1)
        sub_row += 1

    if show_obv and "OBV" in df:
        fig.add_trace(go.Scatter(
            x=idx, y=df["OBV"], name="OBV",
            line=dict(color="#7b61ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(123,97,255,0.06)",
        ), row=sub_row, col=1)

    # ── Apply layout ─────────────────────────────────────────────────────────
    fig.update_layout(**CHART_LAYOUT,
        height=680 + n_rows * 30,
        dragmode="zoom",
        newshape=dict(line_color="#00ffc8"),
    )
    for i in range(1, n_rows + 1):
        fig.update_xaxes(
            row=i, col=1,
            gridcolor="rgba(255,255,255,0.04)",
            showgrid=True,
            rangeslider_visible=False,
            showspikes=True, spikecolor="#00ffc8", spikethickness=1,
        )
        fig.update_yaxes(
            row=i, col=1,
            gridcolor="rgba(255,255,255,0.04)",
            showgrid=True,
            showspikes=True, spikecolor="#00ffc8", spikethickness=1,
        )

    # Subplot title color
    for ann in fig.layout.annotations:
        ann.font.color = "#5a6a7a"
        ann.font.family = "Space Mono, monospace"
        ann.font.size = 10

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="logo-text">⚡ NeuroTrade</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">AI-Powered Trading Terminal</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Data Source ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio("", ["Live Market (yfinance)", "Upload CSV"],
                           label_visibility="collapsed")

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("OHLCV CSV", type=["csv"])
    else:
        uploaded_file = None
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_input = st.text_input("Ticker", value="AAPL",
                                         placeholder="e.g. AAPL, BTC-USD")
        with col2:
            # Quick-pick
            quick = st.selectbox("Quick", [""] + TICKERS, label_visibility="visible")
        ticker = (quick if quick else ticker_input).upper().strip()

        timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=3)

    st.markdown("---")

    # ── Indicator Toggles ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Indicators</div>', unsafe_allow_html=True)

    show_sma = st.toggle("SMA", value=True)
    sma_p    = st.slider("SMA Period", 5, 200, 20, step=5,
                         disabled=not show_sma, label_visibility="collapsed") if show_sma else 20

    show_ema = st.toggle("EMA", value=True)
    ema_p    = st.slider("EMA Period", 5, 200, 20, step=5,
                         disabled=not show_ema, label_visibility="collapsed") if show_ema else 20

    show_bb  = st.toggle("Bollinger Bands", value=True)
    bb_p     = st.slider("BB Period", 5, 50, 20, step=5,
                         disabled=not show_bb, label_visibility="collapsed") if show_bb else 20

    show_vwap    = st.toggle("VWAP", value=True)
    show_rsi     = st.toggle("RSI", value=True)
    show_macd    = st.toggle("MACD", value=True)
    show_obv     = st.toggle("OBV", value=False)
    show_adx     = st.toggle("ADX Panel", value=False)
    show_signals = st.toggle("Buy/Sell Signals", value=True)

    st.markdown("---")
    st.markdown('<div class="section-header">RSI Period</div>', unsafe_allow_html=True)
    rsi_p = st.slider("", 5, 30, 14, label_visibility="collapsed")

    st.markdown("---")
    fetch_btn = st.button("⚡  Load Data", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "df_raw"  not in st.session_state: st.session_state.df_raw  = None
if "df_ind"  not in st.session_state: st.session_state.df_ind  = None
if "ticker"  not in st.session_state: st.session_state.ticker  = "AAPL"
if "loaded"  not in st.session_state: st.session_state.loaded  = False


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOAD
# ══════════════════════════════════════════════════════════════════════════════

if fetch_btn or (not st.session_state.loaded):
    with st.spinner("Fetching market data…"):
        try:
            if data_source == "Upload CSV" and uploaded_file is not None:
                df_raw = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                df_raw.columns = [c.strip().title() for c in df_raw.columns]
                st.session_state.ticker = "CUSTOM"
            elif data_source == "Live Market (yfinance)":
                period, interval = TIMEFRAMES[timeframe]
                df_raw = fetch_yfinance(ticker, period, interval)
                st.session_state.ticker = ticker
            else:
                df_raw = None

            if df_raw is not None and not df_raw.empty:
                params = dict(sma=sma_p, ema=ema_p, bb=bb_p, rsi=rsi_p)
                if CUSTOM_UTILS:
                    df_ind = add_all_indicators(df_raw.copy())
                else:
                    df_ind = add_all_indicators_builtin(df_raw.copy(), params)
                st.session_state.df_raw = df_raw
                st.session_state.df_ind = df_ind
                st.session_state.loaded = True
            else:
                st.error("No data found. Try a different ticker or timeframe.")
        except Exception as e:
            st.error(f"Data error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

df  = st.session_state.df_ind
raw = st.session_state.df_raw

if df is None or df.empty:
    # ── Hero empty state ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
    height:70vh;gap:16px;opacity:0.5;">
      <div style="font-size:4rem;">⚡</div>
      <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;
      background:linear-gradient(90deg,#00ffc8,#7b61ff);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;">NeuroTrade</div>
      <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#5a6a7a;
      letter-spacing:0.15em;text-transform:uppercase;">Select a ticker and click Load Data</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Header row ───────────────────────────────────────────────────────────────
last_close  = float(df["Close"].iloc[-1])
prev_close  = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change      = last_close - prev_close
change_pct  = (change / prev_close) * 100

signal, sig_color = detect_signals(df, sma_p, ema_p)
badge_class = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[signal]

h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    direction = "▲" if change >= 0 else "▼"
    color     = "#00ffc8" if change >= 0 else "#ff4566"
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:16px;">'
        f'<span style="font-size:1.8rem;font-weight:800;font-family:Syne;">'
        f'{st.session_state.ticker}</span>'
        f'<span style="font-family:Space Mono;font-size:1.4rem;color:{color};">'
        f'{last_close:,.2f}</span>'
        f'<span style="font-family:Space Mono;font-size:0.9rem;color:{color};">'
        f'{direction} {abs(change):.2f} ({abs(change_pct):.2f}%)</span>'
        f'&nbsp;&nbsp;<span class="signal-badge {badge_class}">{signal}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with h_col2:
    st.markdown(
        f'<div style="text-align:right;font-family:Space Mono;font-size:0.7rem;color:#5a6a7a;">'
        f'{len(df)} candles &nbsp;·&nbsp; {str(df.index[-1])[:10]}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin:4px 0 10px;'></div>", unsafe_allow_html=True)

# ── Metrics strip ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)

rsi_val  = df["RSI"].iloc[-1]  if "RSI"  in df else np.nan
macd_val = df["MACD"].iloc[-1] if "MACD" in df else np.nan
adx_val  = df["ADX"].iloc[-1]  if "ADX"  in df else np.nan
atr_val  = df["ATR"].iloc[-1]  if "ATR"  in df else np.nan
vwap_val = df["VWAP"].iloc[-1] if "VWAP" in df else np.nan
cci_val  = df["CCI"].iloc[-1]  if "CCI"  in df else np.nan

with m1: st.metric("RSI",  f"{rsi_val:.1f}"  if not np.isnan(rsi_val)  else "—", f"{rsi_val - df['RSI'].iloc[-2]:.1f}"  if len(df)>1 and "RSI" in df else None)
with m2: st.metric("MACD", f"{macd_val:.3f}" if not np.isnan(macd_val) else "—")
with m3: st.metric("ADX",  f"{adx_val:.1f}"  if not np.isnan(adx_val)  else "—")
with m4: st.metric("ATR",  f"{atr_val:.2f}"  if not np.isnan(atr_val)  else "—")
with m5: st.metric("VWAP", f"{vwap_val:.2f}" if not np.isnan(vwap_val) else "—")
with m6: st.metric("CCI",  f"{cci_val:.1f}"  if not np.isnan(cci_val)  else "—")

st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)

# ── Main chart ────────────────────────────────────────────────────────────────
cfg = dict(
    show_sma=show_sma, sma_p=sma_p,
    show_ema=show_ema, ema_p=ema_p,
    show_bb=show_bb, bb_p=bb_p,
    show_vwap=show_vwap,
    show_rsi=show_rsi,
    show_macd=show_macd,
    show_obv=show_obv,
    show_signals=show_signals,
)
fig = build_chart(df, cfg)
st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "scrollZoom": True,
})

# ── ADX panel (optional) ─────────────────────────────────────────────────────
if show_adx and "ADX" in df:
    adx_fig = go.Figure()
    adx_fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX",
                                  line=dict(color="#ffd166", width=2)))
    adx_fig.add_trace(go.Scatter(x=df.index, y=df["DI_Pos"], name="+DI",
                                  line=dict(color="#00ffc8", width=1.2, dash="dot")))
    adx_fig.add_trace(go.Scatter(x=df.index, y=df["DI_Neg"], name="−DI",
                                  line=dict(color="#ff4566", width=1.2, dash="dot")))
    adx_fig.add_hline(y=25, line=dict(color="rgba(255,209,102,0.4)", dash="dot", width=1))
    adx_fig.update_layout(**CHART_LAYOUT, height=200,
                           title=dict(text="ADX — Trend Strength", font=dict(
                               family="Space Mono", color="#5a6a7a", size=11)))
    st.plotly_chart(adx_fig, use_container_width=True, config={"displaylogo": False})

st.markdown("---")

# ── AI Insights ───────────────────────────────────────────────────────────────
st.markdown('<div style="font-family:Syne;font-weight:800;font-size:1.1rem;'
            'background:linear-gradient(90deg,#00ffc8,#7b61ff);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
            'margin-bottom:12px;">🧠 AI Insights</div>', unsafe_allow_html=True)

insights = generate_insights(df)

if insights:
    cols = st.columns(min(len(insights), 3))
    for i, ins in enumerate(insights):
        with cols[i % 3]:
            st.markdown(
                f'<div class="insight-card {ins["kind"]}">'
                f'  <div class="insight-icon">{ins["icon"]}</div>'
                f'  <div>'
                f'    <div class="insight-title">{ins["title"]}</div>'
                f'    <div class="insight-body">{ins["body"]}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
else:
    st.info("Not enough data to generate insights.")

st.markdown("---")

# ── Raw data expander ─────────────────────────────────────────────────────────
with st.expander("📋  Raw Indicator Data", expanded=False):
    display_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in [f"SMA_{sma_p}", f"EMA_{ema_p}", "RSI", "MACD", "MACD_Signal",
                "BB_Upper", "BB_Mid", "BB_Lower", "ATR", "ADX", "OBV", "VWAP", "CCI"]:
        if col in df.columns:
            display_cols.append(col)
    st.dataframe(
        df[display_cols].tail(50).style
          .background_gradient(subset=["Close"], cmap="RdYlGn")
          .format(precision=3),
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;font-family:Space Mono;font-size:0.65rem;'
    'color:#2a3a4a;padding:20px 0 10px;">NeuroTrade · For educational purposes only · '
    'Not financial advice</div>',
    unsafe_allow_html=True,
)
