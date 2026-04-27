"""
NeuroTrade — AI-Powered Trading Assistant
app.py — Main Streamlit Application with full backend integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import traceback
import io

warnings.filterwarnings("ignore")

from utils import add_all_indicators

# Optional backend modules — graceful fallback for each
try:
    from ml_models import MLConfig, MLTrainer
    _ML = True
except Exception:
    _ML = False

try:
    from dl_models import DLConfig, DLTrainer, ModelType
    _DL = True
except Exception:
    _DL = False

try:
    from quantum_models import QuantumConfig, QuantumTrainer
    _QM = True
except Exception:
    _QM = False

try:
    from Backtester import BacktestConfig, Backtester as BacktestEngine, RiskAnalytics, PositionSide
    _BT = True
except Exception:
    _BT = False

try:
    from regime_detector import RegimeDetector, prepare_regime_features, Regime
    _RD = True
except Exception:
    _RD = False

try:
    from portfolio_sim import PortfolioConfig, PortfolioSimulator, CorrelationAnalyzer, AllocationEngine
    _PS = True
except Exception:
    _PS = False

try:
    from stress_tester import StressTester, CRISIS_PROFILES
    _ST = True
except Exception:
    _ST = False

try:
    from llm_agent import NewsCollector, LLMConfig
    _LLM = True
except Exception:
    _LLM = False

try:
    from reports import TearsheetGenerator
    _RPT = True
except Exception:
    _RPT = False

try:
    from data import YFinanceLoader, OHLCVCleaner
    _DATA = True
except Exception:
    _DATA = False


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR ENGINE (self-contained fallback)
# ══════════════════════════════════════════════════════════════════════════════

def compute_sma(series, period):
    return series.rolling(window=period).mean()

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_bollinger(series, period=20, std_dev=2):
    mid = compute_sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def add_all_indicators_builtin(df, params):
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df[f"SMA_{params['sma']}"] = compute_sma(c, params["sma"])
    df[f"EMA_{params['ema']}"] = compute_ema(c, params["ema"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(c)
    df["RSI"] = compute_rsi(c, params["rsi"])
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(c, params["bb"])
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Mid"] + 1e-9)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    up = h.diff()
    dn = -l.diff()
    pos_dm = np.where((up > dn) & (up > 0), up, 0)
    neg_dm = np.where((dn > up) & (dn > 0), dn, 0)
    atr14 = tr.rolling(14).mean()
    df["DI_Pos"] = 100 * pd.Series(pos_dm, index=c.index).rolling(14).mean() / atr14
    df["DI_Neg"] = 100 * pd.Series(neg_dm, index=c.index).rolling(14).mean() / atr14
    dx = (100 * (df["DI_Pos"] - df["DI_Neg"]).abs() / (df["DI_Pos"] + df["DI_Neg"] + 1e-9))
    df["ADX"] = dx.rolling(14).mean()
    tp = (h + l + c) / 3
    df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True) + 1e-9)
    low_min = l.rolling(14).min()
    high_max = h.rolling(14).max()
    df["Stoch_K"] = 100 * (c - low_min) / (high_max - low_min + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    direction = np.sign(c.diff()).fillna(0)
    df["OBV"] = (direction * v).cumsum()
    df["VWAP"] = (tp * v).cumsum() / v.cumsum()
    df["Returns"] = c.pct_change()
    df["Log_Returns"] = np.log(c / c.shift(1))
    df["HL_Spread"] = (h - l) / (c + 1e-9)
    df["Realized_Vol_21"] = df["Returns"].rolling(21).std() * np.sqrt(252)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["Z_Score_20"] = (c - sma20) / (std20 + 1e-9)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NeuroTrade",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
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
  html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-head);
  }
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] * { color: var(--text) !important; }
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
  div[data-baseweb="select"] > div,
  input[type="text"], input[type="number"] {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
  }
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
  hr { border-color: var(--border) !important; }
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
  .section-header {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 16px 0 8px;
  }
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
  .stPlotlyChart { border-radius: 16px; overflow: hidden; }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.2rem !important; }
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--surface);
    border-radius: 12px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    border-radius: 8px;
    padding: 6px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,255,200,0.15), rgba(123,97,255,0.15));
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent);
  }
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
def fetch_yfinance(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def detect_signals(df, sma_p, ema_p):
    last = df.iloc[-1]
    score = 0
    rsi = last.get("RSI", 50)
    if rsi > 70:   score -= 2
    elif rsi < 30: score += 2
    elif rsi > 55: score += 1
    elif rsi < 45: score -= 1
    macd = last.get("MACD", 0)
    sig = last.get("MACD_Signal", 0)
    if macd > sig: score += 2
    else:          score -= 2
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
    if score >= 3:    return "BUY",  "#00ffc8"
    elif score <= -3: return "SELL", "#ff4566"
    else:             return "HOLD", "#ffd166"


def generate_insights(df):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    insights = []
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
    macd = last.get("MACD", np.nan)
    macd_sig = last.get("MACD_Signal", np.nan)
    p_macd = prev.get("MACD", np.nan)
    p_sig = prev.get("MACD_Signal", np.nan)
    if not any(np.isnan(x) for x in [macd, macd_sig, p_macd, p_sig]):
        if p_macd <= p_sig and macd > macd_sig:
            insights.append(dict(kind="bull", icon="⚡", title="MACD · Crossover",
                body="Bullish crossover — MACD crossed <b>above</b> Signal Line."))
        elif p_macd >= p_sig and macd < macd_sig:
            insights.append(dict(kind="bear", icon="⚠️", title="MACD · Crossover",
                body="Bearish crossover — MACD crossed <b>below</b> Signal Line."))
        elif macd > 0:
            insights.append(dict(kind="bull", icon="📈", title="MACD · Trend",
                body="MACD positive — <b>bullish momentum</b> persisting."))
        else:
            insights.append(dict(kind="bear", icon="📉", title="MACD · Trend",
                body="MACD negative — <b>bearish momentum</b> persisting."))
    adx = last.get("ADX", np.nan)
    di_pos = last.get("DI_Pos", np.nan)
    di_neg = last.get("DI_Neg", np.nan)
    if not np.isnan(adx):
        if adx > 25:
            direction = "bullish" if di_pos > di_neg else "bearish"
            insights.append(dict(kind="bull" if direction == "bullish" else "bear",
                icon="💪", title="ADX · Trend Strength",
                body=f"ADX at {adx:.1f} — <b>strong {direction} trend</b>."))
        else:
            insights.append(dict(kind="neut", icon="〰️", title="ADX · Trend Strength",
                body=f"ADX at {adx:.1f} — <b>weak trend / ranging</b>."))
    close = last.get("Close", np.nan)
    bb_upper = last.get("BB_Upper", np.nan)
    bb_lower = last.get("BB_Lower", np.nan)
    bb_mid = last.get("BB_Mid", np.nan)
    if not any(np.isnan(x) for x in [close, bb_upper, bb_lower]):
        band_width = bb_upper - bb_lower
        pct_b = (close - bb_lower) / (band_width + 1e-9)
        if close > bb_upper:
            insights.append(dict(kind="bear", icon="🔺", title="Bollinger Bands",
                body="Price above upper band — potential <b>mean reversion sell</b>."))
        elif close < bb_lower:
            insights.append(dict(kind="bull", icon="🔻", title="Bollinger Bands",
                body="Price below lower band — potential <b>mean reversion buy</b>."))
        else:
            narrow = band_width < (bb_mid * 0.03) if not np.isnan(bb_mid) else False
            insights.append(dict(kind="neut", icon="📊", title="Bollinger Bands",
                body=f"%B at {pct_b:.0%} — {'squeeze detected' if narrow else 'normal width'}."))
    cci = last.get("CCI", np.nan)
    if not np.isnan(cci):
        if cci > 100:
            insights.append(dict(kind="bear", icon="🌡️", title="CCI · Overbought",
                body=f"CCI at {cci:.1f} — <b>overbought</b> conditions."))
        elif cci < -100:
            insights.append(dict(kind="bull", icon="❄️", title="CCI · Oversold",
                body=f"CCI at {cci:.1f} — <b>oversold</b> conditions."))
    return insights


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_chart(df, cfg):
    show_rsi = cfg["show_rsi"]
    show_macd = cfg["show_macd"]
    show_obv = cfg["show_obv"]
    sma_p = cfg["sma_p"]
    ema_p = cfg["ema_p"]
    n_rows = 1 + show_rsi + show_macd + show_obv
    row_h = [0.55]
    specs = [[{"secondary_y": False}]]
    sub_titles = [""]
    if show_rsi:
        row_h.append(0.18); specs.append([{"secondary_y": False}]); sub_titles.append("RSI")
    if show_macd:
        row_h.append(0.18); specs.append([{"secondary_y": False}]); sub_titles.append("MACD")
    if show_obv:
        row_h.append(0.12); specs.append([{"secondary_y": False}]); sub_titles.append("OBV")
    total = sum(row_h)
    row_h = [r / total for r in row_h]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.015,
                        row_heights=row_h, specs=specs, subplot_titles=sub_titles)
    idx = df.index
    fig.add_trace(go.Candlestick(
        x=idx, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#00ffc8", width=1), fillcolor="rgba(0,255,200,0.75)"),
        decreasing=dict(line=dict(color="#ff4566", width=1), fillcolor="rgba(255,69,102,0.75)"),
        whiskerwidth=0.3,
    ), row=1, col=1)
    colors = ["rgba(0,255,200,0.25)" if c >= o else "rgba(255,69,102,0.25)"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=idx, y=df["Volume"], name="Volume", marker_color=colors,
                         yaxis="y2", showlegend=False), row=1, col=1)
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                  showticklabels=False, range=[0, df["Volume"].max() * 5]))
    sma_col = f"SMA_{sma_p}"
    if cfg["show_sma"] and sma_col in df:
        fig.add_trace(go.Scatter(x=idx, y=df[sma_col], name=f"SMA {sma_p}",
                                  line=dict(color="#ffd166", width=1.5, dash="dot")), row=1, col=1)
    ema_col = f"EMA_{ema_p}"
    if cfg["show_ema"] and ema_col in df:
        fig.add_trace(go.Scatter(x=idx, y=df[ema_col], name=f"EMA {ema_p}",
                                  line=dict(color="#7b61ff", width=1.5)), row=1, col=1)
    if cfg["show_bb"] and "BB_Upper" in df:
        fig.add_trace(go.Scatter(x=idx, y=df["BB_Upper"], name="BB Upper",
                                  line=dict(color="rgba(123,97,255,0.5)", width=1),
                                  showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=idx, y=df["BB_Lower"], name="BB Bands",
                                  fill="tonexty", fillcolor="rgba(123,97,255,0.06)",
                                  line=dict(color="rgba(123,97,255,0.5)", width=1)), row=1, col=1)
    if cfg["show_vwap"] and "VWAP" in df:
        fig.add_trace(go.Scatter(x=idx, y=df["VWAP"], name="VWAP",
                                  line=dict(color="#ff6b9d", width=1.2, dash="dashdot")), row=1, col=1)
    if cfg["show_signals"] and "RSI" in df and "MACD" in df:
        buy_mask = (df["RSI"] < 35) & (df["MACD"] > df["MACD_Signal"])
        sell_mask = (df["RSI"] > 65) & (df["MACD"] < df["MACD_Signal"])
        if buy_mask.any():
            fig.add_trace(go.Scatter(x=idx[buy_mask], y=df["Low"][buy_mask] * 0.997,
                                      mode="markers", name="BUY Signal",
                                      marker=dict(symbol="triangle-up", size=9, color="#00ffc8",
                                                  line=dict(color="#000", width=1))), row=1, col=1)
        if sell_mask.any():
            fig.add_trace(go.Scatter(x=idx[sell_mask], y=df["High"][sell_mask] * 1.003,
                                      mode="markers", name="SELL Signal",
                                      marker=dict(symbol="triangle-down", size=9, color="#ff4566",
                                                  line=dict(color="#000", width=1))), row=1, col=1)
    sub_row = 2
    if show_rsi and "RSI" in df:
        fig.add_trace(go.Scatter(x=idx, y=df["RSI"], name="RSI",
                                  line=dict(color="#00ffc8", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(0,255,200,0.04)"), row=sub_row, col=1)
        for level, col in [(70, "rgba(255,69,102,0.3)"), (30, "rgba(0,255,200,0.3)")]:
            fig.add_hline(y=level, line=dict(color=col, width=1, dash="dot"), row=sub_row, col=1)
        fig.update_yaxes(range=[0, 100], row=sub_row, col=1)
        sub_row += 1
    if show_macd and "MACD" in df:
        hist_colors = ["rgba(0,255,200,0.7)" if v >= 0 else "rgba(255,69,102,0.7)"
                       for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=idx, y=df["MACD_Hist"], name="MACD Hist",
                              marker_color=hist_colors, showlegend=False), row=sub_row, col=1)
        fig.add_trace(go.Scatter(x=idx, y=df["MACD"], name="MACD",
                                  line=dict(color="#00ffc8", width=1.5)), row=sub_row, col=1)
        fig.add_trace(go.Scatter(x=idx, y=df["MACD_Signal"], name="Signal",
                                  line=dict(color="#ff6b9d", width=1.2, dash="dot")), row=sub_row, col=1)
        sub_row += 1
    if show_obv and "OBV" in df:
        fig.add_trace(go.Scatter(x=idx, y=df["OBV"], name="OBV",
                                  line=dict(color="#7b61ff", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(123,97,255,0.06)"), row=sub_row, col=1)
    fig.update_layout(**CHART_LAYOUT, height=680 + n_rows * 30, dragmode="zoom",
                       newshape=dict(line_color="#00ffc8"))
    for i in range(1, n_rows + 1):
        fig.update_xaxes(row=i, col=1, gridcolor="rgba(255,255,255,0.04)", showgrid=True,
                         rangeslider_visible=False, showspikes=True, spikecolor="#00ffc8", spikethickness=1)
        fig.update_yaxes(row=i, col=1, gridcolor="rgba(255,255,255,0.04)", showgrid=True,
                         showspikes=True, spikecolor="#00ffc8", spikethickness=1)
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

    st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio("", ["Live Market (yfinance)", "Upload CSV"], label_visibility="collapsed")

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("OHLCV CSV", type=["csv"])
    else:
        uploaded_file = None
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_input = st.text_input("Ticker", value="AAPL", placeholder="e.g. AAPL, BTC-USD")
        with col2:
            quick = st.selectbox("Quick", [""] + TICKERS, label_visibility="visible")
        ticker = (quick if quick else ticker_input).upper().strip()
        timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=3)

    st.markdown("---")
    st.markdown('<div class="section-header">Indicators</div>', unsafe_allow_html=True)
    show_sma = st.toggle("SMA", value=True)
    sma_p = st.slider("SMA Period", 5, 200, 20, step=5, disabled=not show_sma,
                       label_visibility="collapsed") if show_sma else 20
    show_ema = st.toggle("EMA", value=True)
    ema_p = st.slider("EMA Period", 5, 200, 20, step=5, disabled=not show_ema,
                       label_visibility="collapsed") if show_ema else 20
    show_bb = st.toggle("Bollinger Bands", value=True)
    bb_p = st.slider("BB Period", 5, 50, 20, step=5, disabled=not show_bb,
                      label_visibility="collapsed") if show_bb else 20
    show_vwap = st.toggle("VWAP", value=True)
    show_rsi = st.toggle("RSI", value=True)
    show_macd = st.toggle("MACD", value=True)
    show_obv = st.toggle("OBV", value=False)
    show_adx = st.toggle("ADX Panel", value=False)
    show_signals = st.toggle("Buy/Sell Signals", value=True)

    st.markdown("---")
    st.markdown('<div class="section-header">RSI Period</div>', unsafe_allow_html=True)
    rsi_p = st.slider("", 5, 30, 14, label_visibility="collapsed")

    st.markdown("---")
    fetch_btn = st.button("⚡  Load Data", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Modules Available</div>', unsafe_allow_html=True)
    modules = {
        "ML Models": _ML, "DL Models": _DL, "Quantum": _QM,
        "Backtester": _BT, "Regime Detector": _RD,
        "Portfolio Sim": _PS, "Stress Test": _ST,
        "News/LLM": _LLM, "Reports": _RPT,
    }
    for name, available in modules.items():
        status = "🟢" if available else "🔴"
        st.markdown(f"<span style='font-family:Space Mono;font-size:0.7rem;color:{'#00ffc8' if available else '#ff4566'};'>"
                    f"{status} {name}</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "df_raw" not in st.session_state: st.session_state.df_raw = None
if "df_ind" not in st.session_state: st.session_state.df_ind = None
if "ticker" not in st.session_state: st.session_state.ticker = "AAPL"
if "loaded" not in st.session_state: st.session_state.loaded = False


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOAD
# ══════════════════════════════════════════════════════════════════════════════

if fetch_btn or (not st.session_state.loaded):
    with st.spinner("Fetching market data..."):
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
                try:
                    df_ind = add_all_indicators(df_raw.copy())
                except Exception:
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

df = st.session_state.df_ind
raw = st.session_state.df_raw

if df is None or df.empty:
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
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

signal, sig_color = detect_signals(df, sma_p, ema_p)
badge_class = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[signal]

h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    direction = "▲" if change >= 0 else "▼"
    color = "#00ffc8" if change >= 0 else "#ff4566"
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:16px;">'
        f'<span style="font-size:1.8rem;font-weight:800;font-family:Syne;">{st.session_state.ticker}</span>'
        f'<span style="font-family:Space Mono;font-size:1.4rem;color:{color};">{last_close:,.2f}</span>'
        f'<span style="font-family:Space Mono;font-size:0.9rem;color:{color};">'
        f'{direction} {abs(change):.2f} ({abs(change_pct):.2f}%)</span>'
        f'&nbsp;&nbsp;<span class="signal-badge {badge_class}">{signal}</span>'
        f'</div>', unsafe_allow_html=True,
    )
with h_col2:
    st.markdown(
        f'<div style="text-align:right;font-family:Space Mono;font-size:0.7rem;color:#5a6a7a;">'
        f'{len(df)} candles &nbsp;·&nbsp; {str(df.index[-1])[:10]}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin:4px 0 10px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_names = ["📊 Dashboard"]
if _ML:  tab_names.append("🤖 ML Models")
if _DL:  tab_names.append("🧠 Deep Learning")
if _QM:  tab_names.append("⚛️ Quantum")
if _BT:  tab_names.append("📈 Backtester")
if _RD:  tab_names.append("🔄 Regime")
if _ST:  tab_names.append("🔥 Stress Test")
if _LLM: tab_names.append("📰 News")
tab_names.append("📋 Data")

tabs = st.tabs(tab_names)
tab_idx = 0


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tabs[tab_idx]:
    tab_idx += 1

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    rsi_val = df["RSI"].iloc[-1] if "RSI" in df else np.nan
    macd_val = df["MACD"].iloc[-1] if "MACD" in df else np.nan
    adx_val = df["ADX"].iloc[-1] if "ADX" in df else np.nan
    atr_val = df["ATR"].iloc[-1] if "ATR" in df else np.nan
    vwap_val = df["VWAP"].iloc[-1] if "VWAP" in df else np.nan
    cci_val = df["CCI"].iloc[-1] if "CCI" in df else np.nan

    with m1: st.metric("RSI", f"{rsi_val:.1f}" if not np.isnan(rsi_val) else "—",
                        f"{rsi_val - df['RSI'].iloc[-2]:.1f}" if len(df) > 1 and "RSI" in df else None)
    with m2: st.metric("MACD", f"{macd_val:.3f}" if not np.isnan(macd_val) else "—")
    with m3: st.metric("ADX", f"{adx_val:.1f}" if not np.isnan(adx_val) else "—")
    with m4: st.metric("ATR", f"{atr_val:.2f}" if not np.isnan(atr_val) else "—")
    with m5: st.metric("VWAP", f"{vwap_val:.2f}" if not np.isnan(vwap_val) else "—")
    with m6: st.metric("CCI", f"{cci_val:.1f}" if not np.isnan(cci_val) else "—")

    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)

    cfg = dict(show_sma=show_sma, sma_p=sma_p, show_ema=show_ema, ema_p=ema_p,
               show_bb=show_bb, bb_p=bb_p, show_vwap=show_vwap, show_rsi=show_rsi,
               show_macd=show_macd, show_obv=show_obv, show_signals=show_signals)
    fig = build_chart(df, cfg)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displaylogo": False, "scrollZoom": True})

    if show_adx and "ADX" in df:
        adx_fig = go.Figure()
        adx_fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX",
                                      line=dict(color="#ffd166", width=2)))
        if "DI_Pos" in df:
            adx_fig.add_trace(go.Scatter(x=df.index, y=df["DI_Pos"], name="+DI",
                                          line=dict(color="#00ffc8", width=1.2, dash="dot")))
        if "DI_Neg" in df:
            adx_fig.add_trace(go.Scatter(x=df.index, y=df["DI_Neg"], name="−DI",
                                          line=dict(color="#ff4566", width=1.2, dash="dot")))
        adx_fig.add_hline(y=25, line=dict(color="rgba(255,209,102,0.4)", dash="dot", width=1))
        adx_fig.update_layout(**CHART_LAYOUT, height=200,
                               title=dict(text="ADX — Trend Strength",
                                          font=dict(family="Space Mono", color="#5a6a7a", size=11)))
        st.plotly_chart(adx_fig, use_container_width=True, config={"displaylogo": False})

    st.markdown("---")

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
                    f'</div>', unsafe_allow_html=True)
    else:
        st.info("Not enough data to generate insights.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: ML MODELS
# ══════════════════════════════════════════════════════════════════════════════

if _ML:
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("### 🤖 Classical ML Models")
        st.markdown("Train XGBoost, Random Forest, and SVM on your data for direction prediction.")

        ml_col1, ml_col2, ml_col3 = st.columns(3)
        with ml_col1:
            ml_target = st.selectbox("Target", ["direction", "returns"], key="ml_target")
        with ml_col2:
            ml_horizon = st.slider("Forecast Horizon (bars)", 1, 20, 5, key="ml_horizon")
        with ml_col3:
            ml_lookback = st.slider("Lookback Window", 20, 120, 60, key="ml_lookback")

        if st.button("🚀 Train ML Models", key="train_ml"):
            with st.spinner("Training ML models..."):
                try:
                    cfg = MLConfig(
                        target_type=ml_target,
                        forecast_horizon=ml_horizon,
                        lookback=ml_lookback,
                    )
                    trainer = MLTrainer(cfg)
                    results = trainer.train_all(df)
                    st.session_state["ml_trainer"] = trainer
                    st.session_state["ml_results"] = results

                    for name, result in results.items():
                        with st.expander(f"📊 {name.upper()} Results", expanded=True):
                            mcols = st.columns(len(result.metrics))
                            for i, (k, v) in enumerate(result.metrics.items()):
                                with mcols[i]:
                                    st.metric(k.replace("_", " ").title(), f"{v}")

                            if result.feature_importance:
                                top_fi = dict(list(sorted(result.feature_importance.items(),
                                                          key=lambda x: x[1], reverse=True))[:15])
                                fi_fig = go.Figure(go.Bar(
                                    x=list(top_fi.values()),
                                    y=list(top_fi.keys()),
                                    orientation='h',
                                    marker_color="#00ffc8",
                                ))
                                fi_fig.update_layout(**CHART_LAYOUT, height=350,
                                                     title="Feature Importance (Top 15)")
                                st.plotly_chart(fi_fig, use_container_width=True)

                    pred = trainer.predict_latest(df)
                    sig_text = "BUY" if pred > 0.55 else ("SELL" if pred < 0.45 else "HOLD")
                    sig_col = "#00ffc8" if sig_text == "BUY" else ("#ff4566" if sig_text == "SELL" else "#ffd166")
                    st.markdown(f"<div style='text-align:center;padding:20px;'>"
                                f"<span style='font-family:Space Mono;font-size:1.5rem;color:{sig_col};'>"
                                f"Ensemble Prediction: {pred:.3f} → {sig_text}</span></div>",
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"ML Training Error: {e}")
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: DEEP LEARNING
# ══════════════════════════════════════════════════════════════════════════════

if _DL:
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("### 🧠 Deep Learning Models")
        st.markdown("Train LSTM, TCN, or TFT neural networks for price forecasting.")

        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        with dl_col1:
            dl_model = st.selectbox("Model", ["LSTM", "TCN", "TFT"], key="dl_model")
        with dl_col2:
            dl_epochs = st.slider("Epochs", 5, 100, 30, key="dl_epochs")
        with dl_col3:
            dl_seq = st.slider("Sequence Length", 10, 120, 60, key="dl_seq")
        with dl_col4:
            dl_horizon = st.slider("Forecast Horizon", 1, 20, 5, key="dl_horizon")

        if st.button("🚀 Train DL Model", key="train_dl"):
            with st.spinner(f"Training {dl_model}... (this may take a minute)"):
                try:
                    dl_cfg = DLConfig(
                        epochs=dl_epochs,
                        seq_len=dl_seq,
                        forecast_horizon=dl_horizon,
                    )
                    model_type = {"LSTM": ModelType.LSTM, "TCN": ModelType.TCN, "TFT": ModelType.TFT}[dl_model]
                    trainer = DLTrainer(dl_cfg)
                    result = trainer.train(df, model_type)

                    st.session_state["dl_trainer"] = trainer
                    st.session_state["dl_result"] = result

                    with st.expander(f"📊 {dl_model} Training Results", expanded=True):
                        if "metrics" in result:
                            mcols = st.columns(min(len(result["metrics"]), 6))
                            for i, (k, v) in enumerate(result["metrics"].items()):
                                if i < 6:
                                    with mcols[i]:
                                        st.metric(k.upper(), f"{v}")

                        if "history" in result and result["history"]:
                            hist = result["history"]
                            loss_fig = go.Figure()
                            loss_fig.add_trace(go.Scatter(y=hist.get("train_loss", []), name="Train Loss",
                                                          line=dict(color="#00ffc8")))
                            loss_fig.add_trace(go.Scatter(y=hist.get("val_loss", []), name="Val Loss",
                                                          line=dict(color="#ff4566")))
                            loss_fig.update_layout(**CHART_LAYOUT, height=300, title="Training Loss Curve")
                            st.plotly_chart(loss_fig, use_container_width=True)

                    forecast = trainer.forecast(df)
                    if forecast and "forecast" in forecast and len(forecast["forecast"]) > 0:
                        st.markdown("#### Price Forecast")
                        fc_fig = go.Figure()
                        fc_fig.add_trace(go.Scatter(x=df.index[-30:], y=df["Close"].iloc[-30:],
                                                     name="Actual", line=dict(color="#00ffc8")))
                        fc_fig.add_trace(go.Scatter(x=forecast["dates"], y=forecast["forecast"],
                                                     name="Forecast", line=dict(color="#ffd166", dash="dash")))
                        fc_fig.update_layout(**CHART_LAYOUT, height=350, title=f"{dl_model} Forecast")
                        st.plotly_chart(fc_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"DL Training Error: {e}")
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: QUANTUM MODELS
# ══════════════════════════════════════════════════════════════════════════════

if _QM:
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("### ⚛️ Quantum Models")
        st.markdown("Train VQC/QCNN quantum circuits for market direction prediction.")

        qm_col1, qm_col2, qm_col3 = st.columns(3)
        with qm_col1:
            qm_qubits = st.slider("Qubits", 2, 12, 4, key="qm_qubits")
        with qm_col2:
            qm_layers = st.slider("Layers", 1, 8, 2, key="qm_layers")
        with qm_col3:
            qm_epochs = st.slider("Epochs", 5, 100, 20, key="qm_epochs")

        if st.button("🚀 Train Quantum Model", key="train_qm"):
            with st.spinner("Training quantum circuit... (this is slow on CPU)"):
                try:
                    qm_cfg = QuantumConfig(
                        n_qubits=qm_qubits,
                        n_layers=qm_layers,
                        epochs=qm_epochs,
                    )
                    qm_trainer = QuantumTrainer(qm_cfg)
                    qm_results = qm_trainer.train_all(df)

                    for name, result in qm_results.items():
                        with st.expander(f"⚛️ {name} Results", expanded=True):
                            mcols = st.columns(len(result.metrics))
                            for i, (k, v) in enumerate(result.metrics.items()):
                                with mcols[i]:
                                    st.metric(k.replace("_", " ").title(), f"{v}")

                    st.success("Quantum model training complete!")
                except Exception as e:
                    st.error(f"Quantum Training Error: {e}")
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

if _BT:
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("### 📈 Backtester")
        st.markdown("Run a backtest using indicator-based signals on your loaded data.")

        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
        with bt_col1:
            bt_capital = st.number_input("Initial Capital ($)", 1000, 10_000_000, 100_000, step=10000, key="bt_cap")
        with bt_col2:
            bt_commission = st.number_input("Commission (%)", 0.0, 1.0, 0.1, step=0.01, key="bt_comm")
        with bt_col3:
            bt_sl = st.number_input("Stop Loss (%)", 0.0, 20.0, 0.0, step=0.5, key="bt_sl")
        with bt_col4:
            bt_tp = st.number_input("Take Profit (%)", 0.0, 50.0, 0.0, step=1.0, key="bt_tp")

        if st.button("🚀 Run Backtest", key="run_bt"):
            with st.spinner("Running backtest..."):
                try:
                    bt_cfg = BacktestConfig(
                        initial_capital=bt_capital,
                        commission_pct=bt_commission / 100,
                        stop_loss_pct=bt_sl / 100 if bt_sl > 0 else None,
                        take_profit_pct=bt_tp / 100 if bt_tp > 0 else None,
                    )

                    def signal_func(df_bt, i, **kwargs):
                        if i < 30:
                            return None
                        row = df_bt.iloc[i]
                        rsi_v = row.get("RSI", 50)
                        macd_v = row.get("MACD", 0)
                        macd_s = row.get("MACD_Signal", 0)
                        if rsi_v < 35 and macd_v > macd_s:
                            return PositionSide.LONG
                        elif rsi_v > 65 and macd_v < macd_s:
                            return PositionSide.SHORT
                        return None

                    engine = BacktestEngine(bt_cfg)
                    results = engine.run(df, signal_func)

                    st.session_state["bt_results"] = results

                    r = results
                    bt_m1, bt_m2, bt_m3, bt_m4, bt_m5, bt_m6 = st.columns(6)
                    with bt_m1: st.metric("Total Return", f"{r.get('total_return_pct', 0):.2f}%")
                    with bt_m2: st.metric("Sharpe Ratio", f"{r.get('sharpe_ratio', 0):.3f}")
                    with bt_m3: st.metric("Max Drawdown", f"{r.get('max_drawdown_pct', 0):.2f}%")
                    with bt_m4: st.metric("Win Rate", f"{r.get('win_rate_pct', 0):.1f}%")
                    with bt_m5: st.metric("Trades", f"{r.get('total_trades', 0)}")
                    with bt_m6: st.metric("Profit Factor", f"{r.get('profit_factor', 0):.2f}")

                    if "equity_curve" in r and r["equity_curve"] is not None:
                        eq = r["equity_curve"]
                        eq_fig = go.Figure()
                        eq_fig.add_trace(go.Scatter(
                            x=eq.index if hasattr(eq, 'index') else list(range(len(eq))),
                            y=eq.values if hasattr(eq, 'values') else eq,
                            name="Equity",
                            line=dict(color="#00ffc8", width=2),
                            fill="tozeroy", fillcolor="rgba(0,255,200,0.06)",
                        ))
                        eq_fig.update_layout(**CHART_LAYOUT, height=350,
                                              title="Equity Curve")
                        st.plotly_chart(eq_fig, use_container_width=True)

                    if "trades" in r and r["trades"]:
                        trades_df = pd.DataFrame([{
                            "Entry": t.entry_date, "Exit": t.exit_date,
                            "Side": t.side.value, "Entry $": f"{t.entry_price:.2f}",
                            "Exit $": f"{t.exit_price:.2f}" if t.exit_price else "Open",
                            "PnL": f"{t.pnl:.2f}", "PnL %": f"{t.pnl_pct:.2f}%",
                            "Bars": t.duration_bars,
                        } for t in r["trades"][:50]])
                        st.dataframe(trades_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Backtest Error: {e}")
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

if _RD:
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("### 🔄 Market Regime Detection")
        st.markdown("Detect BULL / BEAR / VOLATILE regimes using KMeans clustering.")

        rd_col1, rd_col2 = st.columns(2)
        with rd_col1:
            rd_clusters = st.slider("Number of Clusters", 2, 5, 3, key="rd_clusters")
        with rd_col2:
            rd_backend = st.selectbox("Backend", ["kmeans"], key="rd_backend")

        if st.button("🚀 Detect Regimes", key="detect_regime"):
            with st.spinner("Detecting market regimes..."):
                try:
                    regime_df = prepare_regime_features(df)
                    detector = RegimeDetector(backend=rd_backend, n_clusters=rd_clusters)
                    regimes = detector.fit_predict(regime_df)
                    st.session_state["regimes"] = regimes

                    color_map = {"BULL": "#00ffc8", "BEAR": "#ff4566", "VOLATILE": "#ffd166", "UNKNOWN": "#5a6a7a"}
                    regime_colors = [color_map.get(r, "#5a6a7a") for r in regimes]

                    reg_fig = go.Figure()
                    reg_fig.add_trace(go.Scatter(
                        x=df.index, y=df["Close"], name="Close",
                        line=dict(color="#8899aa", width=1),
                    ))
                    for regime_name, color in color_map.items():
                        mask = regimes == regime_name
                        if mask.any():
                            reg_fig.add_trace(go.Scatter(
                                x=df.index[mask], y=df["Close"][mask],
                                mode="markers", name=regime_name,
                                marker=dict(color=color, size=5, opacity=0.7),
                            ))
                    reg_fig.update_layout(**CHART_LAYOUT, height=400,
                                           title="Price with Regime Overlay")
                    st.plotly_chart(reg_fig, use_container_width=True)

                    rc = regimes.value_counts()
                    regime_pie = go.Figure(go.Pie(
                        labels=rc.index, values=rc.values,
                        marker=dict(colors=[color_map.get(r, "#5a6a7a") for r in rc.index]),
                        textfont=dict(color="#e0e8f0"),
                    ))
                    regime_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                             plot_bgcolor="#0a0e16",
                                             font=dict(color="#8899aa"),
                                             height=300, title="Regime Distribution")
                    st.plotly_chart(regime_pie, use_container_width=True)

                    current_regime = regimes.iloc[-1]
                    st.markdown(f"<div style='text-align:center;padding:15px;'>"
                                f"<span style='font-family:Space Mono;font-size:1.3rem;"
                                f"color:{color_map.get(current_regime, '#5a6a7a')};'>"
                                f"Current Regime: {current_regime}</span></div>",
                                unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Regime Detection Error: {e}")
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: STRESS TEST
# ══════════════════════════════════════════════════════════════════════════════

if _ST:
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("### 🔥 Stress Testing")
        st.markdown("Simulate historical crisis scenarios on your portfolio.")

        crisis_options = list(CRISIS_PROFILES.keys())
        selected_crisis = st.selectbox("Select Crisis Scenario", crisis_options,
                                       format_func=lambda x: CRISIS_PROFILES[x]["name"],
                                       key="st_crisis")

        crisis = CRISIS_PROFILES[selected_crisis]
        st.markdown(f"**{crisis['name']}** — {crisis['description']}")
        st.markdown(f"Duration: {crisis['duration_days']} days · "
                    f"Peak Drawdown: {crisis['peak_drawdown']:.1%} · "
                    f"VIX Peak: {crisis['vix_peak']:.1f}")

        if st.button("🚀 Run Stress Test", key="run_stress"):
            with st.spinner("Simulating crisis..."):
                try:
                    tester = StressTester()
                    result = tester.run_scenario(selected_crisis)

                    if "equity_curve" in result:
                        eq = result["equity_curve"]
                        st_fig = go.Figure()
                        st_fig.add_trace(go.Scatter(
                            x=eq.index if hasattr(eq, 'index') else list(range(len(eq))),
                            y=eq.values if hasattr(eq, 'values') else eq,
                            name="Portfolio Value",
                            line=dict(color="#ff4566", width=2),
                            fill="tozeroy", fillcolor="rgba(255,69,102,0.06)",
                        ))
                        st_fig.update_layout(**CHART_LAYOUT, height=350,
                                              title=f"Stress Test: {crisis['name']}")
                        st.plotly_chart(st_fig, use_container_width=True)

                    st_m1, st_m2, st_m3, st_m4 = st.columns(4)
                    with st_m1:
                        st.metric("Total Return", f"{result.get('total_return_pct', 0):.2f}%")
                    with st_m2:
                        st.metric("Max Drawdown", f"{result.get('max_drawdown_pct', 0):.2f}%")
                    with st_m3:
                        st.metric("Final Capital", f"${result.get('final_capital', 100000):,.0f}")
                    with st_m4:
                        st.metric("Worst Day", f"{result.get('worst_day_pct', 0):.2f}%")

                    if result.get("phase_breakdown"):
                        st.markdown("#### Phase Breakdown")
                        phase_data = []
                        for phase, stats in result["phase_breakdown"].items():
                            phase_data.append({"Phase": phase,
                                               "Return (%)": stats["return_pct"],
                                               "Bars": stats["bars"]})
                        st.dataframe(pd.DataFrame(phase_data), use_container_width=True)

                except Exception as e:
                    st.error(f"Stress Test Error: {e}")
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: NEWS
# ══════════════════════════════════════════════════════════════════════════════

if _LLM:
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("### 📰 News & Sentiment")
        st.markdown("Aggregate financial news and sentiment scores.")

        if st.button("📰 Fetch Latest News", key="fetch_news"):
            with st.spinner("Collecting news..."):
                try:
                    collector = NewsCollector()
                    headlines = collector.collect(st.session_state.ticker)

                    if headlines:
                        for h in headlines:
                            score = h.get("score", 0)
                            if score > 0.15:
                                badge = '<span style="color:#00ffc8;">BULLISH</span>'
                            elif score < -0.15:
                                badge = '<span style="color:#ff4566;">BEARISH</span>'
                            else:
                                badge = '<span style="color:#ffd166;">NEUTRAL</span>'

                            st.markdown(
                                f'<div style="background:#131a24;border:1px solid rgba(0,255,200,0.12);'
                                f'border-radius:10px;padding:12px;margin-bottom:8px;">'
                                f'<div style="font-family:Space Mono;font-size:0.7rem;color:#5a6a7a;">'
                                f'{h.get("source", "")} · {h.get("date", "")} · {badge}</div>'
                                f'<div style="font-size:0.9rem;color:#e0e8f0;margin-top:4px;">'
                                f'{h.get("title", "")}</div>'
                                f'</div>', unsafe_allow_html=True)
                    else:
                        st.info("No news found for this ticker.")
                except Exception as e:
                    st.error(f"News Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

with tabs[tab_idx]:
    tab_idx += 1
    st.markdown("### 📋 Raw Indicator Data")

    display_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in [f"SMA_{sma_p}", f"EMA_{ema_p}", "RSI", "MACD", "MACD_Signal",
                "BB_Upper", "BB_Mid", "BB_Lower", "ATR", "ADX", "OBV", "VWAP", "CCI",
                "Returns", "Realized_Vol_21", "Z_Score_20"]:
        if col in df.columns:
            display_cols.append(col)
    st.dataframe(
        df[display_cols].tail(100).style
          .background_gradient(subset=["Close"], cmap="RdYlGn")
          .format(precision=3),
        use_container_width=True,
    )

    csv_data = df[display_cols].to_csv()
    st.download_button("📥 Download CSV", csv_data, f"{st.session_state.ticker}_data.csv",
                       "text/csv", key="download_csv")

    st.markdown("---")
    st.markdown("#### Data Summary")
    sum_cols = st.columns(4)
    with sum_cols[0]: st.metric("Rows", len(df))
    with sum_cols[1]: st.metric("Start", str(df.index[0])[:10])
    with sum_cols[2]: st.metric("End", str(df.index[-1])[:10])
    with sum_cols[3]:
        missing = df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum()
        st.metric("Missing Values", int(missing))


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div style="text-align:center;font-family:Space Mono;font-size:0.65rem;'
    'color:#2a3a4a;padding:20px 0 10px;">NeuroTrade · For educational purposes only · '
    'Not financial advice</div>',
    unsafe_allow_html=True,
)
