
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── data.py  ──────────────────────────────────────────────────────────────────
from data import (
    DataPipeline, YFinanceLoader, AlphaVantageLoader,
    FREDLoader, MacroFeatureBuilder, SyntheticDataGenerator,
    OHLCVCleaner, PolymarketLoader, KalshiLoader,
    CONFIG, FRED_SERIES, _YF_AVAILABLE, _FRED_AVAILABLE,
)

# ── indicators.py  ────────────────────────────────────────────────────────────
try:
    from indicators import add_all_indicators
    _IND = True
except ImportError:
    _IND = False

# ── backtester.py  ────────────────────────────────────────────────────────────
try:
    from backtester import (
        Backtester, BacktestConfig,
        macd_crossover_signal, rsi_mean_reversion_signal,
        sma_trend_signal, bollinger_breakout_signal,
    )
    _BT = True
except ImportError:
    _BT = False

# ── models.py  ────────────────────────────────────────────────────────────────
try:
    from models import DeepLearningModel, ModelConfig
    _DL = True
except ImportError:
    _DL = False


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroTrade",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  — dark terminal aesthetic, IBM Plex Mono + DM Serif Display
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display:ital@0;1&display=swap');

:root{
  --bg:#05070f; --s1:#090d18; --s2:#0e1422; --s3:#131b2e;
  --cyan:#38bdf8; --green:#4ade80; --red:#f87171;
  --amber:#fbbf24; --purple:#a78bfa; --orange:#fb923c;
  --text:#cbd5e1; --muted:#475569; --dim:#1e293b;
  --border:rgba(56,189,248,.10); --border2:rgba(56,189,248,.22);
  --mono:'IBM Plex Mono',monospace; --serif:'DM Serif Display',serif;
}

html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}

section[data-testid="stSidebar"]{background:var(--s1)!important;border-right:1px solid var(--border2);}
section[data-testid="stSidebar"] *{color:var(--text)!important;}

button[data-baseweb="tab"]{font-family:var(--mono)!important;font-size:.68rem!important;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)!important;padding:8px 14px!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--cyan)!important;border-bottom:2px solid var(--cyan)!important;}
div[data-testid="stTabs"]{border-bottom:1px solid var(--border);}

div[data-testid="metric-container"]{background:var(--s2);border:1px solid var(--border);border-radius:5px;padding:11px 15px;}
div[data-testid="metric-container"] label{font-family:var(--mono)!important;font-size:.58rem!important;letter-spacing:.15em;text-transform:uppercase;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:var(--mono)!important;font-size:1.2rem!important;font-weight:600;color:var(--cyan)!important;}
[data-testid="stMetricDelta"]{font-size:.68rem!important;}

div[data-baseweb="select"]>div,input[type="text"],input[type="number"]{background:var(--s2)!important;border-color:var(--border2)!important;color:var(--text)!important;border-radius:3px!important;font-family:var(--mono)!important;font-size:.8rem!important;}

.stButton>button{background:transparent!important;border:1px solid var(--cyan)!important;color:var(--cyan)!important;font-family:var(--mono)!important;font-size:.68rem!important;letter-spacing:.12em;text-transform:uppercase;border-radius:2px!important;transition:all .15s;}
.stButton>button:hover{background:rgba(56,189,248,.08)!important;}
.stButton>button[kind="primary"]{background:var(--cyan)!important;color:#000!important;font-weight:600!important;}

[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:4px;}
details summary{font-family:var(--mono)!important;font-size:.68rem!important;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)!important;}

.pill{display:inline-block;padding:2px 10px;border-radius:2px;font-family:var(--mono);font-size:.65rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;}
.buy {background:rgba(74,222,128,.12);color:#4ade80;border:1px solid rgba(74,222,128,.30);}
.sell{background:rgba(248,113,113,.12);color:#f87171;border:1px solid rgba(248,113,113,.30);}
.hold{background:rgba(251,191,36,.12); color:#fbbf24;border:1px solid rgba(251,191,36,.30);}

.ins{background:var(--s2);border:1px solid var(--border);border-left-width:3px;border-radius:3px;padding:10px 14px;margin-bottom:7px;}
.ins.bull{border-left-color:#4ade80;} .ins.bear{border-left-color:#f87171;} .ins.neut{border-left-color:#a78bfa;}
.ins-lbl{font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:3px;}
.ins-txt{font-size:.8rem;color:var(--text);line-height:1.5;}

.slbl{font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);padding:3px 0 7px;border-bottom:1px solid var(--border);margin-bottom:9px;}
.hdr-name{font-family:var(--serif);font-size:1.7rem;color:var(--cyan);}
.hdr-sub{font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);}

.dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:#4ade80;margin-right:5px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.stPlotlyChart{border-radius:4px;overflow:hidden;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:.9rem!important;max-width:100%!important;}
hr{border-color:var(--border)!important;margin:10px 0!important;}
::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
div[data-testid="stSlider"] label{font-size:.6rem!important;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)!important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY BASE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#07090f",
    font=dict(family="IBM Plex Mono,monospace", color="#475569", size=10),
    margin=dict(l=0, r=0, t=22, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(56,189,248,.12)",
                borderwidth=1, font=dict(size=9), orientation="h",
                yanchor="bottom", y=1.01, xanchor="left", x=0),
    xaxis=dict(gridcolor="rgba(255,255,255,.025)", zerolinecolor="rgba(255,255,255,.04)",
               rangeslider=dict(visible=False), showspikes=True,
               spikecolor="#38bdf8", spikethickness=1, spikedash="dot"),
    yaxis=dict(gridcolor="rgba(255,255,255,.025)", zerolinecolor="rgba(255,255,255,.04)",
               showspikes=True, spikecolor="#38bdf8", spikethickness=1),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#0e1422", bordercolor="rgba(56,189,248,.25)",
                    font=dict(family="IBM Plex Mono", color="#cbd5e1", size=10)),
)


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR FALLBACK  (used when indicators.py is absent)
# ══════════════════════════════════════════════════════════════════════════════
def _fallback_indicators(df: pd.DataFrame, sma=20, ema=21) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df[f"SMA_{sma}"] = c.rolling(sma).mean()
    df[f"EMA_{ema}"] = c.ewm(span=ema, adjust=False).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = e12 - e26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    df["BB_Upper"] = mid + 2*std
    df["BB_Mid"]   = mid
    df["BB_Lower"] = mid - 2*std
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    up = h.diff().clip(lower=0); dn = (-l.diff()).clip(lower=0)
    a14 = tr.rolling(14).mean()
    dip = 100 * up.rolling(14).mean() / a14
    din = 100 * dn.rolling(14).mean() / a14
    df["DI_Pos"] = dip; df["DI_Neg"] = din
    df["ADX"] = (100*(dip-din).abs()/(dip+din+1e-9)).rolling(14).mean()
    tp = (h+l+c)/3
    df["CCI"]  = (tp - tp.rolling(20).mean()) / (0.015*tp.rolling(20).std()+1e-9)
    lk = l.rolling(14).min(); hk = h.rolling(14).max()
    df["%K"] = 100*(c-lk)/(hk-lk+1e-9)
    df["%D"] = df["%K"].rolling(3).mean()
    df["OBV"]  = (np.sign(c.diff()).fillna(0)*v).cumsum() # type: ignore
    df["VWAP"] = (tp*v).cumsum()/v.cumsum()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL  +  INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def compute_signal(df, sma_p=20, ema_p=21):
    if df is None or len(df) < 2:
        return "HOLD", "#fbbf24", 0
    last  = df.iloc[-1]; score = 0
    rsi   = last.get("RSI", 50)
    if   rsi > 70: score -= 2
    elif rsi < 30: score += 2
    elif rsi > 55: score += 1
    elif rsi < 45: score -= 1
    m = last.get("MACD", 0); s = last.get("MACD_Signal", 0)
    score += 2 if m > s else -2
    if last.get("ADX", 0) > 25:
        score += 1 if last.get("DI_Pos",0) > last.get("DI_Neg",0) else -1
    c = last["Close"]
    if   c > last.get("BB_Upper", c*1.1): score -= 1
    elif c < last.get("BB_Lower", c*.9):  score += 1
    score += 1 if c > last.get(f"SMA_{sma_p}", c) else -1
    score += 1 if c > last.get(f"EMA_{ema_p}", c) else -1
    if   score >= 3:  return "BUY",  "#4ade80", score
    elif score <= -3: return "SELL", "#f87171", score
    else:             return "HOLD", "#fbbf24", score


def gen_insights(df):
    if df is None or len(df) < 2: return []
    last = df.iloc[-1]; prev = df.iloc[-2]; ins = []
    rsi = last.get("RSI", np.nan)
    if not np.isnan(rsi):
        if   rsi > 70: ins.append(("bear","RSI",f"RSI {rsi:.1f} — overbought. Mean-reversion risk."))
        elif rsi < 30: ins.append(("bull","RSI",f"RSI {rsi:.1f} — oversold. Watch for bounce."))
        else:          ins.append(("neut","RSI",f"RSI {rsi:.1f} — neutral, no extreme pressure."))
    m  = last.get("MACD",np.nan); s  = last.get("MACD_Signal",np.nan)
    pm = prev.get("MACD",np.nan); ps = prev.get("MACD_Signal",np.nan)
    if not any(np.isnan(x) for x in [m,s,pm,ps]):
        if   pm<=ps and m>s: ins.append(("bull","MACD","Bullish crossover — MACD above signal."))
        elif pm>=ps and m<s: ins.append(("bear","MACD","Bearish crossover — MACD below signal."))
        else: ins.append(("bull" if m>0 else "bear","MACD",
                          f"MACD {m:.3f} — {'bullish' if m>0 else 'bearish'} momentum."))
    adx = last.get("ADX",np.nan)
    if not np.isnan(adx):
        if adx > 25:
            d = "bull" if last.get("DI_Pos",0) > last.get("DI_Neg",0) else "bear"
            ins.append((d,"ADX",f"ADX {adx:.1f} — strong {'bullish' if d=='bull' else 'bearish'} trend."))
        else: ins.append(("neut","ADX",f"ADX {adx:.1f} — weak/ranging. Avoid trend strategies."))
    cl = last["Close"]
    bu = last.get("BB_Upper",np.nan); bl = last.get("BB_Lower",np.nan)
    if not any(np.isnan(x) for x in [bu,bl]):
        bw = bu-bl; pb = (cl-bl)/(bw+1e-9)
        if   cl > bu: ins.append(("bear","BB","Price above upper band — mean-reversion setup."))
        elif cl < bl: ins.append(("bull","BB","Price below lower band — potential reversal zone."))
        else:         ins.append(("neut","BB",
                                  f"%B {pb:.0%}.{' Squeeze!' if bw < last.get('BB_Mid',cl)*.03 else ''}"))
    return ins


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def main_chart(df, cfg):
    sma_p, ema_p = cfg["sma_p"], cfg["ema_p"]
    subs  = [s for s in ["vol","rsi","macd"] if cfg.get(f"show_{s}")]
    n     = 1 + len(subs)
    h     = [0.58] + [0.42/len(subs)]*len(subs) if subs else [1.0]
    total = sum(h); h = [x/total for x in h]
    fig   = make_subplots(rows=n, cols=1, shared_xaxes=True,
                          vertical_spacing=0.010, row_heights=h)
    idx = df.index

    fig.add_trace(go.Candlestick(
        x=idx, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#4ade80",width=1),fillcolor="rgba(74,222,128,.7)"),
        decreasing=dict(line=dict(color="#f87171",width=1),fillcolor="rgba(248,113,113,.7)"),
        whiskerwidth=.4,
    ), row=1, col=1)

    if cfg["show_sma"] and f"SMA_{sma_p}" in df:
        fig.add_trace(go.Scatter(x=idx, y=df[f"SMA_{sma_p}"], name=f"SMA{sma_p}",
            line=dict(color="#fbbf24",width=1.1,dash="dot")), row=1, col=1)

    if cfg["show_ema"] and f"EMA_{ema_p}" in df:
        fig.add_trace(go.Scatter(x=idx, y=df[f"EMA_{ema_p}"], name=f"EMA{ema_p}",
            line=dict(color="#a78bfa",width=1.3)), row=1, col=1)

    if cfg["show_bb"] and "BB_Upper" in df:
        fig.add_trace(go.Scatter(x=idx, y=df["BB_Upper"],
            line=dict(color="rgba(56,189,248,.35)",width=.8),showlegend=False), row=1,col=1)
        fig.add_trace(go.Scatter(x=idx, y=df["BB_Lower"], name="BB",
            fill="tonexty", fillcolor="rgba(56,189,248,.04)",
            line=dict(color="rgba(56,189,248,.35)",width=.8)), row=1,col=1)

    if cfg["show_vwap"] and "VWAP" in df:
        fig.add_trace(go.Scatter(x=idx, y=df["VWAP"], name="VWAP",
            line=dict(color="#fb923c",width=1.1,dash="dashdot")), row=1,col=1)

    if cfg["show_signals"] and "RSI" in df and "MACD" in df:
        buy  = (df["RSI"]<35)&(df["MACD"]>df["MACD_Signal"])
        sell = (df["RSI"]>65)&(df["MACD"]<df["MACD_Signal"])
        if buy.any():
            fig.add_trace(go.Scatter(x=idx[buy], y=df["Low"][buy]*.997,
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up",size=7,color="#4ade80",
                            line=dict(color="#000",width=.4))), row=1,col=1)
        if sell.any():
            fig.add_trace(go.Scatter(x=idx[sell], y=df["High"][sell]*1.003,
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down",size=7,color="#f87171",
                            line=dict(color="#000",width=.4))), row=1,col=1)

    sr = 2
    if cfg.get("show_vol") and "Volume" in df:
        vc = ["rgba(74,222,128,.45)" if c>=o else "rgba(248,113,113,.45)"
              for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=idx,y=df["Volume"],name="Vol",
            marker_color=vc,showlegend=False), row=sr,col=1); sr+=1

    if cfg.get("show_rsi") and "RSI" in df:
        fig.add_trace(go.Scatter(x=idx,y=df["RSI"],name="RSI",
            line=dict(color="#38bdf8",width=1.3),
            fill="tozeroy",fillcolor="rgba(56,189,248,.04)"), row=sr,col=1)
        fig.add_hline(y=70,line=dict(color="rgba(248,113,113,.4)",width=.8,dash="dot"),row=sr,col=1)
        fig.add_hline(y=30,line=dict(color="rgba(74,222,128,.4)",width=.8,dash="dot"),row=sr,col=1)
        fig.update_yaxes(range=[0,100],row=sr,col=1); sr+=1

    if cfg.get("show_macd") and "MACD" in df:
        hc = ["rgba(74,222,128,.6)" if v>=0 else "rgba(248,113,113,.6)"
              for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=idx,y=df["MACD_Hist"],name="Hist",
            marker_color=hc,showlegend=False), row=sr,col=1)
        fig.add_trace(go.Scatter(x=idx,y=df["MACD"],name="MACD",
            line=dict(color="#38bdf8",width=1.3)), row=sr,col=1)
        fig.add_trace(go.Scatter(x=idx,y=df["MACD_Signal"],name="Sig",
            line=dict(color="#fb923c",width=1,dash="dot")), row=sr,col=1)

    fig.update_layout(**LAYOUT, height=620+len(subs)*55, dragmode="zoom") # type: ignore
    for i in range(1, n+1):
        fig.update_xaxes(row=i,col=1,gridcolor="rgba(255,255,255,.022)",
                         rangeslider_visible=False)
        fig.update_yaxes(row=i,col=1,gridcolor="rgba(255,255,255,.022)")
    return fig


def equity_chart(equity: pd.Series, trades: list) -> go.Figure:
    fig = go.Figure()
    dd  = (equity - equity.cummax()) / equity.cummax() * 100
    fig.add_trace(go.Scatter(x=equity.index, y=dd, name="DD",
        fill="tozeroy", fillcolor="rgba(248,113,113,.07)",
        line=dict(color="#f87171",width=.8), yaxis="y2"))
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity",
        line=dict(color="#38bdf8",width=1.8),
        fill="tozeroy", fillcolor="rgba(56,189,248,.05)"))
    wins  = [t for t in trades if t.pnl > 0]
    loses = [t for t in trades if t.pnl <= 0]
    def _y(t): return float(equity.get(t.exit_date, equity.iloc[-1]))
    if wins:
        fig.add_trace(go.Scatter(x=[t.exit_date for t in wins],
            y=[_y(t) for t in wins], mode="markers", name="Win",
            marker=dict(symbol="circle",size=4,color="#4ade80",opacity=.7)))
    if loses:
        fig.add_trace(go.Scatter(x=[t.exit_date for t in loses],
            y=[_y(t) for t in loses], mode="markers", name="Loss",
            marker=dict(symbol="circle",size=4,color="#f87171",opacity=.7)))
    fig.update_layout(**LAYOUT, height=300,
        yaxis2=dict(overlaying="y",side="right",ticksuffix="%",showgrid=False,
                    tickfont=dict(color="#f87171",size=8),zeroline=False))
    return fig


def macro_line(s: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name=name,
        line=dict(color="#38bdf8",width=1.4),
        fill="tozeroy",fillcolor="rgba(56,189,248,.04)"))
    fig.update_layout(**LAYOUT, height=185, margin=dict(l=0,r=0,t=16,b=0))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k,v in dict(df=None, df_raw=None, ticker="AAPL", source="yfinance",
                macro_df=None, bt_results=None, loaded=False, health={}).items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hdr-name">⚡ NeuroTrade</div>', unsafe_allow_html=True)
    st.markdown('<div class="hdr-sub">Quant Intelligence Terminal</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="slbl">Data Source</div>', unsafe_allow_html=True)
    SOURCE = st.radio("", [
        "yfinance", "Alpha Vantage", "FRED Macro",
        "Prediction Markets", "Synthetic", "Upload CSV",
    ], label_visibility="collapsed")

    st.markdown("---")

    # per-source controls
    if SOURCE in ("yfinance", "Alpha Vantage"):
        st.markdown('<div class="slbl">Ticker</div>', unsafe_allow_html=True)
        ticker = st.text_input("", value="AAPL",
                               placeholder="AAPL  BTC-USD  RELIANCE.NS",
                               label_visibility="collapsed").upper().strip()
        TF_MAP = {"1M":("1mo","1h"),"3M":("3mo","1d"),"6M":("6mo","1d"),
                  "1Y":("1y","1d"),"2Y":("2y","1wk"),"5Y":("5y","1wk")}
        st.markdown('<div class="slbl">Timeframe</div>', unsafe_allow_html=True)
        tf = st.select_slider("", list(TF_MAP.keys()), value="1Y",
                              label_visibility="collapsed")
        tf_period, tf_interval = TF_MAP[tf]
        overlay_macro = st.checkbox("Overlay FRED macro", value=False)
        av_key = "JB95ETWHJRT5AP7I"
        if SOURCE == "Alpha Vantage":
            av_key = st.text_input("AV API Key", type="password",
                                   placeholder="paste Alpha Vantage key")

    elif SOURCE == "FRED Macro":
        fred_key = st.text_input("FRED API Key", type="password",
                                 value=CONFIG.fred_api_key or "",
                                 placeholder="fred.stlouisfed.org → free key")
        macro_sel = st.multiselect(
            "Series", list(FRED_SERIES.keys()),
            default=["DGS10","T10Y2Y","CPIAUCSL","VIXCLS","BAMLH0A0HYM2"],
            format_func=lambda x: f"{x} — {FRED_SERIES.get(x,'')}",
        )

    elif SOURCE == "Prediction Markets":
        pm_src = st.radio("Exchange", ["Kalshi","Polymarket"],
                          horizontal=True, label_visibility="collapsed")
        pm_cat = st.selectbox("Category",
                              ["all","economics","politics","financials"])

    elif SOURCE == "Synthetic":
        syn_n     = st.slider("Bars",      200, 3000, 1000, step=100)
        syn_price = st.slider("Start price", 100, 5000, 1000, step=100)
        syn_seed  = st.slider("Seed",       0, 100, 42)

    elif SOURCE == "Upload CSV":
        uploaded = st.file_uploader("OHLCV CSV", type=["csv"])

    st.markdown("---")

    if SOURCE not in ("FRED Macro","Prediction Markets"):
        st.markdown('<div class="slbl">Overlays</div>', unsafe_allow_html=True)
        show_sma  = st.checkbox("SMA",             value=True)
        sma_p     = st.slider("SMA period", 5,200,20,step=5) if show_sma else 20
        show_ema  = st.checkbox("EMA",             value=True)
        ema_p     = st.slider("EMA period", 5,200,21,step=1) if show_ema else 21
        show_bb   = st.checkbox("Bollinger Bands", value=True)
        show_vwap = st.checkbox("VWAP",            value=True)
        show_sig  = st.checkbox("Signal markers",  value=True)
        st.markdown('<div class="slbl">Sub-panels</div>', unsafe_allow_html=True)
        show_vol  = st.checkbox("Volume",  value=False)
        show_rsi  = st.checkbox("RSI",    value=True)
        show_macd = st.checkbox("MACD",   value=True)
    else:
        sma_p=20; ema_p=21
        show_sma=show_ema=show_bb=show_vwap=show_sig=False
        show_vol=show_rsi=show_macd=False

    st.markdown("---")
    load = st.button("⚡  Execute", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOAD  — every route goes through data.py
# ══════════════════════════════════════════════════════════════════════════════
if load or not st.session_state.loaded:
    with st.spinner("DataPipeline executing…"):
        try:
            if SOURCE == "yfinance":
                if not _YF_AVAILABLE:
                    st.error("pip install yfinance"); st.stop()
                days = {"1M":31,"3M":92,"6M":183,"1Y":365,"2Y":730,"5Y":1825}[tf]
                start_dt = (datetime.today()-timedelta(days=days)).strftime("%Y-%m-%d")
                df_raw = DataPipeline.get_ohlcv(
                    ticker=ticker, start=start_dt,
                    interval=tf_interval, source="yfinance")
                st.session_state.ticker = ticker
                st.session_state.source = "yfinance"
                if overlay_macro and CONFIG.fred_api_key and _FRED_AVAILABLE:
                    fl  = FREDLoader()
                    mac = fl.fetch_macro_panel(
                        ["DGS10","T10Y2Y","CPIAUCSL","VIXCLS"],
                        start=df_raw.index[0].strftime("%Y-%m-%d"))
                    df_raw = MacroFeatureBuilder.merge(df_raw, mac)
                    df_raw = MacroFeatureBuilder.add_macro_features(df_raw)
                    st.session_state.macro_df = mac

            elif SOURCE == "Alpha Vantage":
                if not av_key:
                    st.error("Alpha Vantage API key required."); st.stop()
                df_raw = DataPipeline.get_ohlcv(
                    ticker=ticker, source="alphavantage", av_key=av_key)
                st.session_state.ticker = ticker
                st.session_state.source = "alphavantage"

            elif SOURCE == "FRED Macro":
                key = fred_key or CONFIG.fred_api_key
                if not key:
                    st.error("FRED API key required — 13764e0aafcb6799faae047a5f291731"); st.stop()
                fl  = FREDLoader(api_key=key)
                mac = fl.fetch_macro_panel(macro_sel or ["DGS10","CPIAUCSL","VIXCLS"])
                st.session_state.macro_df = mac
                st.session_state.df       = mac
                st.session_state.df_raw   = mac
                st.session_state.ticker   = "MACRO"
                st.session_state.loaded   = True
                st.rerun()

            elif SOURCE == "Prediction Markets":
                pm_df = (KalshiLoader().get_markets(
                             category=None if pm_cat=="all" else pm_cat)
                         if pm_src=="Kalshi"
                         else PolymarketLoader().get_markets())
                st.session_state.df      = pm_df
                st.session_state.df_raw  = pm_df
                st.session_state.ticker  = pm_src
                st.session_state.loaded  = True
                st.rerun()

            elif SOURCE == "Synthetic":
                df_raw = SyntheticDataGenerator.generate(
                    n_bars=syn_n, start_price=float(syn_price),
                    seed=syn_seed, ticker="SYNTHETIC")
                st.session_state.ticker = "SYNTHETIC"
                st.session_state.source = "synthetic"

            elif SOURCE == "Upload CSV":
                if uploaded is None:
                    st.warning("Upload a CSV file first."); st.stop()
                df_raw = OHLCVCleaner.clean(
                    pd.read_csv(uploaded, index_col=0, parse_dates=True), "CUSTOM")
                st.session_state.ticker = "CUSTOM"
                st.session_state.source = "csv"

            # apply indicators
            df_ind = (add_all_indicators(df_raw.copy()) if _IND
                      else _fallback_indicators(df_raw.copy(), sma_p, ema_p))
            st.session_state.df_raw  = df_raw
            st.session_state.df      = df_ind
            st.session_state.health  = OHLCVCleaner.validate(df_raw)
            st.session_state.loaded  = True

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            with st.expander("Traceback"): st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION MARKETS  VIEW
# ══════════════════════════════════════════════════════════════════════════════
if SOURCE == "Prediction Markets" and st.session_state.df is not None:
    pm = st.session_state.df
    st.markdown(f'<div class="hdr-name">Prediction Markets — {st.session_state.ticker}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="hdr-sub">{len(pm)} markets</div>', unsafe_allow_html=True)
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Markets", len(pm))
    if "volume" in pm.columns:
        with c2: st.metric("Total Volume", f"${pm['volume'].sum():,.0f}")
    if "implied_prob" in pm.columns:
        with c3: st.metric("Avg Implied Prob", f"{pm['implied_prob'].mean():.1%}")
    sort_by = "implied_prob" if "implied_prob" in pm.columns else pm.columns[0]
    st.dataframe(pm.sort_values(sort_by, ascending=False).head(60),
                 use_container_width=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  FRED MACRO  VIEW
# ══════════════════════════════════════════════════════════════════════════════
if SOURCE == "FRED Macro" and st.session_state.macro_df is not None:
    mac = st.session_state.macro_df
    st.markdown('<div class="hdr-name">FRED Macro Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hdr-sub">{len(mac.columns)} series · '
                f'{str(mac.index[0].date())} → {str(mac.index[-1].date())}</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    cols = st.columns(min(4,len(mac.columns)))
    for i,col in enumerate(mac.columns[:4]):
        vals = mac[col].dropna()
        if len(vals)>=2:
            with cols[i]:
                st.metric(FRED_SERIES.get(col,col)[:22],
                          f"{vals.iloc[-1]:.3f}",
                          f"{vals.iloc[-1]-vals.iloc[-2]:+.3f}")
    st.markdown("---")
    picks = st.multiselect("Plot series", mac.columns.tolist(),
                           default=mac.columns[:3].tolist())
    for col in picks:
        st.markdown(f'<div class="slbl">{FRED_SERIES.get(col,col)}</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(macro_line(mac[col].dropna(), col),
                        use_container_width=True, config={"displaylogo":False})
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  GUARD — need OHLCV data beyond this point
# ══════════════════════════════════════════════════════════════════════════════
df = st.session_state.df
if df is None or (hasattr(df,"empty") and df.empty):
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
    justify-content:center;height:72vh;gap:10px;opacity:.4;">
      <div style="font-size:3rem;">⚡</div>
      <div style="font-family:'DM Serif Display';font-size:1.5rem;color:#38bdf8;">NeuroTrade</div>
      <div style="font-size:.62rem;letter-spacing:.2em;text-transform:uppercase;
      color:#334155;">Select a source and click Execute</div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
last = float(df["Close"].iloc[-1])
prev = float(df["Close"].iloc[-2]) if len(df)>1 else last
chg  = last - prev; chgp = chg/prev*100
sig, sig_col, sig_score = compute_signal(df, sma_p, ema_p)
arrow = "▲" if chg>=0 else "▼"
pcol  = "#4ade80" if chg>=0 else "#f87171"
bcls  = {"BUY":"buy","SELL":"sell","HOLD":"hold"}[sig]

ha, hb, hc = st.columns([4,2,1])
with ha:
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:12px;padding:4px 0;">'
        f'<span style="font-family:\'DM Serif Display\';font-size:1.9rem;color:#38bdf8;">'
        f'{st.session_state.ticker}</span>'
        f'<span style="font-size:1.35rem;font-weight:600;color:{pcol};">{last:,.2f}</span>'
        f'<span style="font-size:.8rem;color:{pcol};">'
        f'{arrow} {abs(chg):.2f} ({abs(chgp):.2f}%)</span>'
        f'&nbsp;<span class="pill {bcls}">{sig}</span>'
        f'</div>', unsafe_allow_html=True)
with hb:
    h_ = st.session_state.health
    st.markdown(
        f'<div style="font-size:.6rem;color:#475569;padding-top:10px;">'
        f'<span class="dot"></span>'
        f'{h_.get("rows","?")} bars · {h_.get("start","?")} → {h_.get("end","?")}'
        f'</div>', unsafe_allow_html=True)
with hc:
    st.markdown(
        f'<div style="font-size:.6rem;color:#475569;padding-top:10px;text-align:right;">'
        f'score {sig_score:+d}</div>', unsafe_allow_html=True)

st.markdown("<div style='margin:2px 0 8px;'></div>", unsafe_allow_html=True)

# ── metrics strip ─────────────────────────────────────────────────────────────
mc = st.columns(7)
for i,(label,col) in enumerate([("RSI","RSI"),("MACD","MACD"),("ADX","ADX"),
                                  ("ATR","ATR"),("VWAP","VWAP"),("CCI","CCI"),("OBV","OBV")]):
    with mc[i]:
        if col in df.columns:
            v = float(df[col].iloc[-1])
            p = float(df[col].iloc[-2]) if len(df)>1 else v
            st.metric(label, f"{v:.2f}",
                      f"{v-p:+.2f}" if col not in ("OBV","VWAP") else None)
        else:
            st.metric(label,"—")

st.markdown("<div style='margin:4px 0 8px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
T_CHART, T_INSIGHT, T_BT, T_DL, T_MACRO, T_DATA = st.tabs(
    ["Chart","Insights","Backtest","DL Models","Macro","Data"])


# ── CHART ─────────────────────────────────────────────────────────────────────
with T_CHART:
    cfg = dict(show_sma=show_sma, sma_p=sma_p, show_ema=show_ema, ema_p=ema_p,
               show_bb=show_bb, show_vwap=show_vwap, show_signals=show_sig,
               show_vol=show_vol, show_rsi=show_rsi, show_macd=show_macd)
    st.plotly_chart(main_chart(df, cfg), use_container_width=True,
                    config={"displayModeBar":True,"scrollZoom":True,
                            "displaylogo":False,
                            "modeBarButtonsToRemove":["lasso2d","select2d"]})

    if "%K" in df:
        with st.expander("Stochastic"):
            sf = go.Figure()
            sf.add_trace(go.Scatter(x=df.index,y=df["%K"],name="%K",
                line=dict(color="#38bdf8",width=1.3)))
            sf.add_trace(go.Scatter(x=df.index,y=df["%D"],name="%D",
                line=dict(color="#fbbf24",width=1.1,dash="dot")))
            sf.add_hline(y=80,line=dict(color="rgba(248,113,113,.4)",width=.8,dash="dot"))
            sf.add_hline(y=20,line=dict(color="rgba(74,222,128,.4)",width=.8,dash="dot"))
            sf.update_layout(**LAYOUT,height=175)
            st.plotly_chart(sf,use_container_width=True,config={"displaylogo":False})

    if "ADX" in df:
        with st.expander("ADX — Trend Strength"):
            af = go.Figure()
            af.add_trace(go.Scatter(x=df.index,y=df["ADX"],name="ADX",
                line=dict(color="#fbbf24",width=1.6)))
            if "DI_Pos" in df:
                af.add_trace(go.Scatter(x=df.index,y=df["DI_Pos"],name="+DI",
                    line=dict(color="#4ade80",width=1,dash="dot")))
                af.add_trace(go.Scatter(x=df.index,y=df["DI_Neg"],name="-DI",
                    line=dict(color="#f87171",width=1,dash="dot")))
            af.add_hline(y=25,line=dict(color="rgba(251,191,36,.4)",width=.8,dash="dot"))
            af.update_layout(**LAYOUT,height=175)
            st.plotly_chart(af,use_container_width=True,config={"displaylogo":False})


# ── INSIGHTS ──────────────────────────────────────────────────────────────────
with T_INSIGHT:
    insights = gen_insights(df)
    if not insights:
        st.info("Load data to generate insights.")
    else:
        cols = st.columns(3)
        for i,(kind,title,body) in enumerate(insights):
            with cols[i%3]:
                st.markdown(
                    f'<div class="ins {kind}">'
                    f'<div class="ins-lbl">{title}</div>'
                    f'<div class="ins-txt">{body}</div>'
                    f'</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="slbl">Data health report</div>', unsafe_allow_html=True)
    hc2 = st.columns(4)
    for i,(k,v) in enumerate(st.session_state.health.items()):
        with hc2[i%4]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.72rem;padding:3px 0;border-bottom:1px solid '
                f'rgba(56,189,248,.06);">'
                f'<span style="color:#475569">{k.replace("_"," ")}</span>'
                f'<span style="color:#cbd5e1">{v}</span></div>',
                unsafe_allow_html=True)


# ── BACKTEST ──────────────────────────────────────────────────────────────────
with T_BT:
    if not _BT:
        st.warning("backtester.py not found — place it in the same directory.")
    else:
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown('<div class="slbl">Strategy</div>', unsafe_allow_html=True)
            strategy  = st.selectbox("Signal", [
                "MACD Crossover","RSI Mean Reversion","SMA Trend","Bollinger Breakout"])
            init_cap  = st.number_input("Capital ($)", value=100_000, step=10_000)
            allow_sht = st.checkbox("Allow shorting", value=True)
        with bc2:
            st.markdown('<div class="slbl">Risk parameters</div>', unsafe_allow_html=True)
            comm  = st.slider("Commission %",    0.0,1.0, 0.10,.01)/100
            slip  = st.slider("Slippage %",      0.0,0.5, 0.05,.01)/100
            sl    = st.slider("Stop loss %",     0.0,10., 3.0, .5 )/100
            tp    = st.slider("Take profit %",   0.0,20., 6.0, .5 )/100
            trail = st.slider("Trailing stop %", 0.0,10., 2.5, .5 )/100

        run_bt = st.button("Run Backtest", use_container_width=True)

        if run_bt:
            SIG_MAP = {
                "MACD Crossover":     macd_crossover_signal,
                "RSI Mean Reversion": rsi_mean_reversion_signal,
                "SMA Trend":          sma_trend_signal,
                "Bollinger Breakout": bollinger_breakout_signal,
            }
            cfg_bt = BacktestConfig(
                initial_capital   = float(init_cap),
                commission_pct    = comm,
                slippage_pct      = slip,
                stop_loss_pct     = sl    or None,
                take_profit_pct   = tp    or None,
                trailing_stop_pct = trail or None,
                allow_shorting    = allow_sht,
                risk_free_rate    = 0.06,
            )
            with st.spinner(f"{strategy} · {len(df.dropna())} bars…"):
                res = Backtester(cfg_bt).run(df.dropna().copy(), SIG_MAP[strategy])
                st.session_state.bt_results = res

        if st.session_state.bt_results:
            r = st.session_state.bt_results
            st.markdown("---")
            r1,r2,r3,r4,r5,r6 = st.columns(6)
            with r1: st.metric("Return",    f"{r['total_return_pct']:.2f}%",
                                f"Ann {r['annualized_return_pct']:.2f}%")
            with r2: st.metric("Sharpe",  f"{r['sharpe_ratio']:.3f}")
            with r3: st.metric("Sortino", f"{r['sortino_ratio']:.3f}")
            with r4: st.metric("Max DD",  f"{r['max_drawdown_pct']:.2f}%")
            with r5: st.metric("Win %",   f"{r['win_rate_pct']:.1f}%",
                                f"{r['total_trades']} trades")
            with r6: st.metric("PF",      f"{r['profit_factor']:.3f}")

            eq = r["equity_curve"]["equity"]
            st.plotly_chart(equity_chart(eq, r["trades"]),
                            use_container_width=True, config={"displaylogo":False})

            with st.expander("Full Risk Report"):
                rc = st.columns(4)
                for i,(lbl,val) in enumerate([
                    ("Calmar",    f"{r['calmar_ratio']:.3f}"),
                    ("Omega",     f"{r['omega_ratio']:.3f}"),
                    ("VaR 95%",   f"{r['var_95_pct']:.2f}%"),
                    ("CVaR 95%",  f"{r['cvar_95_pct']:.2f}%"),
                    ("Ulcer",     f"{r['ulcer_index']:.4f}"),
                    ("Tail",      f"{r['tail_ratio']:.3f}"),
                    ("Kelly %",   f"{r['kelly_criterion_pct']:.1f}%"),
                    ("Expect.",   f"${r['expectancy_per_trade']:,.0f}"),
                    ("Avg Win",   f"${r['avg_win']:,.0f}"),
                    ("Avg Loss",  f"${r['avg_loss']:,.0f}"),
                    ("Max CW",    str(r["max_consecutive_wins"])),
                    ("Max CL",    str(r["max_consecutive_losses"])),
                ]):
                    with rc[i%4]: st.metric(lbl, val)

            with st.expander("Regime breakdown"):
                reg = r.get("regime_breakdown",{})
                if reg:
                    st.dataframe(pd.DataFrame([
                        {"Regime":k,"Trades":v["n_trades"],
                         "PnL":f"${v['total_pnl']:,.0f}",
                         "Avg":f"${v['avg_pnl']:,.0f}"}
                        for k,v in reg.items()
                    ]), use_container_width=True)

            exits = r.get("exit_reasons",{})
            if exits:
                ef = go.Figure(go.Bar(x=list(exits.keys()),y=list(exits.values()),
                    marker_color="#38bdf8"))
                ef.update_layout(**LAYOUT,height=175)
                st.plotly_chart(ef,use_container_width=True,config={"displaylogo":False})


# ── DL MODELS TAB ─────────────────────────────────────────────────────────────
with T_DL:
    if not _DL:
        st.warning("models.py not found — place it in the same directory.\n\n"
                   "Requires: `pip install torch`")
    else:
        st.markdown('<div class="slbl">Deep Learning Models</div>', unsafe_allow_html=True)

        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            dl_arch = st.selectbox("Architecture", ["LSTM", "TCN", "TFT"],
                                   help="LSTM: recurrent memory cells\n"
                                        "TCN: dilated causal convolutions\n"
                                        "TFT: transformer with variable selection")
            dl_epochs = st.slider("Epochs", 5, 200, 50, 5)
            dl_seq = st.slider("Sequence length (bars)", 10, 100, 30, 5)
        with dl_c2:
            dl_lr = st.select_slider("Learning rate",
                                     options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                     value=1e-3)
            dl_batch = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
            dl_horizon = st.selectbox("Forecast horizon (bars)", [1, 3, 5], index=0)

        dl_long_th = 0.001
        dl_short_th = -0.001
        with st.expander("Signal thresholds"):
            dl_long_th = st.number_input("Long threshold (return)", value=0.001,
                                          format="%.4f", step=0.0005)
            dl_short_th = st.number_input("Short threshold (return)", value=-0.001,
                                           format="%.4f", step=0.0005)

        run_dl = st.button("Train & Predict", use_container_width=True, key="run_dl")

        if run_dl:
            dl_cfg = ModelConfig(
                sequence_length=dl_seq,
                forecast_horizon=dl_horizon,
                epochs=dl_epochs,
                batch_size=dl_batch,
                learning_rate=dl_lr,
                long_threshold=dl_long_th,
                short_threshold=dl_short_th,
            )
            dl_model = DeepLearningModel(dl_arch.lower(), dl_cfg)

            with st.spinner(f"Training {dl_arch} on {len(df)} bars ..."):
                try:
                    history = dl_model.train(df.dropna().copy(), verbose=False)
                    preds = dl_model.predict(df.dropna().copy())
                    signal = dl_model.get_signal(df.dropna().copy())
                    st.session_state.dl_model = dl_model
                    st.session_state.dl_history = history
                    st.session_state.dl_preds = preds
                    st.session_state.dl_signal = signal
                    st.session_state.dl_arch = dl_arch
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    with st.expander("Traceback"): st.exception(e)

        if st.session_state.get("dl_model") is not None:
            dl_model = st.session_state.dl_model
            history = st.session_state.dl_history
            preds = st.session_state.dl_preds
            signal = st.session_state.dl_signal
            arch_name = st.session_state.dl_arch

            st.markdown("---")

            # signal display
            sig_str = signal.value.upper() if signal else "FLAT"
            sig_color = {"long": "#4ade80", "short": "#f87171"}.get(
                signal.value if signal else "", "#fbbf24")
            st.markdown(
                f'<div style="text-align:center;padding:8px;">' \
                f'<span style="font-family:var(--font-mono);font-size:1.2rem;' \
                f'color:{sig_color};font-weight:700;">' \
                f'{arch_name} Signal: {sig_str}</span></div>',
                unsafe_allow_html=True)

            # metrics
            dm1, dm2, dm3, dm4 = st.columns(4)
            with dm1:
                st.metric("Final Train Loss", f"{history['train_loss'][-1]:.6f}")
            with dm2:
                st.metric("Final Val Loss", f"{history['val_loss'][-1]:.6f}")
            with dm3:
                best_val = min(history['val_loss'])
                st.metric("Best Val Loss", f"{best_val:.6f}")
            with dm4:
                st.metric("Predictions", f"{len(preds):,}")

            # training loss chart
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                y=history["train_loss"], name="Train",
                line=dict(color="#38bdf8", width=1.5)))
            loss_fig.add_trace(go.Scatter(
                y=history["val_loss"], name="Val",
                line=dict(color="#fbbf24", width=1.5)))
            loss_fig.update_layout(**LAYOUT, height=250,
                                   xaxis_title="Epoch", yaxis_title="MSE Loss")
            st.plotly_chart(loss_fig, use_container_width=True,
                            config={"displaylogo": False})

            # predictions overlay on price
            pred_df = df.dropna().copy()
            seq_len = dl_model.config.sequence_length
            if len(preds) > 0 and len(preds) <= len(pred_df):
                pred_idx = pred_df.index[-len(preds):]
                pred_returns = pd.Series(preds, index=pred_idx)
                pred_price = pred_df["Close"].loc[pred_idx] * (1 + pred_returns)

                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(
                    x=pred_df.index, y=pred_df["Close"],
                    name="Actual", line=dict(color="#e2e8f0", width=1.2)))
                price_fig.add_trace(go.Scatter(
                    x=pred_idx, y=pred_price,
                    name=f"{arch_name} Predicted",
                    line=dict(color="#7b61ff", width=1.5, dash="dot")))

                # shade signal regions
                long_mask = pred_returns > dl_model.config.long_threshold
                short_mask = pred_returns < dl_model.config.short_threshold
                if long_mask.any():
                    long_dates = pred_idx[long_mask]
                    price_fig.add_trace(go.Scatter(
                        x=long_dates,
                        y=pred_df["Close"].loc[long_dates],
                        mode="markers", name="Long Signal",
                        marker=dict(color="#4ade80", size=5, symbol="triangle-up")))
                if short_mask.any():
                    short_dates = pred_idx[short_mask]
                    price_fig.add_trace(go.Scatter(
                        x=short_dates,
                        y=pred_df["Close"].loc[short_dates],
                        mode="markers", name="Short Signal",
                        marker=dict(color="#f87171", size=5, symbol="triangle-down")))

                price_fig.update_layout(**LAYOUT, height=350)
                st.plotly_chart(price_fig, use_container_width=True,
                                config={"displaylogo": False})

            # predicted returns distribution
            with st.expander("Predicted returns distribution"):
                hist_fig = go.Figure(go.Histogram(
                    x=preds, nbinsx=50,
                    marker_color="#7b61ff", opacity=0.7))
                hist_fig.add_vline(x=dl_model.config.long_threshold,
                                   line=dict(color="#4ade80", dash="dash", width=1))
                hist_fig.add_vline(x=dl_model.config.short_threshold,
                                   line=dict(color="#f87171", dash="dash", width=1))
                hist_fig.update_layout(**LAYOUT, height=200,
                                       xaxis_title="Predicted Return")
                st.plotly_chart(hist_fig, use_container_width=True,
                                config={"displaylogo": False})

            # feature importance (TFT only)
            if dl_model.arch == "tft":
                fi = dl_model.get_feature_importance()
                if fi:
                    with st.expander("Feature importance (TFT)"):
                        fi_sorted = dict(sorted(fi.items(),
                                                key=lambda x: x[1], reverse=True)[:15])
                        fi_fig = go.Figure(go.Bar(
                            x=list(fi_sorted.values()),
                            y=list(fi_sorted.keys()),
                            orientation="h",
                            marker_color="#7b61ff"))
                        fi_fig.update_layout(**LAYOUT, height=350)
                        st.plotly_chart(fi_fig, use_container_width=True,
                                        config={"displaylogo": False})


# ── MACRO TAB ─────────────────────────────────────────────────────────────────
with T_MACRO:
    mac_cols = [c for c in df.columns if c in FRED_SERIES or c in (
        "Yield_Curve","Real_Rate_10Y","HY_Spread_Z","CPI_YOY","VIX_MA30","Curve_Inverted")]
    if not mac_cols:
        st.info("No macro data loaded.\n\n"
                "• Enable **Overlay FRED macro** in sidebar (needs FRED API key)\n"
                "• Or switch source to **FRED Macro**")
    else:
        st.markdown(f'<div class="slbl">{len(mac_cols)} macro columns overlaid</div>',
                    unsafe_allow_html=True)
        for col in mac_cols[:10]:
            st.markdown(f'<div class="slbl">{FRED_SERIES.get(col,col)}</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(macro_line(df[col].dropna(), FRED_SERIES.get(col,col)),
                            use_container_width=True, config={"displaylogo":False})


# ── DATA TAB ──────────────────────────────────────────────────────────────────
with T_DATA:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="slbl">Data health</div>', unsafe_allow_html=True)
        for k,v in st.session_state.health.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.72rem;padding:3px 0;border-bottom:1px solid rgba(56,189,248,.05);">'
                f'<span style="color:#475569">{k.replace("_"," ")}</span>'
                f'<span style="color:#cbd5e1">{v}</span></div>',
                unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="slbl">Pipeline status</div>', unsafe_allow_html=True)
        for k,v,col in [
            ("data.py",       "DataPipeline",                            "#4ade80"),
            ("Source",        st.session_state.source,                   "#38bdf8"),
            ("indicators.py", "loaded" if _IND else "fallback",
             "#4ade80" if _IND else "#fbbf24"),
            ("backtester.py", "loaded" if _BT  else "not found",
             "#4ade80" if _BT  else "#f87171"),
            ("yfinance",      "ok" if _YF_AVAILABLE   else "missing",
             "#4ade80" if _YF_AVAILABLE   else "#f87171"),
            ("fredapi",       "ok" if _FRED_AVAILABLE else "missing",
             "#4ade80" if _FRED_AVAILABLE else "#f87171"),
            ("Columns",       str(len(df.columns)),                      "#38bdf8"),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.72rem;padding:3px 0;border-bottom:1px solid rgba(56,189,248,.05);">'
                f'<span style="color:#475569">{k}</span>'
                f'<span style="color:{col}">{v}</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    disp = ["Open","High","Low","Close","Volume"]
    for c in [f"SMA_{sma_p}",f"EMA_{ema_p}","RSI","MACD","MACD_Signal",
              "BB_Upper","BB_Lower","ATR","ADX","OBV","VWAP","CCI"]:
        if c in df.columns: disp.append(c)
    st.dataframe(
        df[disp].tail(100)
          .style.background_gradient(subset=["Close"],cmap="RdYlGn")
          .format(precision=3),
        use_container_width=True)


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;font-size:.55rem;color:#0f172a;'
    'letter-spacing:.12em;text-transform:uppercase;padding:18px 0 6px;">'
    'NeuroTrade · Research purposes only · Not financial advice'
    '</div>', unsafe_allow_html=True)
