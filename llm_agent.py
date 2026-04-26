"""
NeuroTrade — LLM Agent for News + Macro Reasoning
llm_agent.py — DeepSeek R1 powered market analysis via Ollama.

Pipeline:
  1. Collect technical indicator state
  2. Collect news sentiment (Alpha Vantage / RSS)
  3. Summarize macro regime from FRED data
  4. Build structured prompt → DeepSeek R1 chain-of-thought
  5. Parse verdict: BUY / SELL / HOLD + confidence + reasoning
"""

import os
import re
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

warnings.filterwarnings("ignore")

# ── HTTP ─────────────────────────────────────────────────────────────────────
try:
    import requests
    _REQ = True
except ImportError:
    _REQ = False
    print("[llm_agent] requests not found — pip install requests")

# ── RSS feeds ────────────────────────────────────────────────────────────────
try:
    import feedparser
    _FEED = True
except ImportError:
    _FEED = False

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
class LLMConfig:
    """Configuration for the LLM reasoning agent."""
    model_name:      str   = "deepseek-r1"
    ollama_url:      str   = "http://localhost:11434"
    temperature:     float = 0.3
    max_tokens:      int   = 2048
    timeout:         int   = 120
    # news
    news_max:        int   = 10
    av_key:          str   = os.getenv("ALPHA_VANTAGE_KEY", "JB95ETWHJRT5AP7I")
    rss_feeds:       List[str] = field(default_factory=lambda: [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
    ])
    # signal thresholds
    confidence_min:  int   = 60


# ══════════════════════════════════════════════════════════════════════════════
#  NEWS COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

class NewsCollector:
    """Aggregates news headlines from Alpha Vantage sentiment API + RSS."""

    def __init__(self, cfg: LLMConfig = None):
        self.cfg = cfg or LLMConfig()

    def collect(self, ticker: str = "SPY") -> List[Dict]:
        """Returns list of {title, source, score, date}."""
        headlines = []
        headlines.extend(self._from_alphavantage(ticker))
        headlines.extend(self._from_rss())
        # deduplicate by title
        seen = set()
        unique = []
        for h in headlines:
            key = h["title"][:60].lower()
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique[:self.cfg.news_max]

    def _from_alphavantage(self, ticker: str) -> List[Dict]:
        if not _REQ or not self.cfg.av_key:
            return []
        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "NEWS_SENTIMENT", "tickers": ticker,
                        "limit": 20, "apikey": self.cfg.av_key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            items = []
            for art in data.get("feed", [])[:15]:
                score = float(art.get("overall_sentiment_score", 0))
                items.append({
                    "title": art.get("title", ""),
                    "source": art.get("source", ""),
                    "score": score,
                    "label": art.get("overall_sentiment_label", ""),
                    "date": art.get("time_published", "")[:10],
                })
            return items
        except Exception as e:
            print(f"  [AV News] Error: {e}")
            return []

    def _from_rss(self) -> List[Dict]:
        if not _FEED:
            return []
        items = []
        for url in self.cfg.rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    items.append({
                        "title": entry.get("title", ""),
                        "source": feed.feed.get("title", "RSS"),
                        "score": 0.0,
                        "label": "neutral",
                        "date": entry.get("published", "")[:10],
                    })
            except Exception:
                pass
        return items


# ══════════════════════════════════════════════════════════════════════════════
#  MACRO SUMMARIZER
# ══════════════════════════════════════════════════════════════════════════════

class MacroSummarizer:
    """Converts FRED macro DataFrame into natural language summary."""

    SERIES_NAMES = {
        "DGS10": "10Y Treasury Yield", "DGS2": "2Y Treasury Yield",
        "T10Y2Y": "Yield Curve (10Y-2Y)", "CPIAUCSL": "CPI",
        "VIXCLS": "VIX", "BAMLH0A0HYM2": "HY Credit Spread",
        "DFF": "Fed Funds Rate", "UNRATE": "Unemployment",
        "Yield_Curve": "Yield Curve Spread", "VIX_MA30": "VIX 30d MA",
        "CPI_YOY": "CPI Year-over-Year",
    }

    @classmethod
    def summarize(cls, df: pd.DataFrame) -> str:
        """Generate a text summary of current macro regime."""
        if df is None or df.empty:
            return "No macro data available."

        lines = []
        last = df.iloc[-1]

        # Check key series
        for col in df.columns:
            if col in cls.SERIES_NAMES:
                val = last.get(col, None)
                if val is not None and not np.isnan(val):
                    name = cls.SERIES_NAMES[col]
                    # calculate recent change
                    if len(df) > 21:
                        prev = df[col].iloc[-22]
                        if not np.isnan(prev) and prev != 0:
                            chg = val - prev
                            lines.append(f"• {name}: {val:.3f} ({chg:+.3f} vs 1mo ago)")
                            continue
                    lines.append(f"• {name}: {val:.3f}")

        # Regime assessment
        regime = []
        vix = last.get("VIXCLS", None)
        if vix is not None and not np.isnan(vix):
            if vix > 30:
                regime.append("FEAR regime (VIX elevated)")
            elif vix > 20:
                regime.append("CAUTIOUS regime (VIX moderate)")
            else:
                regime.append("CALM regime (VIX low)")

        curve = last.get("T10Y2Y", last.get("Yield_Curve", None))
        if curve is not None and not np.isnan(curve):
            if curve < 0:
                regime.append("Yield curve INVERTED (recession signal)")
            else:
                regime.append(f"Yield curve normal ({curve:.2f}%)")

        if regime:
            lines.append("\nRegime: " + " | ".join(regime))

        return "\n".join(lines) if lines else "Macro data present but no key series found."


# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL SUMMARIZER
# ══════════════════════════════════════════════════════════════════════════════

class TechnicalSummarizer:
    """Converts OHLCV+indicator DataFrame into text summary."""

    @staticmethod
    def summarize(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No price data available."

        last = df.iloc[-1]
        close = last.get("Close", 0)
        lines = [f"Current Price: {close:.2f}"]

        # Trend
        sma = last.get("SMA_20", None)
        ema = last.get("EMA_21", last.get("EMA_20", None))
        if sma and not np.isnan(sma):
            pos = "ABOVE" if close > sma else "BELOW"
            lines.append(f"• Price {pos} SMA20 ({sma:.2f})")
        if ema and not np.isnan(ema):
            pos = "ABOVE" if close > ema else "BELOW"
            lines.append(f"• Price {pos} EMA21 ({ema:.2f})")

        # Momentum
        rsi = last.get("RSI", None)
        if rsi and not np.isnan(rsi):
            zone = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            lines.append(f"• RSI: {rsi:.1f} ({zone})")

        macd = last.get("MACD", None)
        macd_sig = last.get("MACD_Signal", None)
        if macd is not None and macd_sig is not None:
            if not np.isnan(macd) and not np.isnan(macd_sig):
                cross = "BULLISH" if macd > macd_sig else "BEARISH"
                lines.append(f"• MACD: {macd:.4f} vs Signal: {macd_sig:.4f} ({cross})")

        # Trend strength
        adx = last.get("ADX", None)
        if adx and not np.isnan(adx):
            strength = "STRONG" if adx > 25 else "WEAK"
            lines.append(f"• ADX: {adx:.1f} ({strength} trend)")

        # Volatility
        atr = last.get("ATR", None)
        if atr and not np.isnan(atr):
            atr_pct = atr / close * 100
            lines.append(f"• ATR: {atr:.2f} ({atr_pct:.2f}% of price)")

        # Bollinger
        bb_u = last.get("BB_Upper", None)
        bb_l = last.get("BB_Lower", None)
        if bb_u and bb_l and not np.isnan(bb_u) and not np.isnan(bb_l):
            if close > bb_u:
                lines.append("• Price ABOVE upper Bollinger Band (overbought)")
            elif close < bb_l:
                lines.append("• Price BELOW lower Bollinger Band (oversold)")
            else:
                pct_b = (close - bb_l) / (bb_u - bb_l + 1e-9)
                lines.append(f"• Bollinger %B: {pct_b:.2f}")

        # Recent returns
        if "Close" in df.columns and len(df) > 5:
            ret_1d = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
            ret_5d = (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100
            lines.append(f"• Returns: 1d={ret_1d:+.2f}%  5d={ret_5d:+.2f}%")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  DEEPSEEK R1 AGENT
# ══════════════════════════════════════════════════════════════════════════════

class DeepSeekAgent:
    """Calls local Ollama DeepSeek R1 model for reasoning."""

    def __init__(self, cfg: LLMConfig = None):
        self.cfg = cfg or LLMConfig()
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama is running and model is present."""
        if self._available is not None:
            return self._available
        if not _REQ:
            self._available = False
            return False
        try:
            r = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            self._available = any(self.cfg.model_name in m for m in models)
            if not self._available:
                print(f"[LLM] Model '{self.cfg.model_name}' not found. "
                      f"Available: {models}")
                print(f"[LLM] Run: ollama pull {self.cfg.model_name}")
            return self._available
        except Exception as e:
            print(f"[LLM] Ollama not reachable: {e}")
            self._available = False
            return False

    def generate(self, prompt: str) -> str:
        """Send prompt to DeepSeek R1 and return response."""
        if not _REQ:
            return "[ERROR] requests library not available"

        payload = {
            "model": self.cfg.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }
        try:
            r = requests.post(
                f"{self.cfg.ollama_url}/api/generate",
                json=payload,
                timeout=self.cfg.timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except requests.exceptions.Timeout:
            return "[ERROR] Ollama request timed out"
        except Exception as e:
            return f"[ERROR] Ollama error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  MARKET REASONING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketVerdict:
    signal:     str    # BUY / SELL / HOLD
    confidence: int    # 0-100
    reasoning:  str    # chain-of-thought explanation
    raw_response: str  # full LLM output
    timestamp:  str    = ""
    ticker:     str    = ""


class MarketReasoningEngine:
    """
    Orchestrates the full LLM reasoning pipeline:
    Technical state + News + Macro → Structured prompt → DeepSeek R1 → Verdict.
    """

    PROMPT_TEMPLATE = """You are a senior quantitative analyst at a top hedge fund.
Analyze the following market data and provide a precise trading recommendation.

═══ TICKER: {ticker} ═══

── TECHNICAL INDICATORS ──
{technical}

── NEWS SENTIMENT ──
{news}

── MACRO REGIME ──
{macro}

═══════════════════════════

Based on this data, provide your analysis in EXACTLY this format:

SIGNAL: [BUY or SELL or HOLD]
CONFIDENCE: [0-100]
REASONING:
1. [Technical analysis point]
2. [Sentiment analysis point]
3. [Macro analysis point]
4. [Risk assessment]
5. [Final verdict with timeframe]
"""

    def __init__(self, cfg: LLMConfig = None):
        self.cfg = cfg or LLMConfig()
        self.agent = DeepSeekAgent(self.cfg)
        self.news_collector = NewsCollector(self.cfg)

    def analyze(self, df: pd.DataFrame,
                macro_df: pd.DataFrame = None,
                ticker: str = "UNKNOWN") -> MarketVerdict:
        """Run full reasoning pipeline."""

        # 1. Technical summary
        tech_summary = TechnicalSummarizer.summarize(df)

        # 2. News
        news_items = self.news_collector.collect(ticker)
        if news_items:
            news_text = "\n".join([
                f"• [{h['label']}] {h['title']} (score: {h['score']:.2f})"
                for h in news_items
            ])
        else:
            news_text = "No recent news available."

        # 3. Macro
        macro_text = MacroSummarizer.summarize(macro_df) if macro_df is not None else "No macro data."

        # 4. Build prompt
        prompt = self.PROMPT_TEMPLATE.format(
            ticker=ticker,
            technical=tech_summary,
            news=news_text,
            macro=macro_text,
        )

        # 5. Call LLM
        if not self.agent.is_available():
            return self._fallback_analysis(df, ticker)

        t0 = time.time()
        raw = self.agent.generate(prompt)
        dur = time.time() - t0
        print(f"  [LLM] Response in {dur:.1f}s ({len(raw)} chars)")

        # 6. Parse response
        return self._parse_response(raw, ticker)

    def _parse_response(self, raw: str, ticker: str) -> MarketVerdict:
        """Parse LLM output into structured verdict."""
        signal = "HOLD"
        confidence = 50
        reasoning = raw

        # extract SIGNAL
        sig_match = re.search(r"SIGNAL:\s*(BUY|SELL|HOLD)", raw, re.IGNORECASE)
        if sig_match:
            signal = sig_match.group(1).upper()

        # extract CONFIDENCE
        conf_match = re.search(r"CONFIDENCE:\s*(\d+)", raw)
        if conf_match:
            confidence = min(100, max(0, int(conf_match.group(1))))

        # extract REASONING
        reason_match = re.search(r"REASONING:\s*\n(.*)", raw, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return MarketVerdict(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=raw,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker=ticker,
        )

    def _fallback_analysis(self, df: pd.DataFrame, ticker: str) -> MarketVerdict:
        """Rule-based fallback when LLM is unavailable."""
        if df is None or df.empty:
            return MarketVerdict("HOLD", 30, "No data available", "",
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticker)

        last = df.iloc[-1]
        score = 0
        reasons = []

        # RSI
        rsi = last.get("RSI", 50)
        if not np.isnan(rsi):
            if rsi > 70:
                score -= 2; reasons.append(f"RSI {rsi:.0f} overbought")
            elif rsi < 30:
                score += 2; reasons.append(f"RSI {rsi:.0f} oversold")

        # MACD
        macd = last.get("MACD", 0)
        macd_s = last.get("MACD_Signal", 0)
        if not np.isnan(macd) and not np.isnan(macd_s):
            if macd > macd_s:
                score += 1; reasons.append("MACD bullish crossover")
            else:
                score -= 1; reasons.append("MACD bearish crossover")

        # Trend
        sma = last.get("SMA_20", None)
        close = last.get("Close", 0)
        if sma and not np.isnan(sma):
            if close > sma:
                score += 1; reasons.append("Price above SMA20")
            else:
                score -= 1; reasons.append("Price below SMA20")

        signal = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
        conf = min(90, 40 + abs(score) * 10)

        return MarketVerdict(
            signal=signal,
            confidence=conf,
            reasoning="[Fallback — LLM unavailable]\n" + "\n".join(f"• {r}" for r in reasons),
            raw_response="",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker=ticker,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATOR (backtester compatible)
# ══════════════════════════════════════════════════════════════════════════════

class LLMSignalGenerator:
    """Wraps MarketReasoningEngine for backtester-compatible signals."""

    def __init__(self, engine: MarketReasoningEngine, min_confidence: int = 60):
        self.engine = engine
        self.min_confidence = min_confidence
        self._cache = {}
        self._cache_interval = 20  # re-analyze every N bars

    def make_signal_func(self, ticker: str = "SPY",
                         macro_df: pd.DataFrame = None):
        gen = self

        def signal_func(df: pd.DataFrame, i: int, **kwargs):
            if i < 30:
                return None
            # only re-analyze every N bars
            last_i = gen._cache.get("last_i", -999)
            if (i - last_i) < gen._cache_interval and "verdict" in gen._cache:
                verdict = gen._cache["verdict"]
            else:
                try:
                    sub = df.iloc[:i+1]
                    verdict = gen.engine.analyze(sub, macro_df, ticker)
                    gen._cache["verdict"] = verdict
                    gen._cache["last_i"] = i
                except Exception:
                    return None

            if verdict.confidence < gen.min_confidence:
                return PositionSide.FLAT

            if verdict.signal == "BUY":
                return PositionSide.LONG
            elif verdict.signal == "SELL":
                return PositionSide.SHORT
            return PositionSide.FLAT

        return signal_func


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  llm_agent.py — LLM Reasoning Agent Self-Test")
    print("=" * 65)

    # Generate synthetic data
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 500 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, n)))
    df = pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": np.random.randint(1e6, 1e7, n).astype(float),
    }, index=dates)
    df["RSI"] = 50 + np.cumsum(np.random.normal(0, 2, n))
    df["RSI"] = df["RSI"].clip(10, 90)
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_21"] = df["Close"].ewm(span=21).mean()
    e12 = df["Close"].ewm(span=12).mean()
    e26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = e12 - e26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ADX"] = 25 + np.random.normal(0, 5, n)
    df["BB_Upper"] = df["SMA_20"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["SMA_20"] - 2 * df["Close"].rolling(20).std()
    df.dropna(inplace=True)

    # Test components
    print("\n  1. Technical Summarizer")
    print("  " + "-" * 40)
    tech = TechnicalSummarizer.summarize(df)
    print(f"  {tech[:200]}...")

    print("\n  2. Macro Summarizer")
    print("  " + "-" * 40)
    macro_test = pd.DataFrame({
        "VIXCLS": [18.5, 19.2, 20.1],
        "DGS10": [4.2, 4.3, 4.25],
        "T10Y2Y": [-0.1, -0.05, 0.02],
    }, index=pd.date_range("2024-01-01", periods=3))
    macro_summary = MacroSummarizer.summarize(macro_test)
    print(f"  {macro_summary}")

    print("\n  3. DeepSeek R1 Agent")
    print("  " + "-" * 40)
    cfg = LLMConfig()
    agent = DeepSeekAgent(cfg)
    available = agent.is_available()
    print(f"  Ollama available: {available}")

    print("\n  4. Market Reasoning Engine")
    print("  " + "-" * 40)
    engine = MarketReasoningEngine(cfg)
    verdict = engine.analyze(df, macro_test, "TEST")
    print(f"  Signal: {verdict.signal}")
    print(f"  Confidence: {verdict.confidence}%")
    print(f"  Reasoning: {verdict.reasoning[:200]}...")

    print("\n  ✓  LLM Agent self-test complete.\n")
