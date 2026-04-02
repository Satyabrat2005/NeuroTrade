import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    print("[data.py] yfinance not installed — pip install yfinance")

try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    print("[data.py] fredapi not installed — pip install fredapi")
    
# CONFIG  —  put your API keys here or in .env
@dataclass
class APIConfig:
    """Central store for all API keys and default settings."""
    # keys 
    fred_api_key:         str  = os.getenv("FRED_API_KEY", "")
    alpha_vantage_key:    str  = os.getenv("ALPHA_VANTAGE_KEY", "")
    polygon_key:          str  = os.getenv("POLYGON_KEY", "")

# defaults 
    default_ticker:       str  = "AAPL"
    default_start:        str  = "2018-01-01"
    default_end:          str  = datetime.today().strftime("%Y-%m-%d")
    default_interval:     str  = "1d"          # 1m 5m 15m 30m 1h 1d 1wk
    cache_dir:            str  = "./cache"
    use_cache:            bool = True
    request_delay:        float = 0.3          # seconds between API calls


CONFIG = APIConfig()

# CACHE HELPERS
def _cache_path(key: str) -> str:
    os.makedirs(CONFIG.cache_dir, exist_ok=True)
    safe = key.replace("/", "_").replace(" ", "_").replace(":", "_")
    return os.path.join(CONFIG.cache_dir, f"{safe}.parquet")

def _cache_save(df: pd.DataFrame, key: str):
    try:
        df.to_parquet(_cache_path(key))
    except Exception:
        pass


def _cache_load(key: str, max_age_hours: int = 4) -> Optional[pd.DataFrame]:
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    age = (time.time() - os.path.getmtime(p)) / 3600
    if age > max_age_hours:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None

# CORE OHLCV CLEANER  — shared by ALL sources
class OHLCVCleaner:
    """
    Takes any raw DataFrame and returns a clean, standardised OHLCV df
    ready for indicators.py → add_all_indicators(df).

    Contract:
      - Index    : pd.DatetimeIndex, tz-naive, sorted ascending
      - Columns  : Open, High, Low, Close, Volume  (float64)
      - No NaN rows (forward-fill then drop)
      - No duplicate timestamps
      - Prices validated: High >= Low, all > 0
    """
    REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

    @classmethod
    def clean(cls, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        df = df.copy()

        # 1. standardise column names
        df.columns = [c.strip().title() for c in df.columns]
        rename = {
            "Adj Close": "Close", "Adjusted_Close": "Close",
            "Adj_Close": "Close", "4. Close": "Close",
            "1. Open": "Open", "2. High": "High",
            "3. Low": "Low", "5. Volume": "Volume",
        }
        df.rename(columns=rename, inplace=True)
        rename = {
            "Adj Close": "Close", "Adjusted_Close": "Close",
            "Adj_Close": "Close", "4. Close": "Close",
            "1. Open": "Open", "2. High": "High",
            "3. Low": "Low", "5. Volume": "Volume",
        }
        df.rename(columns=rename, inplace=True)

        # prefer Adj Close over Close when both exist
        if "Adj Close" in df.columns and "Close" in df.columns:
            df["Close"] = df["Adj Close"]
            df.drop(columns=["Adj Close"], inplace=True, errors="ignore")

        #index → DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        # keep only OHLCV 
        missing = [c for c in cls.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"[OHLCVCleaner] Missing columns: {missing}  (ticker={ticker})")
        df = df[cls.REQUIRED].astype(float)

        # fill / drop NaN 
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        # sanity checks
        bad_price = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        bad_hl    = df["High"] < df["Low"]
        bad_rows  = bad_price | bad_hl
        if bad_rows.sum() > 0:
            print(f"[OHLCVCleaner] Dropping {bad_rows.sum()} invalid rows for {ticker}")
            df = df[~bad_rows]

        #fix High/Low if OHLC data has them inverted
        df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
        df["Low"]  = df[["Open", "High", "Low", "Close"]].min(axis=1)

        #add useful derived columns (lightweight)
        df["Returns"]     = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["HL_Spread"]   = (df["High"] - df["Low"]) / df["Close"]
        df["Ticker"]      = ticker

        return df
    @classmethod
    def validate(cls, df: pd.DataFrame) -> dict:
        """Return a health report dict — useful for dashboard display."""
        return {
            "rows":            len(df),
            "start":           str(df.index[0].date()),
            "end":             str(df.index[-1].date()),
            "missing_pct":     round(df[cls.REQUIRED].isna().mean().mean() * 100, 2),
            "zero_vol_rows":   int((df["Volume"] == 0).sum()),
            "duplicate_dates": int(df.index.duplicated().sum()),
            "price_range":     f"{df['Close'].min():.2f} – {df['Close'].max():.2f}",
        }

# SOURCE 1 yfinance  (primary equities / crypto / forex / indices)


class YFinanceLoader:
    """
    Fetches OHLCV via yfinance.
    Supports stocks, ETFs, crypto (BTC-USD), forex (EURUSD=X),
    indices (^NSEI, ^GSPC, ^VIX) and futures (GC=F, CL=F).
    """

    @staticmethod
    def fetch(
        ticker:   str,
        start:    str  = CONFIG.default_start,
        end:      str  = CONFIG.default_end,
        interval: str  = CONFIG.default_interval,
        use_cache: bool = CONFIG.use_cache,
    ) -> pd.DataFrame:

        if not _YF_AVAILABLE:
            raise ImportError("yfinance not installed — pip install yfinance")

        cache_key = f"yf_{ticker}_{start}_{end}_{interval}"
        if use_cache:
            cached = _cache_load(cache_key)
            if cached is not None:
                print(f"[YFinance] Loaded {ticker} from cache")
                return cached
        print(f"[YFinance] Fetching {ticker}  {start} → {end}  ({interval})")
        raw = yf.download(
            ticker, start=start, end=end,
            interval=interval, auto_adjust=True,
            progress=False, threads=False,
        )

        if raw.empty:
            raise ValueError(f"[YFinance] No data returned for {ticker}")

        # yfinance returns MultiIndex columns when downloading single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = OHLCVCleaner.clean(raw, ticker)

        if use_cache:
            _cache_save(df, cache_key)

        return df

    @staticmethod
    def fetch_multiple(
        tickers:  list,
        start:    str  = CONFIG.default_start,
        end:      str  = CONFIG.default_end,
        interval: str  = CONFIG.default_interval,
    ) -> dict:
        """Returns {ticker: DataFrame}."""
        result = {}
        for t in tickers:
            try:
                result[t] = YFinanceLoader.fetch(t, start, end, interval)
                time.sleep(CONFIG.request_delay)
            except Exception as e:
                print(f"[YFinance] Failed {t}: {e}")
        return result

    @staticmethod
    def fetch_info(ticker: str) -> dict:
        """Returns company metadata dict."""
        if not _YF_AVAILABLE:
            return {}
        try:
            return yf.Ticker(ticker).info
        except Exception:
            return {}

# SOURCE 2 — FRED  (macro: rates, CPI, GDP, unemployment, VIX)


# Curated series most useful for regime detection + macro features
FRED_SERIES = {
    #interest rates 
    "DFF":     "Fed Funds Rate",
    "DGS2":    "2Y Treasury Yield",
    "DGS10":   "10Y Treasury Yield",
    "DGS30":   "30Y Treasury Yield",
    "T10Y2Y":  "Yield Curve (10Y-2Y)",
    "T10YFF":  "10Y Treasury - Fed Funds Spread",

    # inflation 
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "CPILFESL": "Core CPI (ex Food & Energy)",
    "PCEPI":    "PCE Price Index",
    "T5YIE":    "5Y Breakeven Inflation",
    "T10YIE":   "10Y Breakeven Inflation",

    # growth / activity 
    "GDP":      "Nominal GDP",
    "GDPC1":    "Real GDP",
    "INDPRO":   "Industrial Production",
    "RETAILSMNSA": "Retail Sales",
    "HOUST":    "Housing Starts",

    #labour 
    "UNRATE":   "Unemployment Rate",
    "PAYEMS":   "Nonfarm Payrolls",
    "ICSA":     "Initial Jobless Claims",

    #credit / liquidity 
    "BAMLH0A0HYM2": "HY Credit Spread (OAS)",
    "BAMLC0A0CM":   "IG Credit Spread (OAS)",
    "M2SL":         "M2 Money Supply",
    "DPSACBW027SBOG": "Bank Deposits",

    #sentiment / fear 
    "VIXCLS":   "CBOE VIX",
    "UMCSENT":  "U Michigan Consumer Sentiment",
}

class FREDLoader:
    """
    Fetches macro time-series from FRED.
    Returns a single merged DataFrame indexed by date.
    """

    def __init__(self, api_key: str = CONFIG.fred_api_key):
        if not _FRED_AVAILABLE:
            raise ImportError("fredapi not installed — pip install fredapi")
        if not api_key:
            raise ValueError(
                "FRED API key missing. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "Then set: export FRED_API_KEY='your_key_here'"
            )
        self.fred = Fred(api_key=api_key)
    def fetch_series(
        self,
        series_id:  str,
        start:      str = "2000-01-01",
        end:        str = CONFIG.default_end,
        use_cache:  bool = CONFIG.use_cache,
    ) -> pd.Series:
        cache_key = f"fred_{series_id}_{start}_{end}"
        if use_cache:
            p = _cache_path(cache_key)
            if os.path.exists(p):
                try:
                    return pd.read_parquet(p).squeeze()
                except Exception:
                    pass

        time.sleep(CONFIG.request_delay)
        s = self.fred.get_series(series_id, observation_start=start, observation_end=end)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = series_id

        if use_cache:
            s.to_frame().to_parquet(_cache_path(cache_key))

        return s

    def fetch_macro_panel(
        self,
        series_ids: list = None,
        start:      str  = "2000-01-01",
        end:        str  = CONFIG.default_end,
        resample:   str  = "D",        # 'D' daily, 'W' weekly, 'M' monthly
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series → merged + resampled DataFrame.
        Ready to merge with OHLCV df.
        """
        ids = series_ids or list(FRED_SERIES.keys())
        frames = []

        for sid in ids:
            try:
                s = self.fetch_series(sid, start, end)
                frames.append(s)
                print(f"  [FRED] {sid:20s} {FRED_SERIES.get(sid, '')}")
            except Exception as e:
                print(f"  [FRED] SKIP {sid}: {e}")

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, axis=1)
        panel = panel.resample(resample).last()
        panel.ffill(inplace=True)
        panel.dropna(how="all", inplace=True)
        return panel

    def fetch_yield_curve(
        self,
        start: str = "2000-01-01",
        end:   str = CONFIG.default_end,
    ) -> pd.DataFrame:
        """
        Returns daily yield curve with derived slope + curvature features.
        Usefuul for regime detection.
        """
        tenors = {
            "DGS1MO": "Y0_08", "DGS3MO": "Y0_25", "DGS6MO": "Y0_5",
            "DGS1": "Y1", "DGS2": "Y2", "DGS3": "Y3",
            "DGS5": "Y5", "DGS7": "Y7", "DGS10": "Y10",
            "DGS20": "Y20", "DGS30": "Y30",
        }
        frames = {}
        for fid, col in tenors.items():
            try:
                frames[col] = self.fetch_series(fid, start, end)
            except Exception:
                pass
        df = pd.DataFrame(frames).resample("D").last().ffill()

        # derived features
        if "Y2" in df and "Y10" in df:
            df["Curve_2_10"]  = df["Y10"] - df["Y2"]
        if "Y3MO" in df.columns or "Y0_25" in df:
            short = df.get("Y0_25", df.get("Y1", df.get("Y2")))
            long  = df.get("Y30", df.get("Y10"))
            if short is not None and long is not None:
                df["Curve_3M_30Y"] = long - short
        if "Y2" in df and "Y5" in df and "Y10" in df:
            df["Curvature"]   = 2 * df["Y5"] - df["Y2"] - df["Y10"]
        if "Y10" in df:
            df["Y10_MOM_3M"]  = df["Y10"].diff(63)
            df["Y10_MOM_1M"]  = df["Y10"].diff(21)

        return df

# SOURCE 3 — Alpha Vantage  (backup + intraday + forex + crypto)

class AlphaVantageLoader:
    """
    Alpha Vantage REST API wrapper.
    Free tier: 25 requests/day.  Premium: 75–1200 requests/min.
    """

    BASE = "https://www.alphavantage.co/query"
    def __init__(self, api_key: str = CONFIG.alpha_vantage_key):
        if not api_key:
            raise ValueError(
                "Alpha Vantage API key missing. "
                "Get one free at https://www.alphavantage.co/support/#api-key"
            )
        self.key = api_key

    def _get(self, params: dict) -> dict:
        params["apikey"] = self.key
        r = requests.get(self.BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "Error Message" in data:
            raise ValueError(data["Error Message"])
        if "Note" in data:
            print(f"[AlphaVantage] Rate limit note: {data['Note']}")
        return data

    def fetch_daily(
        self,
        ticker:    str,
        outputsize: str = "full",       # "compact" = 100 rows, "full" = 20yr
        adjusted:  bool = True,
        use_cache: bool = CONFIG.use_cache,
    ) -> pd.DataFrame:

        cache_key = f"av_daily_{ticker}_{outputsize}_{adjusted}"
        if use_cache:
            cached = _cache_load(cache_key, max_age_hours=24)
            if cached is not None:
                return cached

        func = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        data = self._get({"function": func, "symbol": ticker, "outputsize": outputsize})

        key = [k for k in data if "Time Series" in k][0]
        raw = pd.DataFrame(data[key]).T
        raw.index = pd.to_datetime(raw.index)

        rename = {
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "5. adjusted close": "Close", "4. close": "Close",
            "6. volume": "Volume",
        }
        raw.rename(columns=rename, inplace=True)

        df = OHLCVCleaner.clean(raw, ticker)
        if use_cache:
            _cache_save(df, cache_key)
        return df

    def fetch_intraday(
        self,
        ticker:   str,
        interval: str = "5min",       # 1min 5min 15min 30min 60min
        month:    str = None,         # "2024-01" for historical premium
        use_cache: bool = CONFIG.use_cache,
    ) -> pd.DataFrame:

        cache_key = f"av_intra_{ticker}_{interval}_{month or 'live'}"
        if use_cache:
            cached = _cache_load(cache_key, max_age_hours=1)
            if cached is not None:
                return cached

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": interval,
            "outputsize": "full",
        }
        if month:
            params["month"] = month

        data = self._get(params)
        key = [k for k in data if "Time Series" in k][0]
        raw = pd.DataFrame(data[key]).T
        raw.index = pd.to_datetime(raw.index)

        rename = {
            "1. open": "Open", "2. high": "High",
            "3. low": "Low", "4. close": "Close", "5. volume": "Volume",
        }
        raw.rename(columns=rename, inplace=True)
        df = OHLCVCleaner.clean(raw, ticker)
        if use_cache:
            _cache_save(df, cache_key)
        return df

    def fetch_forex(self, from_sym: str, to_sym: str) -> pd.DataFrame:
        data = self._get({
            "function": "FX_DAILY",
            "from_symbol": from_sym,
            "to_symbol": to_sym,
            "outputsize": "full",
        })
        raw = pd.DataFrame(data["Time Series FX (Daily)"]).T
        raw.index = pd.to_datetime(raw.index)
        raw.rename(columns={
            "1. open": "Open", "2. high": "High",
            "3. low": "Low", "4. close": "Close",
        }, inplace=True)
        raw["Volume"] = 0
        return OHLCVCleaner.clean(raw, f"{from_sym}/{to_sym}")

    def fetch_crypto(self, symbol: str, market: str = "USD") -> pd.DataFrame:
        data = self._get({
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
        })
        raw = pd.DataFrame(data["Time Series (Digital Currency Daily)"]).T
        raw.index = pd.to_datetime(raw.index)
        raw.rename(columns={
            f"1a. open ({market})":   "Open",
            f"2a. high ({market})":   "High",
            f"3a. low ({market})":    "Low",
            f"4a. close ({market})":  "Close",
            "5. volume":              "Volume",
        }, inplace=True)
        return OHLCVCleaner.clean(raw, f"{symbol}/{market}")

    def fetch_earnings_sentiment(self, ticker: str) -> pd.DataFrame:
        """
        Returns news sentiment scored per article.
        Useful for LLM integration layer.
        """
        data = self._get({
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": 200,
        })
        rows = []
        for item in data.get("feed", []):
            ts = pd.to_datetime(item["time_published"], format="%Y%m%dT%H%M%S")
            for ts_info in item.get("ticker_sentiment", []):
                if ts_info["ticker"] == ticker:
                    rows.append({
                        "date":             ts,
                        "title":            item["title"],
                        "source":           item["source"],
                        "overall_score":    float(item.get("overall_sentiment_score", 0)),
                        "ticker_score":     float(ts_info.get("ticker_sentiment_score", 0)),
                        "relevance":        float(ts_info.get("relevance_score", 0)),
                        "label":            ts_info.get("ticker_sentiment_label", ""),
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
        return df

# SOURCE 4 — Polymarket  (prediction market odds — hackathon track 1)

class PolymarketLoader:
    """
    Fetches live market probabilities from Polymarket (public CLOB API).
    No API key required for read access.
    """

    BASE = "https://clob.polymarket.com"

    def get_markets(self, limit: int = 50, active_only: bool = True) -> pd.DataFrame:
        params = {"limit": limit}
        if active_only:
            params["active"] = "true"
        try:
            r = requests.get(f"{self.BASE}/markets", params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("data", [])
            rows = []
            for m in data:
                rows.append({
                    "market_id":    m.get("condition_id", ""),
                    "question":     m.get("question", ""),
                    "end_date":     m.get("end_date_iso", ""),
                    "volume_usd":   float(m.get("volume", 0)),
                    "liquidity":    float(m.get("liquidity", 0)),
                    "yes_price":    None,
                    "no_price":     None,
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[Polymarket] Error: {e}")
            return pd.DataFrame()

    def get_orderbook(self, token_id: str) -> dict:
        """Returns best bid/ask for a binary outcome token."""
        try:
            r = requests.get(
                f"{self.BASE}/book",
                params={"token_id": token_id},
                timeout=10
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[Polymarket] Orderbook error: {e}")
            return {}

    def get_price_history(self, market_id: str, interval: str = "1d") -> pd.DataFrame:
        """Returns price history for a market outcome (YES token)."""
        try:
            r = requests.get(
                f"{self.BASE}/prices-history",
                params={"market": market_id, "interval": interval, "fidelity": 60},
                timeout=10
            )
            r.raise_for_status()
            history = r.json().get("history", [])
            df = pd.DataFrame(history)
            if not df.empty:
                df["date"] = pd.to_datetime(df["t"], unit="s")
                df.rename(columns={"p": "yes_prob"}, inplace=True)
                df.set_index("date", inplace=True)
                df = df[["yes_prob"]]
            return df
        except Exception as e:
            print(f"[Polymarket] History error: {e}")
            return pd.DataFrame()

# SOURCE 5 — Kalshi  (regulated US prediction market)

class KalshiLoader:
    """
    Fetches market data from Kalshi (regulated CFTC exchange).
    Public endpoints need no auth.  Trading requires OAuth.
    """

    BASE = "https://trading-api.kalshi.com/trade-api/v2"

    def get_markets(
        self,
        limit:    int = 100,
        status:   str = "open",        # open, closed, settled
        category: str = None,          # "economics", "politics", "financials"
    ) -> pd.DataFrame:
        params = {"limit": limit, "status": status}
        if category:
            params["series_ticker"] = category
        try:
            r = requests.get(f"{self.BASE}/markets", params=params, timeout=10)
            r.raise_for_status()
            markets = r.json().get("markets", [])
            rows = []
            for m in markets:
                rows.append({
                    "ticker":       m.get("ticker", ""),
                    "title":        m.get("title", ""),
                    "category":     m.get("category", ""),
                    "close_time":   m.get("close_time", ""),
                    "yes_bid":      m.get("yes_bid", 0) / 100,
                    "yes_ask":      m.get("yes_ask", 0) / 100,
                    "no_bid":       m.get("no_bid", 0) / 100,
                    "no_ask":       m.get("no_ask", 0) / 100,
                    "implied_prob": (m.get("yes_bid", 0) + m.get("yes_ask", 0)) / 200,
                    "volume":       m.get("volume", 0),
                    "open_interest": m.get("open_interest", 0),
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[Kalshi] Error: {e}")
            return pd.DataFrame()

    def get_market_history(self, ticker: str) -> pd.DataFrame:
        try:
            r = requests.get(
                f"{self.BASE}/markets/{ticker}/history",
                timeout=10
            )
            r.raise_for_status()
            history = r.json().get("history", [])
            df = pd.DataFrame(history)
            if not df.empty:
                df["date"] = pd.to_datetime(df["ts"])
                df.set_index("date", inplace=True)
                for col in ["yes_bid", "yes_ask", "no_bid", "no_ask"]:
                    if col in df:
                        df[col] = df[col] / 100
                df["implied_prob"] = (df.get("yes_bid", 0) + df.get("yes_ask", 0)) / 2
            return df
        except Exception as e:
            print(f"[Kalshi] History error: {e}")
            return pd.DataFrame()

# MACRO FEATURE BUILDER  — merges price + macro into one df

class MacroFeatureBuilder:
    """
    Merges OHLCV price data with FRED macro series.
    Adds derived macro features useful for regime detection.
    Output df is ready for indicators.py → add_all_indicators()
    """
    @staticmethod
    def merge(
        price_df:  pd.DataFrame,
        macro_df:  pd.DataFrame,
        method:    str = "ffill",       # how to align lower-freq macro to daily
    ) -> pd.DataFrame:
        """
        Left-join macro onto price by date.
        macro_df dates that don't align to trading days are forward-filled.
        """
        if macro_df.empty:
            return price_df

        aligned = macro_df.reindex(price_df.index, method=method)
        return pd.concat([price_df, aligned], axis=1)

    @staticmethod
    def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds derived macro features assuming standard FRED columns are present.
        Call AFTER merge().
        """
        # Yield curve slope (2-10)
        if "DGS10" in df and "DGS2" in df:
            df["Yield_Curve"]     = df["DGS10"] - df["DGS2"]
            df["Curve_Inverted"]  = (df["Yield_Curve"] < 0).astype(int)
            df["Curve_MOM_1M"]    = df["Yield_Curve"].diff(21)

        # ── Real rate ─────────────────────────────────────────────────
        if "DGS10" in df and "T10YIE" in df:
            df["Real_Rate_10Y"]   = df["DGS10"] - df["T10YIE"]

        # ── Credit stress ─────────────────────────────────────────────
        if "BAMLH0A0HYM2" in df:
            df["HY_Spread_MOM"]   = df["BAMLH0A0HYM2"].diff(21)
            df["HY_Spread_Z"]     = (
                (df["BAMLH0A0HYM2"] - df["BAMLH0A0HYM2"].rolling(252).mean())
                / df["BAMLH0A0HYM2"].rolling(252).std()
            )

        # ── Inflation momentum ────────────────────────────────────────
        if "CPIAUCSL" in df:
            df["CPI_YOY"]         = df["CPIAUCSL"].pct_change(252) * 100
            df["CPI_MOM"]         = df["CPIAUCSL"].pct_change(21) * 100

        # ── VIX regime ────────────────────────────────────────────────
        if "VIXCLS" in df:
            df["VIX_MA30"]        = df["VIXCLS"].rolling(30).mean()
            df["VIX_Regime"]      = pd.cut(
                df["VIXCLS"],
                bins=[0, 15, 20, 30, 40, 999],
                labels=["calm", "normal", "elevated", "fear", "panic"]
            )

        return df

# MASTER DataPipeline  — one call to rule them all
