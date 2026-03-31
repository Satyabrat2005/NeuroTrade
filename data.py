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
