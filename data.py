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
