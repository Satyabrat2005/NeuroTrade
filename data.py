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
