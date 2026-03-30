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
