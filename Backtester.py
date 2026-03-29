import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

# ENUMS & DATA STRUCTURES

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    side: PositionSide
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    mae: float = 0.0        # Maximum Adverse Excursion
    mfe: float = 0.0        
    duration_bars: int = 0
    exit_reason: str = ""
    regime: str = ""
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    side: PositionSide = PositionSide.FLAT
    size: float = 0.0
    entry_price: float = 0.0
    entry_date: Optional[pd.Timestamp] = None
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001          # 0.1% per trade
    slippage_pct: float = 0.0005          # 0.05% slippage
    position_size_pct: float = 0.95       # % of capital per trade
    max_positions: int = 1
    allow_shorting: bool = True
    use_atr_sizing: bool = False
    atr_risk_pct: float = 0.01            # Risk 1% of capital per ATR
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_drawdown_abort: float = 0.30      # Abort if DD exceeds 30%
    risk_free_rate: float = 0.06          # 6% annual (India benchmark)
    annualization_factor: int = 252

# RISK & PERFORMANCE ANALYTICS ENGINE

class RiskAnalytics:
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, rf: float = 0.06, ann: int = 252) -> float:
        excess = returns - rf / ann
        if returns.std() == 0:
            return 0.0
        return np.sqrt(ann) * excess.mean() / returns.std()

    @staticmethod
    def sortino_ratio(returns: pd.Series, rf: float = 0.06, ann: int = 252) -> float:
        excess = returns - rf / ann
        downside = returns[returns < 0].std()
        if downside == 0:
            return 0.0
        return np.sqrt(ann) * excess.mean() / downside

    @staticmethod
    def calmar_ratio(returns: pd.Series, ann: int = 252) -> float:
        annual_return = returns.mean() * ann
        mdd = RiskAnalytics.max_drawdown(returns)
        if mdd == 0:
            return 0.0
        return annual_return / abs(mdd)

    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        return gains / losses if losses != 0 else np.inf

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        cum = (1 + returns).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        return dd.min()

    @staticmethod
    def drawdown_series(equity: pd.Series) -> pd.Series:
        roll_max = equity.cummax()
        return (equity - roll_max) / roll_max
