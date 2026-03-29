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

    @staticmethod
    def max_drawdown_duration(equity: pd.Series) -> int:
        dd = RiskAnalytics.drawdown_series(equity)
        in_dd = dd < 0
        durations = []
        count = 0
        for val in in_dd:
            if val:
                count += 1
            else:
                if count > 0:
                    durations.append(count)
                count = 0
        return max(durations) if durations else 0

    @staticmethod
    def var(returns: pd.Series, confidence: float = 0.95) -> float:
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        var = RiskAnalytics.var(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def ulcer_index(equity: pd.Series) -> float:
        dd = RiskAnalytics.drawdown_series(equity)
        return np.sqrt((dd ** 2).mean())

    @staticmethod
    def tail_ratio(returns: pd.Series) -> float:
        p95 = abs(np.percentile(returns, 95))
        p5 = abs(np.percentile(returns, 5))
        return p95 / p5 if p5 != 0 else np.inf

    @staticmethod
    def win_rate(trades: list) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)

    @staticmethod
    def profit_factor(trades: list) -> float:
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else np.inf

    @staticmethod
    def expectancy(trades: list) -> float:
        if not trades:
            return 0.0
        wr = RiskAnalytics.win_rate(trades)
        avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if any(t.pnl > 0 for t in trades) else 0
        avg_loss = abs(np.mean([t.pnl for t in trades if t.pnl < 0])) if any(t.pnl < 0 for t in trades) else 0
        return (wr * avg_win) - ((1 - wr) * avg_loss)

    @staticmethod
    def kelly_criterion(trades: list) -> float:
        wr = RiskAnalytics.win_rate(trades)
        avg_win = np.mean([t.pnl_pct for t in trades if t.pnl > 0]) if any(t.pnl > 0 for t in trades) else 0
        avg_loss = abs(np.mean([t.pnl_pct for t in trades if t.pnl < 0])) if any(t.pnl < 0 for t in trades) else 0
        if avg_loss == 0:
            return 0.0
        rr = avg_win / avg_loss
        return wr - (1 - wr) / rr

    @staticmethod
    def consecutive_stats(trades: list):
        wins, losses = [], []
        cur_w, cur_l = 0, 0
        for t in trades:
            if t.pnl > 0:
                cur_w += 1
                if cur_l > 0:
                    losses.append(cur_l)
                cur_l = 0
            else:
                cur_l += 1
                if cur_w > 0:
                    wins.append(cur_w)
                cur_w = 0
        if cur_w > 0: wins.append(cur_w)
        if cur_l > 0: losses.append(cur_l)
        return {
            "max_consecutive_wins": max(wins) if wins else 0,
            "max_consecutive_losses": max(losses) if losses else 0,
            "avg_consecutive_wins": np.mean(wins) if wins else 0,
            "avg_consecutive_losses": np.mean(losses) if losses else 0,
        }

# REGIME DETECTION ENGINE


class RegimeDetector:

    @staticmethod
    def detect(df: pd.DataFrame) -> pd.Series:
        """
        Classify market regime at each bar using:
        - 50/200 SMA trend direction
        - ATR-based volatility classification
        """
        regimes = []
        closes = df['Close'].values
        n = len(closes)
        sma50 = pd.Series(closes).rolling(50).mean().values
        sma200 = pd.Series(closes).rolling(200).mean().values

        high_low = df['High'] - df['Low']
        atr = high_low.rolling(14).mean().values
        atr_ma = pd.Series(atr).rolling(50).mean().values

        for i in range(n):
            if np.isnan(sma200[i]) or np.isnan(atr_ma[i]):
                regimes.append(MarketRegime.SIDEWAYS.value)
                continue
            trending_up = sma50[i] > sma200[i]
            high_vol = atr[i] > atr_ma[i] * 1.2

            if high_vol:
                regimes.append(MarketRegime.HIGH_VOL.value)
            elif trending_up:
                regimes.append(MarketRegime.BULL.value)
            elif not trending_up:
                regimes.append(MarketRegime.BEAR.value)
            else:
                regimes.append(MarketRegime.SIDEWAYS.value)

        return pd.Series(regimes, index=df.index)
