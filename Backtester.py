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


# POSITION SIZER

class PositionSizer:

    @staticmethod
    def fixed_fractional(capital: float, price: float, pct: float) -> float:
        return (capital * pct) / price

    @staticmethod
    def atr_based(capital: float, price: float, atr: float,
                  risk_pct: float = 0.01) -> float:
        """Risk a fixed % of capital per 1 ATR move."""
        if atr <= 0:
            return 0
        risk_amount = capital * risk_pct
        return risk_amount / atr

    @staticmethod
    def volatility_scaled(capital: float, price: float,
                          returns: pd.Series, target_vol: float = 0.15,
                          ann: int = 252) -> float:
        """Scale position so portfolio vol matches target_vol."""
        realized_vol = returns.std() * np.sqrt(ann)
        if realized_vol == 0:
            return 0
        scale = target_vol / realized_vol
        return (capital * scale) / price

# CORE BACKTESTING ENGINE

class Backtester:

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        self.capital = self.config.initial_capital
        self.equity_curve = []
        self.trades: list[Trade] = []
        self.position = Position()
        self.daily_returns = []
        self.regimes = []
        self._aborted = False

    # ----------------------------------------------------------
    # SIGNAL EXECUTION
    # ----------------------------------------------------------

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        slip = price * self.config.slippage_pct
        return price + slip if side == OrderSide.BUY else price - slip

    def _apply_commission(self, trade_value: float) -> float:
        return trade_value * self.config.commission_pct
    def _open_position(self, date, price, side: PositionSide,
                       atr: float = None, returns_hist: pd.Series = None):
        exec_price = self._apply_slippage(
            price, OrderSide.BUY if side == PositionSide.LONG else OrderSide.SELL
        )

        cfg = self.config
        if cfg.use_atr_sizing and atr is not None and atr > 0:
            size = PositionSizer.atr_based(self.capital, exec_price, atr, cfg.atr_risk_pct)
        else:
            size = PositionSizer.fixed_fractional(self.capital, exec_price, cfg.position_size_pct)

        trade_value = size * exec_price
        commission = self._apply_commission(trade_value)
        self.capital -= commission

        self.position = Position(
            side=side,
            size=size,
            entry_price=exec_price,
            entry_date=date,
            stop_loss=exec_price * (1 - cfg.stop_loss_pct) if cfg.stop_loss_pct and side == PositionSide.LONG
                      else exec_price * (1 + cfg.stop_loss_pct) if cfg.stop_loss_pct else None,
            take_profit=exec_price * (1 + cfg.take_profit_pct) if cfg.take_profit_pct and side == PositionSide.LONG
                        else exec_price * (1 - cfg.take_profit_pct) if cfg.take_profit_pct else None,
            trailing_stop_pct=cfg.trailing_stop_pct,
        )
        self.position._commission_paid = commission

    def _close_position(self, date, price, reason: str = "signal", regime: str = ""):
        if self.position.side == PositionSide.FLAT:
            return None

        side = self.position.side
        exec_price = self._apply_slippage(
            price, OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
        )

        trade_value = self.position.size * exec_price
        commission = self._apply_commission(trade_value)
        commission += getattr(self.position, '_commission_paid', 0)

        if side == PositionSide.LONG:
            pnl = (exec_price - self.position.entry_price) * self.position.size - commission
            pnl_pct = (exec_price - self.position.entry_price) / self.position.entry_price
        else:
            pnl = (self.position.entry_price - exec_price) * self.position.size - commission
            pnl_pct = (self.position.entry_price - exec_price) / self.position.entry_price

        self.capital += self.position.size * self.position.entry_price + pnl

        trade = Trade(
            entry_date=self.position.entry_date,
            exit_date=date,
            side=side,
            entry_price=self.position.entry_price,
            exit_price=exec_price,
            size=self.position.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            mae=self.position.mae,
            mfe=self.position.mfe,
            duration_bars=self.position.bars_held,
            exit_reason=reason,
            regime=regime,
            commission=commission,
        )

        self.trades.append(trade)
        self.position = Position()
        return trade

    def _update_position(self, high: float, low: float, close: float):
        """Update trailing stops, MAE/MFE, unrealized PnL."""
        pos = self.position
        if pos.side == PositionSide.FLAT:
            return None

        pos.bars_held += 1

        if pos.side == PositionSide.LONG:
            pos.unrealized_pnl = (close - pos.entry_price) * pos.size
            excursion_low = (low - pos.entry_price) / pos.entry_price
            excursion_high = (high - pos.entry_price) / pos.entry_price
            pos.mae = min(pos.mae, excursion_low)
            pos.mfe = max(pos.mfe, excursion_high)

            if pos.trailing_stop_pct:
                new_ts = high * (1 - pos.trailing_stop_pct)
                pos.trailing_stop = max(pos.trailing_stop or 0, new_ts)
                if low <= pos.trailing_stop:
                    return "trailing_stop"

            if pos.stop_loss and low <= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit and high >= pos.take_profit:
                return "take_profit"

        else:  # SHORT
            pos.unrealized_pnl = (pos.entry_price - close) * pos.size
            excursion_high = (pos.entry_price - high) / pos.entry_price
            excursion_low = (pos.entry_price - low) / pos.entry_price
            pos.mae = min(pos.mae, excursion_high)
            pos.mfe = max(pos.mfe, excursion_low)
            if pos.trailing_stop_pct:
                new_ts = low * (1 + pos.trailing_stop_pct)
                pos.trailing_stop = min(pos.trailing_stop or float('inf'), new_ts)
                if high >= pos.trailing_stop:
                    return "trailing_stop"

            if pos.stop_loss and high >= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit and low <= pos.take_profit:
                return "take_profit"

        return None
        
    # MAIN RUN
    def run(self, df: pd.DataFrame, signal_func: Callable,
            signal_kwargs: dict = None) -> dict:
        """
        Run backtest.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with indicators pre-computed.
        signal_func : Callable
            Function(df, i, **kwargs) -> PositionSide or None
            Return PositionSide.LONG / SHORT / FLAT or None.
        signal_kwargs : dict
            Extra args passed to signal_func.

        Returns
        -------
        dict : Full results dictionary.
        """
        self.reset()
        signal_kwargs = signal_kwargs or {}
        regimes = RegimeDetector.detect(df)

        prev_equity = self.config.initial_capital

        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]
            o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
            atr = row.get('ATR', None) if hasattr(row, 'get') else None
            regime = regimes.iloc[i]
            # Update open position
            exit_reason = self._update_position(h, l, c)
            if exit_reason and self.position.side != PositionSide.FLAT:
                self._close_position(date, c, reason=exit_reason, regime=regime)

            #  Get signal
            signal = signal_func(df, i, **signal_kwargs)

            # --- Execute signal ---
            if signal is not None:
                current_side = self.position.side

                # Flip or open
                if signal == PositionSide.FLAT and current_side != PositionSide.FLAT:
                    self._close_position(date, o, reason="signal", regime=regime)

                elif signal == PositionSide.LONG and current_side != PositionSide.LONG:
                    if current_side == PositionSide.SHORT:
                        self._close_position(date, o, reason="signal_flip", regime=regime)
                    self._open_position(date, o, PositionSide.LONG, atr=atr)

                elif signal == PositionSide.SHORT and self.config.allow_shorting and current_side != PositionSide.SHORT:
                    if current_side == PositionSide.LONG:
                        self._close_position(date, o, reason="signal_flip", regime=regime)
                    self._open_position(date, o, PositionSide.SHORT, atr=atr)

            # Equity snapshot
            portfolio_value = self.capital
            if self.position.side != PositionSide.FLAT:
                portfolio_value += self.position.unrealized_pnl
