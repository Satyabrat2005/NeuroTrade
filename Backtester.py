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

            self.equity_curve.append({"date": date, "equity": portfolio_value, "regime": regime})

            ret = (portfolio_value - prev_equity) / prev_equity if prev_equity != 0 else 0
            self.daily_returns.append(ret)
            prev_equity = portfolio_value

            # Drawdown abort 
            peak = max(e['equity'] for e in self.equity_curve)
            if (portfolio_value - peak) / peak < -self.config.max_drawdown_abort:
                self._aborted = True
                break

        # Close any open position at end
        if self.position.side != PositionSide.FLAT:
            last = df.iloc[-1]
            self._close_position(df.index[-1], last['Close'], reason="end_of_data")

        return self._compile_results(df)

    # RESULTS COMPILATION

    def _compile_results(self, df: pd.DataFrame) -> dict:
        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        equity_series = equity_df['equity']
        returns = pd.Series(self.daily_returns)

        ra = RiskAnalytics
        cfg = self.config
        trades = self.trades

        # Core metrics
        total_return = (equity_series.iloc[-1] - cfg.initial_capital) / cfg.initial_capital
        n_days = len(equity_series)
        ann_return = (1 + total_return) ** (cfg.annualization_factor / n_days) - 1

        # Trade stats
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        long_trades = [t for t in trades if t.side == PositionSide.LONG]
        short_trades = [t for t in trades if t.side == PositionSide.SHORT]
        consec = ra.consecutive_stats(trades)

        # Regime breakdown
        regime_pnl = {}
        for t in trades:
            regime_pnl.setdefault(t.regime, []).append(t.pnl)
        regime_breakdown = {r: {
            "total_pnl": sum(pnls),
            "n_trades": len(pnls),
            "avg_pnl": np.mean(pnls)
        } for r, pnls in regime_pnl.items()}

        # MAE / MFE analysis
        avg_mae = np.mean([t.mae for t in trades]) if trades else 0
        avg_mfe = np.mean([t.mfe for t in trades]) if trades else 0

        results = {
            # Portfolio Performance 
            "initial_capital": cfg.initial_capital,
            "final_capital": equity_series.iloc[-1],
            "total_return_pct": total_return * 100,
            "annualized_return_pct": ann_return * 100,
            "aborted_early": self._aborted,

            #Risk-Adjusted Returns
            "sharpe_ratio": ra.sharpe_ratio(returns, cfg.risk_free_rate, cfg.annualization_factor),
            "sortino_ratio": ra.sortino_ratio(returns, cfg.risk_free_rate, cfg.annualization_factor),
            "calmar_ratio": ra.calmar_ratio(returns, cfg.annualization_factor),
            "omega_ratio": ra.omega_ratio(returns),

            # Drawdown
            "max_drawdown_pct": ra.max_drawdown(returns) * 100,
            "max_drawdown_duration_bars": ra.max_drawdown_duration(equity_series),
            "ulcer_index": ra.ulcer_index(equity_series),

            # Tail Risk
            "var_95_pct": ra.var(returns, 0.95) * 100,
            "cvar_95_pct": ra.cvar(returns, 0.95) * 100,
            "tail_ratio": ra.tail_ratio(returns),
            "daily_vol_annualized_pct": returns.std() * np.sqrt(cfg.annualization_factor) * 100,

            #Trade Statistics
            "total_trades": len(trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": ra.win_rate(trades) * 100,
            "profit_factor": ra.profit_factor(trades),
            "expectancy_per_trade": ra.expectancy(trades),
            "kelly_criterion_pct": ra.kelly_criterion(trades) * 100,

            # Trade PnL
            "gross_profit": sum(t.pnl for t in winning_trades),
            "gross_loss": sum(t.pnl for t in losing_trades),
            "avg_win": np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            "avg_loss": np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            "largest_win": max((t.pnl for t in trades), default=0),
            "largest_loss": min((t.pnl for t in trades), default=0),
            "avg_trade_duration_bars": np.mean([t.duration_bars for t in trades]) if trades else 0,

            # MAE / MFE
            "avg_mae_pct": avg_mae * 100,
            "avg_mfe_pct": avg_mfe * 100,

            # Consecutive
            **consec,

            # Exit Reasons
            "exit_reasons": pd.Series([t.exit_reason for t in trades]).value_counts().to_dict(),

            # Regime Breakdown
            "regime_breakdown": regime_breakdown,

            #  Series (for plotting) 
            "equity_curve": equity_df,
            "trades": trades,
            "returns": returns,
        }

        return results
# WALK-FORWARD OPTIMIZER

class WalkForwardOptimizer:
    """
    Expanding / rolling window walk-forward analysis.
    Prevents overfitting by testing on out-of-sample windows.
    """

    def __init__(self, backtester_config: BacktestConfig = None,
                 n_splits: int = 5, train_ratio: float = 0.7):
        self.config = backtester_config or BacktestConfig()
        self.n_splits = n_splits
        self.train_ratio = train_ratio

    def run(self, df: pd.DataFrame, signal_func: Callable,
            param_grid: dict, optimize_metric: str = "sharpe_ratio") -> dict:
        """
        Run walk-forward optimization.

        param_grid : dict of {param_name: [values]}
        """
        n = len(df)
        window = n // self.n_splits
        oos_results = []
        best_params_per_fold = []

        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD OPTIMIZATION  |  {self.n_splits} folds")
        print(f"{'='*60}")

        for fold in range(self.n_splits):
            train_end = int(window * (fold + 1) * self.train_ratio) + fold * window
            test_start = train_end
            test_end = min(train_end + window, n)

            if test_start >= n:
                break

            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]

            if len(train_df) < 50 or len(test_df) < 10:
                continue
            # Grid search on train
            best_score = -np.inf
            best_params = None

            for params in self._param_combinations(param_grid):
                bt = Backtester(self.config)
                try:
                    res = bt.run(train_df, signal_func, params)
                    score = res.get(optimize_metric, -np.inf)
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_params = params
                except Exception:
                    continue

            if best_params is None:
                continue

            # Evaluate on test (OOS)
            bt_oos = Backtester(self.config)
            oos_res = bt_oos.run(test_df, signal_func, best_params)

            oos_results.append({
                "fold": fold + 1,
                "train_period": f"{train_df.index[0].date()} - {train_df.index[-1].date()}",
                "test_period": f"{test_df.index[0].date()} - {test_df.index[-1].date()}",
                "best_params": best_params,
                "train_score": best_score,
                "oos_score": oos_res.get(optimize_metric, np.nan),
                "oos_return_pct": oos_res.get("total_return_pct", np.nan),
                "oos_sharpe": oos_res.get("sharpe_ratio", np.nan),
                "oos_max_dd_pct": oos_res.get("max_drawdown_pct", np.nan),
            })
            best_params_per_fold.append(best_params)

            print(f"  Fold {fold+1}: Train Score={best_score:.3f} | "
                  f"OOS Score={oos_res.get(optimize_metric, 0):.3f} | "
                  f"Params={best_params}")
        summary_df = pd.DataFrame(oos_results)
        return {
            "fold_results": summary_df,
            "avg_oos_score": summary_df["oos_score"].mean() if not summary_df.empty else np.nan,
            "avg_oos_return_pct": summary_df["oos_return_pct"].mean() if not summary_df.empty else np.nan,
            "efficiency_ratio": (summary_df["oos_score"] / summary_df["train_score"]).mean() if not summary_df.empty else np.nan,
            "best_params_per_fold": best_params_per_fold,
        }

    @staticmethod
    def _param_combinations(param_grid: dict):
        import itertools
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            yield dict(zip(keys, values))

# MONTE CARLO SIMULATOR

class MonteCarloSimulator:
    """
    Bootstrap Monte Carlo simulation on trade sequence.
    Answers: "Is this edge statistically significant?"
    """
    def __init__(self, n_simulations: int = 10_000, confidence: float = 0.95):
        self.n_sims = n_simulations
        self.confidence = confidence

    def run(self, trades: list, initial_capital: float = 100_000) -> dict:
        if not trades:
            return {}

        pnls = np.array([t.pnl for t in trades])
        n = len(pnls)

        sim_finals = []
        sim_maxdds = []
        sim_sharpes = []

        for _ in range(self.n_sims):
            shuffled = np.random.choice(pnls, size=n, replace=True)
            equity = initial_capital + np.cumsum(shuffled)
            returns = np.diff(equity) / equity[:-1]

            sim_finals.append(equity[-1])
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            sim_maxdds.append(dd.min())
            if returns.std() > 0:
                sim_sharpes.append(returns.mean() / returns.std() * np.sqrt(252))

        lo = (1 - self.confidence) / 2
        hi = 1 - lo

        return {
            "n_simulations": self.n_sims,
            "confidence_level": self.confidence,
            "final_equity": {
                "mean": np.mean(sim_finals),
                "median": np.median(sim_finals),
                f"p{int(lo*100)}": np.percentile(sim_finals, lo * 100),
                f"p{int(hi*100)}": np.percentile(sim_finals, hi * 100),
                "prob_profit": np.mean(np.array(sim_finals) > initial_capital),
            },
            "max_drawdown_pct": {
                "mean": np.mean(sim_maxdds) * 100,
                "worst_5pct": np.percentile(sim_maxdds, 5) * 100,
            },
            "sharpe_ratio": {
                "mean": np.mean(sim_sharpes),
                "p5": np.percentile(sim_sharpes, 5),
                "p95": np.percentile(sim_sharpes, 95),
            },
            "raw_finals": np.array(sim_finals),
            "raw_maxdds": np.array(sim_maxdds),
        }

# BENCHMARK COMPARATOR

class BenchmarkComparator:
    """Compare strategy against buy-and-hold benchmark."""

    @staticmethod
    def run(strategy_equity: pd.Series, df: pd.DataFrame,
            initial_capital: float, rf: float = 0.06) -> dict:
        bh_returns = df['Close'].pct_change().dropna()
        bh_equity = initial_capital * (1 + bh_returns).cumprod()

        strat_returns = strategy_equity.pct_change().dropna()

        ra = RiskAnalytics

        # Information Ratio
        active_returns = strat_returns.values - bh_returns.reindex(strat_returns.index).fillna(0).values
        ir = (active_returns.mean() / active_returns.std() * np.sqrt(252)
              if active_returns.std() != 0 else 0)

        # Beta
        cov = np.cov(strat_returns, bh_returns.reindex(strat_returns.index).fillna(0))
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
        alpha = (strat_returns.mean() - rf / 252) - beta * (
            bh_returns.reindex(strat_returns.index).fillna(0).mean() - rf / 252
        )
        alpha_annualized = alpha * 252
        return {
            "strategy": {
                "total_return_pct": (strategy_equity.iloc[-1] / initial_capital - 1) * 100,
                "sharpe": ra.sharpe_ratio(strat_returns, rf),
                "max_drawdown_pct": ra.max_drawdown(strat_returns) * 100,
                "annualized_vol_pct": strat_returns.std() * np.sqrt(252) * 100,
            },
            "buy_and_hold": {
                "total_return_pct": (bh_equity.iloc[-1] / initial_capital - 1) * 100,
                "sharpe": ra.sharpe_ratio(bh_returns, rf),
                "max_drawdown_pct": ra.max_drawdown(bh_returns) * 100,
                "annualized_vol_pct": bh_returns.std() * np.sqrt(252) * 100,
            },
            "alpha_annualized_pct": alpha_annualized * 100,
            "beta": beta,
            "information_ratio": ir,
            "correlation_with_benchmark": np.corrcoef(
                strat_returns, bh_returns.reindex(strat_returns.index).fillna(0)
            )[0, 1],
        }

# REPORT PRINTER
class ReportPrinter:

    @staticmethod
    def print_full(results: dict, wf_results: dict = None,
                   mc_results: dict = None, bench: dict = None):

        SEP = "=" * 65
        sep = "-" * 65

        print(f"\n{SEP}")
        print(f"  BACKTEST RESULTS REPORT")
        print(SEP)

        print(f"\n{'PORTFOLIO PERFORMANCE':^65}")
        print(sep)
        print(f"  Initial Capital      : ${results['initial_capital']:>15,.2f}")
        print(f"  Final Capital        : ${results['final_capital']:>15,.2f}")
        print(f"  Total Return         : {results['total_return_pct']:>14.2f}%")
        print(f"  Annualized Return    : {results['annualized_return_pct']:>14.2f}%")
        if results.get("aborted_early"):
            print(f"  Strategy aborted early (max drawdown limit hit)")
        print(f"\n{'RISK-ADJUSTED METRICS':^65}")
        print(sep)
        print(f"  Sharpe Ratio         : {results['sharpe_ratio']:>15.4f}")
        print(f"  Sortino Ratio        : {results['sortino_ratio']:>15.4f}")
        print(f"  Calmar Ratio         : {results['calmar_ratio']:>15.4f}")
        print(f"  Omega Ratio          : {results['omega_ratio']:>15.4f}")

        print(f"\n{'DRAWDOWN & RISK':^65}")
        print(sep)
        print(f"  Max Drawdown         : {results['max_drawdown_pct']:>14.2f}%")
        print(f"  Max DD Duration      : {results['max_drawdown_duration_bars']:>12} bars")
        print(f"  Ulcer Index          : {results['ulcer_index']:>15.4f}")
        print(f"  VaR (95%)            : {results['var_95_pct']:>14.2f}%")
        print(f"  CVaR (95%)           : {results['cvar_95_pct']:>14.2f}%")
        print(f"  Annualized Vol       : {results['daily_vol_annualized_pct']:>14.2f}%")
        print(f"  Tail Ratio           : {results['tail_ratio']:>15.4f}")

        print(f"\n{'TRADE STATISTICS':^65}")
        print(sep)
        print(f"  Total Trades         : {results['total_trades']:>15,}")
        print(f"  Long / Short         : {results['long_trades']:>6,} / {results['short_trades']:<6,}")
        print(f"  Win Rate             : {results['win_rate_pct']:>14.2f}%")
        print(f"  Profit Factor        : {results['profit_factor']:>15.4f}")
        print(f"  Expectancy / Trade   : ${results['expectancy_per_trade']:>14,.2f}")
        print(f"  Kelly Criterion      : {results['kelly_criterion_pct']:>14.2f}%")
        print(f"  Avg Trade Duration   : {results['avg_trade_duration_bars']:>12.1f} bars")

        print(f"\n{'TRADE PnL':^65}")
        print(sep)
        print(f"  Gross Profit         : ${results['gross_profit']:>14,.2f}")
        print(f"  Gross Loss           : ${results['gross_loss']:>14,.2f}")
        print(f"  Avg Win              : ${results['avg_win']:>14,.2f}")
        print(f"  Avg Loss             : ${results['avg_loss']:>14,.2f}")
        print(f"  Largest Win          : ${results['largest_win']:>14,.2f}")
        print(f"  Largest Loss         : ${results['largest_loss']:>14,.2f}")
        print(f"  Avg MAE              : {results['avg_mae_pct']:>14.2f}%")
        print(f"  Avg MFE              : {results['avg_mfe_pct']:>14.2f}%")

        print(f"\n{'STREAK ANALYSIS':^65}")
        print(sep)
        print(f"  Max Consec. Wins     : {results['max_consecutive_wins']:>15}")
        print(f"  Max Consec. Losses   : {results['max_consecutive_losses']:>15}")
        print(f"  Avg Consec. Wins     : {results['avg_consecutive_wins']:>15.1f}")
        print(f"  Avg Consec. Losses   : {results['avg_consecutive_losses']:>15.1f}")

        print(f"\n{'EXIT REASON BREAKDOWN':^65}")
        print(sep)
        for reason, count in results.get("exit_reasons", {}).items():
            print(f"  {reason:<25}: {count:>5}")
        print(f"\n{'REGIME BREAKDOWN':^65}")
        print(sep)
        for regime, stats in results.get("regime_breakdown", {}).items():
            print(f"  {regime:<18}: trades={stats['n_trades']:>4}  "
                  f"total PnL=${stats['total_pnl']:>10,.2f}  "
                  f"avg=${stats['avg_pnl']:>8,.2f}")

        if bench:
            print(f"\n{'BENCHMARK COMPARISON':^65}")
            print(sep)
            for label, d in [("Strategy", bench["strategy"]), ("Buy & Hold", bench["buy_and_hold"])]:
                print(f"  {label}")
                print(f"    Return: {d['total_return_pct']:.2f}%  |  "
                      f"Sharpe: {d['sharpe']:.3f}  |  "
                      f"MaxDD: {d['max_drawdown_pct']:.2f}%  |  "
                      f"Vol: {d['annualized_vol_pct']:.2f}%")
            print(f"  Alpha (ann.)         : {bench['alpha_annualized_pct']:.2f}%")
            print(f"  Beta                 : {bench['beta']:.4f}")
            print(f"  Information Ratio    : {bench['information_ratio']:.4f}")
            print(f"  Benchmark Corr.      : {bench['correlation_with_benchmark']:.4f}")
        if mc_results:
            print(f"\n{'MONTE CARLO ({} sims)'.format(mc_results['n_simulations']):^65}")
            print(sep)
            fe = mc_results["final_equity"]
            print(f"  Prob. of Profit      : {fe['prob_profit']*100:>14.1f}%")
            print(f"  Median Final Equity  : ${fe['median']:>14,.2f}")
            print(f"  5th Pct Equity       : ${fe.get('p2', fe.get('p5', 0)):>14,.2f}")
            print(f"  95th Pct Equity      : ${fe.get('p97', fe.get('p95', 0)):>14,.2f}")
            dd = mc_results["max_drawdown_pct"]
            print(f"  Worst 5% MaxDD       : {dd['worst_5pct']:>14.2f}%")
            sr = mc_results["sharpe_ratio"]
            print(f"  Sharpe (p5-p95)      : [{sr['p5']:.3f}, {sr['p95']:.3f}]")

        if wf_results is not None and not wf_results["fold_results"].empty:
            print(f"\n{'WALK-FORWARD SUMMARY':^65}")
            print(sep)
            print(f"  Avg OOS Score        : {wf_results['avg_oos_score']:>15.4f}")
            print(f"  Avg OOS Return       : {wf_results['avg_oos_return_pct']:>14.2f}%")
            print(f"  Efficiency Ratio     : {wf_results['efficiency_ratio']:>15.4f}")
            print(f"  (Efficiency < 1 = overfitting risk)")
            print(f"\n  Fold Details:")
            print(wf_results["fold_results"].to_string(index=False))

        print(f"\n{SEP}\n")

# BUILT-IN EXAMPLE STRATEGIES (plug-and-play)


def rsi_mean_reversion_signal(df: pd.DataFrame, i: int,
                               oversold: float = 30,
                               overbought: float = 70) -> Optional[PositionSide]:
    if i < 1 or 'RSI' not in df.columns:
        return None
    rsi = df['RSI'].iloc[i]
    if rsi < oversold:
        return PositionSide.LONG
    elif rsi > overbought:
        return PositionSide.SHORT
    return None
