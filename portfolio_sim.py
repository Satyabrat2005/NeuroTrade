"""
NeuroTrade — Portfolio Simulator
portfolio_sim.py — Multi-asset portfolio simulation with correlation analysis.

Features:
  - Multi-asset allocation (equal, risk parity, min variance, max sharpe)
  - Correlation matrix & rolling correlation
  - Portfolio rebalancing engine
  - Risk decomposition (marginal/component VaR)
  - Efficient frontier computation
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

warnings.filterwarnings("ignore")

try:
    from scipy.optimize import minimize
    _SCIPY = True
except ImportError:
    _SCIPY = False
    print("[portfolio_sim] scipy not found — pip install scipy")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class AllocationMethod(Enum):
    EQUAL_WEIGHT    = "equal"
    RISK_PARITY     = "risk_parity"
    MIN_VARIANCE    = "min_variance"
    MAX_SHARPE      = "max_sharpe"
    INVERSE_VOL     = "inverse_vol"

@dataclass
class PortfolioConfig:
    initial_capital:    float = 100_000
    allocation:         str   = "equal"
    rebalance_freq:     str   = "monthly"   # "daily"|"weekly"|"monthly"|"quarterly"
    commission_pct:     float = 0.001
    risk_free_rate:     float = 0.06
    ann_factor:         int   = 252
    max_weight:         float = 0.40
    min_weight:         float = 0.02
    target_vol:         float = 0.15


# ══════════════════════════════════════════════════════════════════════════════
#  CORRELATION ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class CorrelationAnalyzer:
    """Full correlation analysis for multi-asset portfolios."""

    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        return returns.corr()

    @staticmethod
    def covariance_matrix(returns: pd.DataFrame, ann: int = 252) -> pd.DataFrame:
        return returns.cov() * ann

    @staticmethod
    def rolling_correlation(returns: pd.DataFrame, window: int = 60,
                            asset_a: str = None, asset_b: str = None) -> pd.Series:
        if asset_a and asset_b:
            return returns[asset_a].rolling(window).corr(returns[asset_b])
        # return avg pairwise correlation
        corrs = []
        cols = returns.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                c = returns[cols[i]].rolling(window).corr(returns[cols[j]])
                corrs.append(c)
        return pd.concat(corrs, axis=1).mean(axis=1)

    @staticmethod
    def diversification_ratio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.dot(weights, vols)
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        return weighted_vol / port_vol if port_vol > 0 else 1.0

    @staticmethod
    def max_correlation_pair(returns: pd.DataFrame) -> Tuple[str, str, float]:
        corr = returns.corr()
        np.fill_diagonal(corr.values, 0)
        idx = np.unravel_index(np.abs(corr.values).argmax(), corr.shape)
        return corr.columns[idx[0]], corr.columns[idx[1]], corr.values[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  ALLOCATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AllocationEngine:
    """Computes optimal portfolio weights."""

    @staticmethod
    def equal_weight(n_assets: int) -> np.ndarray:
        return np.ones(n_assets) / n_assets

    @staticmethod
    def inverse_volatility(returns: pd.DataFrame) -> np.ndarray:
        vols = returns.std()
        inv_vol = 1.0 / (vols + 1e-8)
        return (inv_vol / inv_vol.sum()).values

    @staticmethod
    def risk_parity(cov_matrix: np.ndarray) -> np.ndarray:
        n = cov_matrix.shape[0]
        w0 = np.ones(n) / n

        def risk_contribution(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            mrc = cov_matrix @ w / port_vol
            rc = w * mrc
            target_rc = port_vol / n
            return np.sum((rc - target_rc) ** 2)

        if not _SCIPY:
            return w0

        bounds = [(0.01, 1.0)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(risk_contribution, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    @staticmethod
    def min_variance(cov_matrix: np.ndarray, max_w: float = 0.4) -> np.ndarray:
        n = cov_matrix.shape[0]
        w0 = np.ones(n) / n

        if not _SCIPY:
            return w0

        bounds = [(0.01, max_w)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(lambda w: w @ cov_matrix @ w, w0,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    @staticmethod
    def max_sharpe(returns: pd.DataFrame, rf: float = 0.06,
                   ann: int = 252, max_w: float = 0.4) -> np.ndarray:
        n = returns.shape[1]
        mu = returns.mean().values * ann
        cov = returns.cov().values * ann
        w0 = np.ones(n) / n

        if not _SCIPY:
            return w0

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            return -(port_ret - rf) / (port_vol + 1e-8)

        bounds = [(0.01, max_w)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(neg_sharpe, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    @classmethod
    def compute(cls, method: str, returns: pd.DataFrame,
                cfg: PortfolioConfig = None) -> np.ndarray:
        cfg = cfg or PortfolioConfig()
        n = returns.shape[1]
        cov = returns.cov().values * cfg.ann_factor

        if method == "equal":
            w = cls.equal_weight(n)
        elif method == "inverse_vol":
            w = cls.inverse_volatility(returns)
        elif method == "risk_parity":
            w = cls.risk_parity(cov)
        elif method == "min_variance":
            w = cls.min_variance(cov, cfg.max_weight)
        elif method == "max_sharpe":
            w = cls.max_sharpe(returns, cfg.risk_free_rate, cfg.ann_factor, cfg.max_weight)
        else:
            w = cls.equal_weight(n)

        # enforce constraints
        w = np.clip(w, cfg.min_weight, cfg.max_weight)
        w = w / w.sum()
        return w


# ══════════════════════════════════════════════════════════════════════════════
#  RISK DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

class RiskDecomposition:
    """Component and marginal risk analysis."""

    @staticmethod
    def marginal_risk(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        port_vol = np.sqrt(weights @ cov @ weights)
        return (cov @ weights) / port_vol

    @staticmethod
    def component_risk(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        mrc = RiskDecomposition.marginal_risk(weights, cov)
        return weights * mrc

    @staticmethod
    def risk_contribution_pct(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        cr = RiskDecomposition.component_risk(weights, cov)
        return cr / cr.sum() * 100

    @staticmethod
    def component_var(weights: np.ndarray, returns: pd.DataFrame,
                      confidence: float = 0.95) -> Dict:
        port_returns = (returns * weights).sum(axis=1)
        total_var = np.percentile(port_returns, (1 - confidence) * 100)
        component_vars = {}
        for i, col in enumerate(returns.columns):
            marginal = np.percentile(returns[col] * weights[i],
                                      (1 - confidence) * 100)
            component_vars[col] = float(marginal)
        return {"total_var": float(total_var), "component_var": component_vars}


# ══════════════════════════════════════════════════════════════════════════════
#  EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════════════════════

class EfficientFrontier:
    """Compute the efficient frontier for a set of assets."""

    @staticmethod
    def compute(returns: pd.DataFrame, n_points: int = 50,
                rf: float = 0.06, ann: int = 252) -> Dict:
        if not _SCIPY:
            return {"error": "scipy required"}

        mu = returns.mean().values * ann
        cov = returns.cov().values * ann
        n = len(mu)

        ret_range = np.linspace(mu.min(), mu.max(), n_points)
        frontier_vols = []
        frontier_rets = []
        frontier_weights = []

        for target_ret in ret_range:
            w0 = np.ones(n) / n
            bounds = [(0.01, 0.5)] * n
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, r=target_ret: w @ mu - r},
            ]
            result = minimize(lambda w: w @ cov @ w, w0,
                             method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                vol = np.sqrt(result.x @ cov @ result.x)
                frontier_vols.append(vol)
                frontier_rets.append(target_ret)
                frontier_weights.append(result.x)

        # find tangency portfolio (max Sharpe)
        if frontier_vols:
            sharpes = [(r - rf) / (v + 1e-8) for r, v in zip(frontier_rets, frontier_vols)]
            best_idx = int(np.argmax(sharpes))
            tangency = {
                "return": frontier_rets[best_idx],
                "volatility": frontier_vols[best_idx],
                "sharpe": sharpes[best_idx],
                "weights": frontier_weights[best_idx].tolist(),
            }
        else:
            tangency = {}

        return {
            "volatilities": frontier_vols,
            "returns": frontier_rets,
            "tangency_portfolio": tangency,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioSimulator:
    """
    Multi-asset portfolio backtester with rebalancing.

    Usage
    -----
        sim = PortfolioSimulator(cfg)
        result = sim.run(price_dict)  # {ticker: DataFrame}
    """

    def __init__(self, cfg: PortfolioConfig = None):
        self.cfg = cfg or PortfolioConfig()

    def run(self, prices: Dict[str, pd.DataFrame]) -> Dict:
        """
        prices: {ticker: DataFrame with 'Close' column}
        """
        cfg = self.cfg

        # align all series
        close_df = pd.DataFrame({
            ticker: df["Close"] for ticker, df in prices.items()
        }).dropna()

        returns = close_df.pct_change().dropna()
        tickers = list(close_df.columns)
        n_assets = len(tickers)

        # initial allocation
        weights = AllocationEngine.compute(cfg.allocation, returns, cfg)
        weight_history = [{"date": returns.index[0], **dict(zip(tickers, weights))}]

        # simulate
        capital = cfg.initial_capital
        holdings = capital * weights / close_df.iloc[0].values
        equity_curve = []
        rebalance_dates = self._get_rebalance_dates(returns.index, cfg.rebalance_freq)

        for i, date in enumerate(close_df.index):
            prices_today = close_df.loc[date].values
            portfolio_value = np.sum(holdings * prices_today)
            equity_curve.append({"date": date, "equity": portfolio_value})

            # rebalance
            if date in rebalance_dates and i > 30:
                lookback = returns.iloc[max(0, i-120):i]
                if len(lookback) > 20:
                    new_weights = AllocationEngine.compute(cfg.allocation, lookback, cfg)
                    # apply commission
                    turnover = np.sum(np.abs(new_weights - weights))
                    commission = portfolio_value * turnover * cfg.commission_pct
                    portfolio_value -= commission
                    weights = new_weights
                    holdings = portfolio_value * weights / prices_today
                    weight_history.append({"date": date, **dict(zip(tickers, weights))})

        eq_df = pd.DataFrame(equity_curve).set_index("date")
        port_returns = eq_df["equity"].pct_change().dropna()

        # correlation analysis
        corr_analyzer = CorrelationAnalyzer()
        corr_matrix = corr_analyzer.correlation_matrix(returns)
        cov_matrix = returns.cov().values * cfg.ann_factor

        # risk decomposition
        risk_pct = RiskDecomposition.risk_contribution_pct(weights, cov_matrix)
        risk_contrib = dict(zip(tickers, risk_pct.tolist()))

        # compute metrics
        total_return = (eq_df["equity"].iloc[-1] / cfg.initial_capital - 1)
        n_days = len(eq_df)
        ann_return = (1 + total_return) ** (cfg.ann_factor / n_days) - 1
        ann_vol = port_returns.std() * np.sqrt(cfg.ann_factor)
        sharpe = (ann_return - cfg.risk_free_rate) / (ann_vol + 1e-8)

        cum_max = eq_df["equity"].cummax()
        dd = (eq_df["equity"] - cum_max) / cum_max
        max_dd = dd.min()

        div_ratio = corr_analyzer.diversification_ratio(weights, cov_matrix)

        return {
            "initial_capital": cfg.initial_capital,
            "final_capital": float(eq_df["equity"].iloc[-1]),
            "total_return_pct": total_return * 100,
            "annualized_return_pct": ann_return * 100,
            "annualized_vol_pct": ann_vol * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "allocation_method": cfg.allocation,
            "final_weights": dict(zip(tickers, weights.tolist())),
            "risk_contribution_pct": risk_contrib,
            "correlation_matrix": corr_matrix.to_dict(),
            "diversification_ratio": div_ratio,
            "n_rebalances": len(weight_history) - 1,
            "equity_curve": eq_df,
            "weight_history": pd.DataFrame(weight_history).set_index("date"),
            "returns": port_returns,
        }

    @staticmethod
    def _get_rebalance_dates(index, freq):
        if freq == "daily":
            return set(index)
        elif freq == "weekly":
            return set(index[index.dayofweek == 0])
        elif freq == "monthly":
            monthly = index.to_period("M")
            return set(index.groupby(monthly).first())
        elif freq == "quarterly":
            quarterly = index.to_period("Q")
            return set(index.groupby(quarterly).first())
        return set()


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  portfolio_sim.py — Portfolio Simulator Self-Test")
    print("=" * 65)

    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # simulate 4 correlated assets
    base = np.cumsum(np.random.normal(0.0003, 0.012, n))
    prices = {}
    for ticker, drift, vol, corr in [
        ("AAPL", 0.0004, 0.015, 0.7),
        ("GOOGL", 0.0003, 0.018, 0.6),
        ("MSFT", 0.00035, 0.014, 0.8),
        ("TSLA", 0.0005, 0.025, 0.4),
    ]:
        noise = np.random.normal(drift, vol, n)
        asset_returns = corr * base / base.std() * vol + (1-corr) * noise
        close = 100 * np.exp(np.cumsum(asset_returns))
        prices[ticker] = pd.DataFrame({
            "Close": close, "Open": close * 0.999,
            "High": close * 1.01, "Low": close * 0.99,
            "Volume": np.random.randint(1e6, 1e7, n).astype(float),
        }, index=dates)

    for method in ["equal", "inverse_vol", "risk_parity", "min_variance", "max_sharpe"]:
        print(f"\n  {method.upper()}")
        print("  " + "-" * 40)
        cfg = PortfolioConfig(allocation=method, rebalance_freq="monthly")
        sim = PortfolioSimulator(cfg)
        result = sim.run(prices)
        print(f"    Return: {result['total_return_pct']:.2f}%"
              f"  Sharpe: {result['sharpe_ratio']:.3f}"
              f"  MaxDD: {result['max_drawdown_pct']:.2f}%")
        print(f"    Weights: {result['final_weights']}")
        print(f"    Risk Contrib: {result['risk_contribution_pct']}")
        print(f"    Div Ratio: {result['diversification_ratio']:.3f}")

    print("\n  ✓  Portfolio simulator self-test complete.\n")
