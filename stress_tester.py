"""
NeuroTrade — Stress Testing Engine
stress_tester.py — Historical crisis replay + synthetic shock simulation.

Scenarios:
  - 2008 GFC (Global Financial Crisis)
  - 2020 COVID crash
  - 2010 Flash Crash
  - 2022 Rate Shock
  - Custom user-defined scenarios
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  HISTORICAL CRISIS PROFILES
# ══════════════════════════════════════════════════════════════════════════════

CRISIS_PROFILES = {
    "2008_gfc": {
        "name": "2008 Global Financial Crisis",
        "duration_days": 350,
        "phases": [
            {"name": "Initial Sell-off",   "days": 30,  "daily_ret": -0.012, "vol": 0.035},
            {"name": "Bear Market Rally",  "days": 15,  "daily_ret":  0.005, "vol": 0.025},
            {"name": "Lehman Collapse",    "days": 20,  "daily_ret": -0.025, "vol": 0.055},
            {"name": "Capitulation",       "days": 40,  "daily_ret": -0.018, "vol": 0.065},
            {"name": "Dead Cat Bounce",    "days": 10,  "daily_ret":  0.008, "vol": 0.040},
            {"name": "Grinding Lower",     "days": 90,  "daily_ret": -0.005, "vol": 0.030},
            {"name": "March 2009 Bottom",  "days": 15,  "daily_ret": -0.020, "vol": 0.050},
            {"name": "Recovery Begin",     "days": 130, "daily_ret":  0.004, "vol": 0.022},
        ],
        "peak_drawdown": -0.567,
        "vix_peak": 80.86,
        "description": "Subprime mortgage crisis → Lehman collapse → global credit freeze",
    },
    "2020_covid": {
        "name": "2020 COVID-19 Crash",
        "duration_days": 120,
        "phases": [
            {"name": "Pre-Crash Rally",  "days": 10,  "daily_ret":  0.002, "vol": 0.010},
            {"name": "Initial Drop",     "days": 10,  "daily_ret": -0.030, "vol": 0.045},
            {"name": "Circuit Breakers", "days": 15,  "daily_ret": -0.040, "vol": 0.080},
            {"name": "Brief Bounce",     "days": 5,   "daily_ret":  0.012, "vol": 0.050},
            {"name": "Final Panic",      "days": 10,  "daily_ret": -0.035, "vol": 0.070},
            {"name": "Fed Intervention", "days": 5,   "daily_ret":  0.020, "vol": 0.045},
            {"name": "V-Recovery",       "days": 65,  "daily_ret":  0.008, "vol": 0.025},
        ],
        "peak_drawdown": -0.339,
        "vix_peak": 82.69,
        "description": "Fastest bear market in history → unprecedented Fed stimulus → V-recovery",
    },
    "2010_flash_crash": {
        "name": "2010 Flash Crash",
        "duration_days": 10,
        "phases": [
            {"name": "Normal Trading",   "days": 2,  "daily_ret": -0.001, "vol": 0.012},
            {"name": "Flash Crash",      "days": 1,  "daily_ret": -0.090, "vol": 0.200},
            {"name": "Partial Recovery",  "days": 1,  "daily_ret":  0.050, "vol": 0.120},
            {"name": "Aftermath",        "days": 6,  "daily_ret":  0.005, "vol": 0.025},
        ],
        "peak_drawdown": -0.098,
        "vix_peak": 40.0,
        "description": "Dow drops ~1000pts in minutes due to HFT liquidity vacuum",
    },
    "2022_rate_shock": {
        "name": "2022 Fed Rate Hike Cycle",
        "duration_days": 280,
        "phases": [
            {"name": "Hawkish Pivot",     "days": 30,  "daily_ret": -0.006, "vol": 0.018},
            {"name": "Tech Sell-off",     "days": 45,  "daily_ret": -0.008, "vol": 0.022},
            {"name": "Bear Rally",        "days": 20,  "daily_ret":  0.005, "vol": 0.018},
            {"name": "Summer Grind",      "days": 50,  "daily_ret": -0.004, "vol": 0.020},
            {"name": "Aug Rally",         "days": 15,  "daily_ret":  0.006, "vol": 0.015},
            {"name": "Sep-Oct Sell-off",  "days": 45,  "daily_ret": -0.007, "vol": 0.022},
            {"name": "Oct Bottom",        "days": 10,  "daily_ret": -0.010, "vol": 0.028},
            {"name": "Q4 Recovery",       "days": 65,  "daily_ret":  0.003, "vol": 0.015},
        ],
        "peak_drawdown": -0.275,
        "vix_peak": 36.45,
        "description": "Aggressive Fed tightening → growth-to-value rotation → tech bear market",
    },
    "1987_black_monday": {
        "name": "1987 Black Monday",
        "duration_days": 30,
        "phases": [
            {"name": "Pre-Crash Weakness", "days": 5,  "daily_ret": -0.003, "vol": 0.015},
            {"name": "Black Monday",       "days": 1,  "daily_ret": -0.226, "vol": 0.300},
            {"name": "Aftershock",         "days": 4,  "daily_ret": -0.010, "vol": 0.080},
            {"name": "Stabilization",      "days": 20, "daily_ret":  0.003, "vol": 0.030},
        ],
        "peak_drawdown": -0.226,
        "vix_peak": 150.0,
        "description": "S&P 500 drops 22.6% in a single day — portfolio insurance cascade",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioGenerator:
    """Generates synthetic price paths from crisis profiles."""

    @staticmethod
    def generate(profile: Dict, initial_price: float = 100.0,
                 seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        all_returns = []
        phase_labels = []

        for phase in profile["phases"]:
            n = phase["days"]
            mu = phase["daily_ret"]
            sigma = phase["vol"]
            rets = np.random.normal(mu, sigma, n)
            all_returns.extend(rets.tolist())
            phase_labels.extend([phase["name"]] * n)

        returns = np.array(all_returns)
        prices = initial_price * np.exp(np.cumsum(returns))
        prices = np.insert(prices, 0, initial_price)
        returns = np.insert(returns, 0, 0)
        phase_labels.insert(0, "Start")

        n_total = len(prices)
        dates = pd.date_range("2020-01-01", periods=n_total, freq="B")

        vol_noise = np.abs(returns) * np.random.uniform(0.3, 0.7, n_total)

        df = pd.DataFrame({
            "Open":   prices * (1 + np.random.normal(0, 0.002, n_total)),
            "High":   prices * (1 + vol_noise),
            "Low":    prices * (1 - vol_noise),
            "Close":  prices,
            "Volume": np.random.randint(5_000_000, 50_000_000, n_total).astype(float),
            "Phase":  phase_labels,
        }, index=dates)

        df["Returns"] = df["Close"].pct_change()
        df["Cumulative_Return"] = (1 + df["Returns"].fillna(0)).cumprod() - 1
        df["Drawdown"] = df["Close"] / df["Close"].cummax() - 1

        return df

    @staticmethod
    def custom_shock(initial_price: float = 100.0,
                     shock_pct: float = -0.30,
                     recovery_days: int = 60,
                     pre_days: int = 20,
                     crash_days: int = 5) -> pd.DataFrame:
        """Generate custom shock scenario."""
        profile = {
            "phases": [
                {"name": "Pre-Shock",  "days": pre_days,
                 "daily_ret": 0.001, "vol": 0.012},
                {"name": "Crash",      "days": crash_days,
                 "daily_ret": shock_pct / crash_days, "vol": abs(shock_pct) / crash_days * 0.5},
                {"name": "Recovery",   "days": recovery_days,
                 "daily_ret": abs(shock_pct) / recovery_days * 0.7, "vol": 0.020},
            ],
        }
        return ScenarioGenerator.generate(profile, initial_price)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO STRESS TESTER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StressConfig:
    initial_capital:  float = 100_000
    position_pct:     float = 1.0
    stop_loss_pct:    Optional[float] = None
    leverage:         float = 1.0


class StressTester:
    """
    Tests a portfolio against historical crisis scenarios.

    Usage
    -----
        tester = StressTester(cfg)
        result = tester.run_scenario("2008_gfc")
        report = tester.run_all()
    """

    def __init__(self, cfg: StressConfig = None):
        self.cfg = cfg or StressConfig()

    def run_scenario(self, scenario_key: str,
                     signal_func: Callable = None) -> Dict:
        """Run a single stress scenario."""
        profile = CRISIS_PROFILES.get(scenario_key)
        if profile is None:
            raise ValueError(f"Unknown scenario: {scenario_key}. "
                             f"Available: {list(CRISIS_PROFILES.keys())}")

        df = ScenarioGenerator.generate(profile, 100.0)
        return self._simulate(df, profile, signal_func)

    def run_custom(self, df: pd.DataFrame, name: str = "Custom",
                   signal_func: Callable = None) -> Dict:
        """Run on user-supplied crisis data."""
        profile = {"name": name, "description": "User-defined scenario"}
        return self._simulate(df, profile, signal_func)

    def run_all(self, signal_func: Callable = None) -> Dict[str, Dict]:
        """Run all predefined crisis scenarios."""
        results = {}
        for key in CRISIS_PROFILES:
            results[key] = self.run_scenario(key, signal_func)
        return results

    def _simulate(self, df: pd.DataFrame, profile: Dict,
                  signal_func: Callable = None) -> Dict:
        """Simulate portfolio through scenario."""
        cfg = self.cfg
        capital = cfg.initial_capital
        position_value = capital * cfg.position_pct * cfg.leverage
        shares = position_value / df["Close"].iloc[0]

        equity = []
        peak_equity = capital
        max_dd = 0
        stopped_out = False
        stop_bar = -1

        for i in range(len(df)):
            price = df["Close"].iloc[i]
            port_value = shares * price
            unrealized_pnl = port_value - position_value
            current_equity = capital + unrealized_pnl * cfg.position_pct

            # stop loss check
            if cfg.stop_loss_pct and not stopped_out:
                loss_pct = (current_equity - cfg.initial_capital) / cfg.initial_capital
                if loss_pct < -cfg.stop_loss_pct:
                    stopped_out = True
                    stop_bar = i
                    capital = current_equity
                    shares = 0

            if stopped_out:
                current_equity = capital  # flat after stop

            equity.append(current_equity)
            peak_equity = max(peak_equity, current_equity)
            dd = (current_equity - peak_equity) / peak_equity
            max_dd = min(max_dd, dd)

        equity_series = pd.Series(equity, index=df.index)
        returns = equity_series.pct_change().dropna()

        # phase analysis
        phase_results = {}
        if "Phase" in df.columns:
            for phase in df["Phase"].unique():
                mask = df["Phase"] == phase
                phase_eq = equity_series[mask]
                if len(phase_eq) > 1:
                    phase_ret = (phase_eq.iloc[-1] / phase_eq.iloc[0] - 1)
                    phase_results[phase] = {
                        "return_pct": round(phase_ret * 100, 2),
                        "bars": int(mask.sum()),
                    }

        total_return = (equity_series.iloc[-1] / cfg.initial_capital - 1)

        return {
            "scenario": profile.get("name", "Unknown"),
            "description": profile.get("description", ""),
            "initial_capital": cfg.initial_capital,
            "final_capital": round(float(equity_series.iloc[-1]), 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "peak_drawdown_expected": round(
                profile.get("peak_drawdown", 0) * 100, 2),
            "worst_day_pct": round(float(returns.min()) * 100, 2) if len(returns) > 0 else 0,
            "best_day_pct": round(float(returns.max()) * 100, 2) if len(returns) > 0 else 0,
            "volatility_ann_pct": round(float(returns.std() * np.sqrt(252)) * 100, 2) if len(returns) > 0 else 0,
            "stopped_out": stopped_out,
            "stop_bar": stop_bar,
            "duration_bars": len(df),
            "phase_breakdown": phase_results,
            "equity_curve": equity_series,
            "scenario_df": df,
            "leverage": cfg.leverage,
        }

    def survival_analysis(self, stop_levels: List[float] = None) -> pd.DataFrame:
        """Test how different stop-loss levels affect survival across crises."""
        if stop_levels is None:
            stop_levels = [0.05, 0.10, 0.15, 0.20, 0.30, None]

        rows = []
        for scenario_key in CRISIS_PROFILES:
            for sl in stop_levels:
                cfg = StressConfig(
                    initial_capital=self.cfg.initial_capital,
                    position_pct=self.cfg.position_pct,
                    stop_loss_pct=sl,
                    leverage=self.cfg.leverage,
                )
                tester = StressTester(cfg)
                res = tester.run_scenario(scenario_key)
                rows.append({
                    "scenario": scenario_key,
                    "stop_loss": f"{sl*100:.0f}%" if sl else "None",
                    "return_pct": res["total_return_pct"],
                    "max_dd_pct": res["max_drawdown_pct"],
                    "stopped_out": res["stopped_out"],
                    "final_capital": res["final_capital"],
                })

        return pd.DataFrame(rows)

    def leverage_analysis(self, levels: List[float] = None) -> pd.DataFrame:
        """Test how different leverage levels affect portfolio in crises."""
        if levels is None:
            levels = [0.5, 1.0, 1.5, 2.0, 3.0]

        rows = []
        for scenario_key in CRISIS_PROFILES:
            for lev in levels:
                cfg = StressConfig(
                    initial_capital=self.cfg.initial_capital,
                    leverage=lev,
                )
                tester = StressTester(cfg)
                res = tester.run_scenario(scenario_key)
                rows.append({
                    "scenario": scenario_key,
                    "leverage": f"{lev:.1f}x",
                    "return_pct": res["total_return_pct"],
                    "max_dd_pct": res["max_drawdown_pct"],
                    "final_capital": res["final_capital"],
                })

        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════════════

class StressReportPrinter:
    """Pretty-print stress test results."""

    @staticmethod
    def print_single(result: Dict):
        SEP = "=" * 65
        sep = "-" * 65

        print(f"\n{SEP}")
        print(f"  STRESS TEST: {result['scenario']}")
        print(f"  {result.get('description', '')}")
        print(SEP)
        print(f"  Initial Capital   : ${result['initial_capital']:>14,.2f}")
        print(f"  Final Capital     : ${result['final_capital']:>14,.2f}")
        print(f"  Total Return      : {result['total_return_pct']:>13.2f}%")
        print(f"  Max Drawdown      : {result['max_drawdown_pct']:>13.2f}%")
        print(f"  Expected DD       : {result['peak_drawdown_expected']:>13.2f}%")
        print(f"  Worst Day         : {result['worst_day_pct']:>13.2f}%")
        print(f"  Best Day          : {result['best_day_pct']:>13.2f}%")
        print(f"  Ann. Volatility   : {result['volatility_ann_pct']:>13.2f}%")
        print(f"  Leverage          : {result['leverage']:>13.1f}x")
        print(f"  Stopped Out       : {'YES' if result['stopped_out'] else 'NO':>13}")
        print(f"  Duration          : {result['duration_bars']:>10} bars")

        phases = result.get("phase_breakdown", {})
        if phases:
            print(f"\n{sep}")
            print(f"  {'Phase':<25} {'Return':>10} {'Bars':>6}")
            print(sep)
            for phase, data in phases.items():
                print(f"  {phase:<25} {data['return_pct']:>9.2f}% {data['bars']:>6}")
        print()

    @staticmethod
    def print_all(results: Dict[str, Dict]):
        print("\n" + "=" * 75)
        print("  STRESS TEST SUMMARY — Portfolio Resilience Report")
        print("=" * 75)

        rows = []
        for key, res in results.items():
            rows.append({
                "Scenario": res["scenario"][:30],
                "Return%": f"{res['total_return_pct']:.1f}",
                "MaxDD%": f"{res['max_drawdown_pct']:.1f}",
                "Final$": f"${res['final_capital']:,.0f}",
                "WorstDay": f"{res['worst_day_pct']:.1f}%",
                "Stopped": "YES" if res["stopped_out"] else "NO",
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  stress_tester.py — Crisis Stress Testing Self-Test")
    print("=" * 65)

    cfg = StressConfig(initial_capital=100_000, position_pct=1.0)
    tester = StressTester(cfg)

    # Run all scenarios
    results = tester.run_all()
    StressReportPrinter.print_all(results)

    # Detailed single
    StressReportPrinter.print_single(results["2008_gfc"])
    StressReportPrinter.print_single(results["2020_covid"])

    # Survival analysis
    print("\n  Survival Analysis (stop-loss sensitivity):")
    survival = tester.survival_analysis()
    print(survival.to_string(index=False))

    # Custom shock
    print("\n  Custom Shock (-40% in 3 days):")
    custom_df = ScenarioGenerator.custom_shock(
        initial_price=100, shock_pct=-0.40, crash_days=3, recovery_days=30)
    custom_res = tester.run_custom(custom_df, "Custom -40% Shock")
    StressReportPrinter.print_single(custom_res)

    print("\n  ✓  Stress tester self-test complete.\n")
