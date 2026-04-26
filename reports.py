"""
NeuroTrade — PDF Tearsheet & Report Generator
reports.py — Professional PDF tearsheet export with performance analytics.

Generates institutional-quality PDF reports using matplotlib + reportlab.
Falls back to HTML if reportlab unavailable.
"""

import os
import io
import time
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    _MPL = True
except ImportError:
    _MPL = False
    print("[reports] matplotlib not found — pip install matplotlib")


# ══════════════════════════════════════════════════════════════════════════════
#  TEARSHEET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class TearsheetGenerator:
    """
    Generates professional PDF tearsheet from backtest results.

    Usage
    -----
        gen = TearsheetGenerator(results)
        gen.save("tearsheet.pdf")
        pdf_bytes = gen.to_bytes()  # for Streamlit download
    """

    COLORS = {
        "bg":      "#0a0e1a",
        "panel":   "#111827",
        "text":    "#e2e8f0",
        "muted":   "#64748b",
        "cyan":    "#38bdf8",
        "green":   "#4ade80",
        "red":     "#f87171",
        "amber":   "#fbbf24",
        "purple":  "#a78bfa",
        "grid":    "#1e293b",
    }

    def __init__(self, results: Dict, title: str = "NeuroTrade Tearsheet"):
        if not _MPL:
            raise ImportError("matplotlib required — pip install matplotlib")
        self.results = results
        self.title = title
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def save(self, filepath: str = "tearsheet.pdf"):
        """Save tearsheet to PDF file."""
        with PdfPages(filepath) as pdf:
            self._page_summary(pdf)
            self._page_equity(pdf)
            self._page_trades(pdf)
        print(f"  [PDF] Saved: {filepath}")

    def to_bytes(self) -> bytes:
        """Generate PDF as bytes (for Streamlit download)."""
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            self._page_summary(pdf)
            self._page_equity(pdf)
            self._page_trades(pdf)
        buf.seek(0)
        return buf.read()

    # ── Page 1: Summary ──────────────────────────────────────────

    def _page_summary(self, pdf):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5),
                                  gridspec_kw={"height_ratios": [1.2, 2, 2]})
        fig.set_facecolor(self.COLORS["bg"])

        r = self.results

        # Header panel
        ax = axes[0]
        ax.set_facecolor(self.COLORS["panel"])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis("off")

        ax.text(0.3, 2.2, self.title,
                fontsize=18, color=self.COLORS["cyan"],
                fontweight="bold", family="monospace")
        ax.text(0.3, 1.4, f"Generated: {self.timestamp}",
                fontsize=8, color=self.COLORS["muted"], family="monospace")

        ret_color = self.COLORS["green"] if r.get("total_return_pct", 0) >= 0 else self.COLORS["red"]
        ax.text(7, 2.2, f"{r.get('total_return_pct', 0):.2f}%",
                fontsize=22, color=ret_color, fontweight="bold",
                family="monospace", ha="right")
        ax.text(7, 1.4, "Total Return", fontsize=8,
                color=self.COLORS["muted"], family="monospace", ha="right")

        # Metrics table
        ax = axes[1]
        ax.set_facecolor(self.COLORS["panel"])
        ax.axis("off")

        metrics_left = [
            ("Initial Capital", f"${r.get('initial_capital', 0):,.0f}"),
            ("Final Capital", f"${r.get('final_capital', 0):,.0f}"),
            ("Annualized Return", f"{r.get('annualized_return_pct', 0):.2f}%"),
            ("Sharpe Ratio", f"{r.get('sharpe_ratio', 0):.4f}"),
            ("Sortino Ratio", f"{r.get('sortino_ratio', 0):.4f}"),
            ("Calmar Ratio", f"{r.get('calmar_ratio', 0):.4f}"),
            ("Omega Ratio", f"{r.get('omega_ratio', 0):.4f}"),
        ]
        metrics_right = [
            ("Max Drawdown", f"{r.get('max_drawdown_pct', 0):.2f}%"),
            ("VaR 95%", f"{r.get('var_95_pct', 0):.2f}%"),
            ("CVaR 95%", f"{r.get('cvar_95_pct', 0):.2f}%"),
            ("Win Rate", f"{r.get('win_rate_pct', 0):.1f}%"),
            ("Profit Factor", f"{r.get('profit_factor', 0):.3f}"),
            ("Total Trades", f"{r.get('total_trades', 0)}"),
            ("Kelly Criterion", f"{r.get('kelly_criterion_pct', 0):.1f}%"),
        ]

        for i, (label, val) in enumerate(metrics_left):
            y = 0.88 - i * 0.12
            ax.text(0.05, y, label, transform=ax.transAxes,
                    fontsize=8, color=self.COLORS["muted"], family="monospace")
            ax.text(0.40, y, val, transform=ax.transAxes,
                    fontsize=9, color=self.COLORS["text"], family="monospace",
                    fontweight="bold")

        for i, (label, val) in enumerate(metrics_right):
            y = 0.88 - i * 0.12
            ax.text(0.55, y, label, transform=ax.transAxes,
                    fontsize=8, color=self.COLORS["muted"], family="monospace")
            ax.text(0.90, y, val, transform=ax.transAxes,
                    fontsize=9, color=self.COLORS["text"], family="monospace",
                    fontweight="bold")

        # Trade PnL
        ax = axes[2]
        ax.set_facecolor(self.COLORS["panel"])
        ax.axis("off")

        trade_metrics = [
            ("Gross Profit", f"${r.get('gross_profit', 0):,.0f}"),
            ("Gross Loss", f"${r.get('gross_loss', 0):,.0f}"),
            ("Avg Win", f"${r.get('avg_win', 0):,.0f}"),
            ("Avg Loss", f"${r.get('avg_loss', 0):,.0f}"),
            ("Largest Win", f"${r.get('largest_win', 0):,.0f}"),
            ("Largest Loss", f"${r.get('largest_loss', 0):,.0f}"),
            ("Expectancy", f"${r.get('expectancy_per_trade', 0):,.0f}"),
            ("Avg Duration", f"{r.get('avg_trade_duration_bars', 0):.0f} bars"),
            ("Long Trades", f"{r.get('long_trades', 0)}"),
            ("Short Trades", f"{r.get('short_trades', 0)}"),
        ]

        for i, (label, val) in enumerate(trade_metrics):
            col = i % 2
            row = i // 2
            x = 0.05 + col * 0.50
            y = 0.88 - row * 0.17
            ax.text(x, y, label, transform=ax.transAxes,
                    fontsize=8, color=self.COLORS["muted"], family="monospace")
            ax.text(x + 0.30, y, val, transform=ax.transAxes,
                    fontsize=9, color=self.COLORS["text"], family="monospace",
                    fontweight="bold")

        plt.tight_layout(pad=1.0)
        pdf.savefig(fig, facecolor=self.COLORS["bg"])
        plt.close(fig)

    # ── Page 2: Equity Curve ─────────────────────────────────────

    def _page_equity(self, pdf):
        r = self.results
        eq_data = r.get("equity_curve")
        if eq_data is None:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5),
                                         gridspec_kw={"height_ratios": [3, 1]})
        fig.set_facecolor(self.COLORS["bg"])

        equity = eq_data["equity"] if isinstance(eq_data, pd.DataFrame) else eq_data

        # Equity curve
        ax1.set_facecolor(self.COLORS["panel"])
        ax1.plot(equity.index, equity.values,
                 color=self.COLORS["cyan"], linewidth=1.2, label="Equity")
        ax1.fill_between(equity.index, equity.values, alpha=0.05,
                         color=self.COLORS["cyan"])
        ax1.set_title("Equity Curve", fontsize=12,
                      color=self.COLORS["text"], family="monospace", pad=10)
        ax1.tick_params(colors=self.COLORS["muted"], labelsize=7)
        ax1.grid(True, alpha=0.1, color=self.COLORS["grid"])
        ax1.set_ylabel("Portfolio Value ($)", fontsize=8,
                       color=self.COLORS["muted"], family="monospace")

        # Drawdown
        dd = (equity - equity.cummax()) / equity.cummax() * 100
        ax2.set_facecolor(self.COLORS["panel"])
        ax2.fill_between(dd.index, dd.values, alpha=0.3, color=self.COLORS["red"])
        ax2.plot(dd.index, dd.values, color=self.COLORS["red"], linewidth=0.8)
        ax2.set_title("Drawdown (%)", fontsize=10,
                      color=self.COLORS["muted"], family="monospace", pad=5)
        ax2.tick_params(colors=self.COLORS["muted"], labelsize=7)
        ax2.grid(True, alpha=0.1, color=self.COLORS["grid"])

        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=self.COLORS["bg"])
        plt.close(fig)

    # ── Page 3: Trade Analysis ───────────────────────────────────

    def _page_trades(self, pdf):
        r = self.results
        trades = r.get("trades", [])
        if not trades:
            return

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.set_facecolor(self.COLORS["bg"])

        # PnL distribution
        ax = axes[0][0]
        ax.set_facecolor(self.COLORS["panel"])
        pnls = [t.pnl for t in trades]
        colors = [self.COLORS["green"] if p > 0 else self.COLORS["red"] for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1.0)
        ax.set_title("Trade PnL", fontsize=10,
                     color=self.COLORS["text"], family="monospace")
        ax.tick_params(colors=self.COLORS["muted"], labelsize=7)
        ax.axhline(y=0, color=self.COLORS["muted"], linewidth=0.5)
        ax.grid(True, alpha=0.1, color=self.COLORS["grid"])

        # PnL histogram
        ax = axes[0][1]
        ax.set_facecolor(self.COLORS["panel"])
        ax.hist(pnls, bins=min(30, len(pnls)),
                color=self.COLORS["cyan"], alpha=0.6, edgecolor="none")
        ax.axvline(x=0, color=self.COLORS["red"], linewidth=0.8, linestyle="--")
        ax.axvline(x=np.mean(pnls), color=self.COLORS["green"],
                   linewidth=0.8, linestyle="--", label=f"Mean: ${np.mean(pnls):,.0f}")
        ax.set_title("PnL Distribution", fontsize=10,
                     color=self.COLORS["text"], family="monospace")
        ax.tick_params(colors=self.COLORS["muted"], labelsize=7)
        ax.legend(fontsize=7, facecolor=self.COLORS["panel"],
                  edgecolor="none", labelcolor=self.COLORS["text"])

        # Win/Loss pie
        ax = axes[1][0]
        ax.set_facecolor(self.COLORS["panel"])
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = len(trades) - wins
        ax.pie([wins, losses], labels=["Wins", "Losses"],
               colors=[self.COLORS["green"], self.COLORS["red"]],
               autopct="%1.1f%%", textprops={"fontsize": 9, "color": self.COLORS["text"]},
               startangle=90)
        ax.set_title("Win/Loss Ratio", fontsize=10,
                     color=self.COLORS["text"], family="monospace")

        # Exit reasons
        ax = axes[1][1]
        ax.set_facecolor(self.COLORS["panel"])
        exit_reasons = r.get("exit_reasons", {})
        if exit_reasons:
            reasons = list(exit_reasons.keys())
            counts = list(exit_reasons.values())
            bars = ax.barh(reasons, counts, color=self.COLORS["cyan"], alpha=0.7)
            ax.set_title("Exit Reasons", fontsize=10,
                         color=self.COLORS["text"], family="monospace")
            ax.tick_params(colors=self.COLORS["muted"], labelsize=7)
        else:
            ax.text(0.5, 0.5, "No exit data", ha="center", va="center",
                    color=self.COLORS["muted"], fontsize=10, family="monospace",
                    transform=ax.transAxes)

        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=self.COLORS["bg"])
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT (fallback when no matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

class HTMLReportGenerator:
    """Generates HTML tearsheet (works without matplotlib)."""

    @staticmethod
    def generate(results: Dict, title: str = "NeuroTrade Report") -> str:
        r = results
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:'IBM Plex Mono',monospace;background:#0a0e1a;color:#e2e8f0;padding:40px;}}
h1{{color:#38bdf8;font-size:1.8rem;margin-bottom:4px;}}
h2{{color:#a78bfa;font-size:1.1rem;margin-top:30px;border-bottom:1px solid #1e293b;padding-bottom:6px;}}
.subtitle{{color:#64748b;font-size:0.75rem;margin-bottom:20px;}}
table{{border-collapse:collapse;width:100%;margin:10px 0;}}
th{{text-align:left;color:#64748b;font-size:0.7rem;text-transform:uppercase;padding:6px 12px;border-bottom:1px solid #1e293b;}}
td{{padding:6px 12px;border-bottom:1px solid #0f172a;font-size:0.8rem;}}
.pos{{color:#4ade80;}} .neg{{color:#f87171;}}
.metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:15px 0;}}
.metric{{background:#111827;border:1px solid #1e293b;border-radius:8px;padding:14px;}}
.metric-label{{color:#64748b;font-size:0.65rem;text-transform:uppercase;}}
.metric-value{{font-size:1.1rem;font-weight:bold;margin-top:4px;}}
</style></head><body>
<h1>⚡ {title}</h1>
<div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<div class="metric-grid">"""

        metrics = [
            ("Total Return", f"{r.get('total_return_pct',0):.2f}%",
             "pos" if r.get("total_return_pct", 0) >= 0 else "neg"),
            ("Sharpe", f"{r.get('sharpe_ratio',0):.4f}", ""),
            ("Max DD", f"{r.get('max_drawdown_pct',0):.2f}%", "neg"),
            ("Win Rate", f"{r.get('win_rate_pct',0):.1f}%", ""),
        ]
        for label, val, cls in metrics:
            html += f"""<div class="metric">
<div class="metric-label">{label}</div>
<div class="metric-value {cls}">{val}</div></div>"""

        html += """</div><h2>Performance</h2><table><tr><th>Metric</th><th>Value</th></tr>"""

        perf = [
            ("Initial Capital", f"${r.get('initial_capital',0):,.0f}"),
            ("Final Capital", f"${r.get('final_capital',0):,.0f}"),
            ("Ann. Return", f"{r.get('annualized_return_pct',0):.2f}%"),
            ("Sortino", f"{r.get('sortino_ratio',0):.4f}"),
            ("Calmar", f"{r.get('calmar_ratio',0):.4f}"),
            ("Profit Factor", f"{r.get('profit_factor',0):.3f}"),
            ("Total Trades", f"{r.get('total_trades',0)}"),
            ("Expectancy", f"${r.get('expectancy_per_trade',0):,.0f}"),
        ]
        for label, val in perf:
            html += f"<tr><td>{label}</td><td>{val}</td></tr>"

        html += "</table></body></html>"
        return html

    @staticmethod
    def save(results: Dict, filepath: str = "report.html",
             title: str = "NeuroTrade Report"):
        html = HTMLReportGenerator.generate(results, title)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  [HTML] Saved: {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED REPORT MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ReportManager:
    """Auto-detect available backend and generate best report format."""

    @staticmethod
    def generate_pdf(results: Dict, filepath: str = "tearsheet.pdf",
                     title: str = "NeuroTrade Tearsheet") -> Optional[str]:
        if _MPL:
            gen = TearsheetGenerator(results, title)
            gen.save(filepath)
            return filepath
        return None

    @staticmethod
    def generate_pdf_bytes(results: Dict,
                           title: str = "NeuroTrade Tearsheet") -> Optional[bytes]:
        if _MPL:
            gen = TearsheetGenerator(results, title)
            return gen.to_bytes()
        return None

    @staticmethod
    def generate_html(results: Dict, filepath: str = "report.html",
                      title: str = "NeuroTrade Report") -> str:
        HTMLReportGenerator.save(results, filepath, title)
        return filepath


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  reports.py — Tearsheet Generator Self-Test")
    print("=" * 65)

    # mock results
    from collections import namedtuple
    MockTrade = namedtuple("MockTrade", ["pnl", "pnl_pct", "exit_date", "side",
                                          "entry_price", "exit_price", "duration_bars",
                                          "exit_reason", "mae", "mfe"])

    np.random.seed(42)
    n_trades = 50
    mock_trades = [
        MockTrade(
            pnl=np.random.normal(200, 500),
            pnl_pct=np.random.normal(0.002, 0.01),
            exit_date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i*3),
            side="long" if np.random.random() > 0.4 else "short",
            entry_price=100 + np.random.normal(0, 5),
            exit_price=100 + np.random.normal(0.5, 5),
            duration_bars=np.random.randint(1, 30),
            exit_reason=np.random.choice(["signal", "stop_loss", "take_profit"]),
            mae=np.random.uniform(-0.05, 0),
            mfe=np.random.uniform(0, 0.08),
        ) for i in range(n_trades)
    ]

    equity = 100_000 + np.cumsum(np.random.normal(50, 300, 200))
    eq_df = pd.DataFrame({
        "equity": equity,
    }, index=pd.date_range("2024-01-01", periods=200, freq="B"))

    mock_results = {
        "initial_capital": 100_000,
        "final_capital": float(equity[-1]),
        "total_return_pct": (equity[-1] / 100_000 - 1) * 100,
        "annualized_return_pct": 12.5,
        "sharpe_ratio": 1.23,
        "sortino_ratio": 1.85,
        "calmar_ratio": 0.95,
        "omega_ratio": 1.45,
        "max_drawdown_pct": -8.5,
        "var_95_pct": -1.8,
        "cvar_95_pct": -2.5,
        "win_rate_pct": 58.0,
        "profit_factor": 1.65,
        "total_trades": n_trades,
        "kelly_criterion_pct": 15.2,
        "gross_profit": sum(t.pnl for t in mock_trades if t.pnl > 0),
        "gross_loss": sum(t.pnl for t in mock_trades if t.pnl < 0),
        "avg_win": 450,
        "avg_loss": -280,
        "largest_win": 1500,
        "largest_loss": -900,
        "expectancy_per_trade": 85,
        "avg_trade_duration_bars": 8,
        "long_trades": 30,
        "short_trades": 20,
        "exit_reasons": {"signal": 25, "stop_loss": 15, "take_profit": 10},
        "equity_curve": eq_df,
        "trades": mock_trades,
    }

    # Generate reports
    if _MPL:
        print("\n  Generating PDF tearsheet...")
        ReportManager.generate_pdf(mock_results, "tearsheet_test.pdf")
        pdf_bytes = ReportManager.generate_pdf_bytes(mock_results)
        print(f"  PDF bytes: {len(pdf_bytes):,}")
    else:
        print("  [SKIP] matplotlib not installed, skipping PDF")

    print("\n  Generating HTML report...")
    ReportManager.generate_html(mock_results, "report_test.html")

    print("\n  ✓  Reports self-test complete.\n")
