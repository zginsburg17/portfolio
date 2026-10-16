"""Charts and tabular reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import drawdown, equity_curve


def summary_table(stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(stats).T


def format_table(df: pd.DataFrame) -> str:
    pct = {"CAGR", "Ann. Vol", "Max DD", "Hit Rate"}
    out = df.copy().astype(object)
    for col in df.columns:
        out[col] = [
            "-" if pd.isna(v) else (f"{v:.1%}" if col in pct else f"{v:.2f}") for v in df[col]
        ]
    return out.to_string()


def plot_report(
    strategy_returns: pd.DataFrame, portfolio: pd.Series, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[3, 1])

    for col in strategy_returns:
        axes[0].plot(equity_curve(strategy_returns[col]), lw=1, alpha=0.6, label=col)
    axes[0].plot(equity_curve(portfolio), lw=2.2, color="black", label="PORTFOLIO")
    axes[0].set_yscale("log")
    axes[0].set_title("Equity curves (net of costs, log scale)")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(portfolio.index, drawdown(portfolio), 0, color="crimson", alpha=0.5)
    axes[1].set_title("Portfolio drawdown")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
