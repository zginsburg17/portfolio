"""Charts and tabular reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import drawdown, equity_curve
from .results import StudyResult


def summary_table(stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(stats).T


PCT_COLUMNS = {"CAGR", "Ann. Vol", "Max DD", "Hit Rate", "alpha (ann)"}


def _fmt(col: str, v: float, nd: int = 2) -> str:
    if pd.isna(v):
        return "-"
    if col in PCT_COLUMNS:
        return f"{v:.1%}"
    return f"{v:.0f}" if col == "rebalances" else f"{v:.{nd}f}"


def format_table(df: pd.DataFrame) -> str:
    out = df.copy().astype(object)
    for col in df.columns:
        out[col] = [_fmt(col, v) for v in df[col]]
    return out.to_string()


def markdown_table(df: pd.DataFrame, nd: int = 2) -> str:
    """GitHub-flavoured markdown table; percent-type columns formatted as percentages."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    rows = [[str(i)] + [_fmt(c, v, nd) if isinstance(v, (int, float)) else str(v)
                        for c, v in zip(df.columns, r)] for i, r in zip(df.index, df.to_numpy())]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def render(result: StudyResult) -> str:
    """Plain-text rendering of a study: a banner, then each section."""
    out = [f"\n{'=' * 8} {result.title} {'=' * 8}"]
    for sec in result.sections:
        body = sec.body if isinstance(sec.body, str) else sec.body.round(3).to_string()
        out.append(f"\n{sec.heading}\n{body}")
    return "\n".join(out)


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
