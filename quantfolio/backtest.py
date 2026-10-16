"""Vectorized backtester with transaction costs and no look-ahead."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import performance_summary


@dataclass
class BacktestResult:
    returns: pd.Series  # net daily strategy returns
    gross_returns: pd.Series
    positions: pd.DataFrame  # positions actually held each day (post-lag)
    turnover: pd.Series  # daily sum of |change in position|
    costs: pd.Series

    def summary(self) -> dict[str, float]:
        stats = performance_summary(self.returns)
        stats["Avg Daily Turnover"] = float(self.turnover.mean())
        return stats


def backtest(
    prices: pd.DataFrame,
    target_positions: pd.DataFrame,
    cost_bps: float = 5.0,
    max_gross: float | None = 2.0,
) -> BacktestResult:
    """Run a backtest of target positions against close-to-close asset returns.

    Signals are shifted one day: a position decided on the close of day t earns
    the return of day t+1. Costs are charged as `cost_bps` per unit of turnover
    (covers commission + slippage + half-spread, one-way).
    """
    target = target_positions.reindex_like(prices).fillna(0.0)
    if max_gross is not None:
        gross = target.abs().sum(axis=1)
        scale = (max_gross / gross).clip(upper=1.0).fillna(1.0)
        target = target.mul(scale, axis=0)

    held = target.shift(1).fillna(0.0)  # the one-day lag that prevents look-ahead
    asset_rets = prices.pct_change().fillna(0.0)

    gross_ret = (held * asset_rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    costs = turnover * cost_bps / 1e4
    return BacktestResult(
        returns=gross_ret - costs,
        gross_returns=gross_ret,
        positions=held,
        turnover=turnover,
        costs=costs,
    )
