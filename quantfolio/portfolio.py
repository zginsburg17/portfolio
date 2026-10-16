"""Combine strategy return streams into one portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS


def combine_equal(strategy_returns: pd.DataFrame) -> pd.Series:
    """Equal-weight the return streams (daily rebalanced)."""
    return strategy_returns.mean(axis=1)


def inverse_vol_weights(returns: pd.DataFrame, window: int = 20, lag: int = 1) -> pd.DataFrame:
    """Weights proportional to 1 / trailing volatility, summing to 1 across columns.

    lag=1 uses volatility through the previous bar only (no look-ahead for
    return streams); lag=0 uses volatility through today (for position sizing
    decided at today's close and traded tomorrow).
    """
    inv = 1.0 / returns.rolling(window).std().shift(lag).replace(0, np.nan)
    return inv.div(inv.sum(axis=1), axis=0)


def combine_inverse_vol(
    strategy_returns: pd.DataFrame, window: int = 60, target_vol: float | None = 0.10
) -> pd.Series:
    """Inverse-volatility weighting, rebalanced daily with a lag, scaled to target vol.

    Weights at day t use only returns through t-1, so there is no look-ahead.
    """
    w = inverse_vol_weights(strategy_returns, window)
    raw = (w * strategy_returns).sum(axis=1, min_count=1)
    if target_vol is None:
        return raw.dropna()

    port_vol = raw.rolling(window).std().shift(1) * np.sqrt(TRADING_DAYS)
    leverage = (target_vol / port_vol).clip(upper=3.0)
    return (raw * leverage).dropna()
