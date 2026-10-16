"""Combine strategy return streams into one portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd


def combine_inverse_vol(
    strategy_returns: pd.DataFrame, window: int = 60, target_vol: float = 0.10
) -> pd.Series:
    """Inverse-volatility weighting, rebalanced daily with a lag, scaled to target vol.

    Weights at day t use only returns through t-1, so there is no look-ahead.
    """
    vol = strategy_returns.rolling(window).std().shift(1)
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    raw = (w * strategy_returns).sum(axis=1, min_count=1)

    port_vol = raw.rolling(window).std().shift(1) * np.sqrt(252)
    leverage = (target_vol / port_vol).clip(upper=3.0)
    return (raw * leverage).dropna()


def correlation_matrix(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    return strategy_returns.corr()
