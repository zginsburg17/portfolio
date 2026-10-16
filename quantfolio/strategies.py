"""Signal generators.

Every strategy maps a price DataFrame to a *target position* DataFrame of the
same shape (positive = long, negative = short, in units of gross exposure
fractions). Signals are computed using data up to and including day t; the
backtester is responsible for lagging them so they trade on day t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma_crossover(prices: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    """Trend following: long when the fast SMA is above the slow SMA, else short."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    sig = np.sign(f - s)
    sig[s.isna()] = 0.0
    return sig / prices.shape[1]


def time_series_momentum(
    prices: pd.DataFrame, lookback: int = 126, vol_window: int = 60, target_vol: float = 0.10
) -> pd.DataFrame:
    """Moskowitz-style TSMOM: sign of trailing return, sized to a per-asset vol target."""
    rets = prices.pct_change()
    direction = np.sign(prices.pct_change(lookback))
    ann_vol = rets.rolling(vol_window).std() * np.sqrt(252)
    size = (target_vol / ann_vol).clip(upper=3.0)
    pos = (direction * size).fillna(0.0)
    return pos / prices.shape[1]


def bollinger_mean_reversion(
    prices: pd.DataFrame, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5
) -> pd.DataFrame:
    """Fade stretched moves: short above +entry_z, long below -entry_z, flat near the mean.

    Positions are held (state machine) until the z-score reverts inside exit_z.
    """
    z = (prices - prices.rolling(window).mean()) / prices.rolling(window).std()
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    cur = np.zeros(prices.shape[1])
    zv = z.to_numpy()
    out = np.zeros_like(zv)
    for t in range(len(zv)):
        row = zv[t]
        valid = ~np.isnan(row)
        cur = np.where(valid & (row > entry_z), -1.0, cur)
        cur = np.where(valid & (row < -entry_z), 1.0, cur)
        cur = np.where(valid & (np.abs(row) < exit_z), 0.0, cur)
        out[t] = cur
    pos[:] = out
    return pos / prices.shape[1]


def cross_sectional_momentum(
    prices: pd.DataFrame, lookback: int = 126, skip: int = 5, quantile: float = 0.25
) -> pd.DataFrame:
    """Dollar-neutral: long top-quantile, short bottom-quantile by trailing return.

    The most recent `skip` days are excluded to sidestep short-term reversal.
    """
    score = prices.shift(skip).pct_change(lookback - skip)
    ranks = score.rank(axis=1, pct=True)
    long = (ranks >= 1 - quantile).astype(float)
    short = (ranks <= quantile).astype(float)
    long = long.div(long.sum(axis=1).replace(0, np.nan), axis=0)
    short = short.div(short.sum(axis=1).replace(0, np.nan), axis=0)
    return (long - short).fillna(0.0) * 0.5  # 0.5 long + 0.5 short = 1.0 gross


STRATEGIES = {
    "sma_crossover": sma_crossover,
    "tsmom": time_series_momentum,
    "mean_reversion": bollinger_mean_reversion,
    "xs_momentum": cross_sectional_momentum,
}
