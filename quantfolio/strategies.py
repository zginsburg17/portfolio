"""Signal generators.

Every strategy maps a price DataFrame to a *target position* DataFrame of the
same shape (positive = long, negative = short, in units of gross exposure
fractions). Signals are computed using data up to and including day t; the
backtester is responsible for lagging them so they trade on day t+1.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest
from .metrics import TRADING_DAYS
from .portfolio import inverse_vol_weights
from .stats import zscore_trailing


def ma_crossover(
    prices: pd.DataFrame, fast: int, slow: int, kind: str = "sma"
) -> pd.DataFrame:
    """Per-instrument +1/-1 trend signal from a fast/slow moving-average crossover.

    kind="ema" uses exponential averages. Flat (0) until the slow average exists.
    Unscaled: each column is its own +/-1 system (use for single-instrument studies).
    """
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    if kind == "ema":
        f = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
        s = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()
    else:
        f, s = prices.rolling(fast).mean(), prices.rolling(slow).mean()
    sig = np.sign(f - s)
    sig[s.isna() | f.isna()] = 0.0
    return sig


def breakout(prices: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Channel breakout: long on a new `window`-day high, short on a new low, else hold."""
    hi = prices.shift(1).rolling(window).max()
    lo = prices.shift(1).rolling(window).min()
    raw = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    raw[prices > hi] = 1.0
    raw[prices < lo] = -1.0
    return raw.ffill().fillna(0.0)


def pairs_spread(
    prices: pd.DataFrame,
    lookback: int = 5,
    z_window: int = 60,
    entry: float = 1.0,
    target: float = 1.0,
    hold: int = 5,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Mean-reversion on the spread of two instruments' normalized N-day returns.

    zdiff = z(ret_A) - z(ret_B), each z-scored against `z_window` trailing values.
    zdiff > +entry  -> short the spread (short A, long B);
    zdiff < -entry  -> long the spread (long A, short B).
    Exit on the holding-period limit or when the profit target is hit
    (long spread: zdiff >= +target; short spread: zdiff <= -target).
    Legs are sized inversely to their trailing `vol_window`-day volatility.
    """
    if prices.shape[1] != 2:
        raise ValueError("pairs_spread needs exactly two price columns")
    a, b = prices.columns
    zret = zscore_trailing(prices.pct_change(lookback), z_window)
    zdiff = (zret[a] - zret[b]).to_numpy()

    pos = np.zeros(len(zdiff))
    cur, held = 0, 0
    for t, z in enumerate(zdiff):
        if cur != 0:
            held += 1
            hit = (cur == 1 and z >= target) or (cur == -1 and z <= -target)
            if held > hold or hit or np.isnan(z):
                cur = 0
        if cur == 0 and not np.isnan(z):
            if z > entry:
                cur, held = -1, 1
            elif z < -entry:
                cur, held = 1, 1
        pos[t] = cur

    w = inverse_vol_weights(prices.pct_change(), vol_window, lag=0).fillna(0.0)
    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    out[a] = pos * w[a]
    out[b] = -pos * w[b]
    return out


def time_series_momentum(
    prices: pd.DataFrame, lookback: int = 126, vol_window: int = 60, target_vol: float = 0.10
) -> pd.DataFrame:
    """Moskowitz-style TSMOM: sign of trailing return, sized to a per-asset vol target."""
    rets = prices.pct_change()
    direction = np.sign(prices.pct_change(lookback))
    ann_vol = rets.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    size = (target_vol / ann_vol).clip(upper=3.0)
    pos = (direction * size).fillna(0.0)
    return pos / prices.shape[1]


def bollinger_mean_reversion(
    prices: pd.DataFrame, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5
) -> pd.DataFrame:
    """Fade stretched moves: short above +entry_z, long below -entry_z, flat near the mean.

    Positions are held (state machine) until the z-score reverts inside exit_z.
    """
    z = zscore_trailing(prices, window, include_current=True)
    events = pd.DataFrame(np.nan, index=z.index, columns=z.columns)
    events[z > entry_z] = -1.0
    events[z < -entry_z] = 1.0
    events[z.abs() < exit_z] = 0.0  # NaN z never triggers: all three tests are False
    return events.ffill().fillna(0.0) / prices.shape[1]


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


def equal_weight(fn):
    """Wrap a per-instrument +/-1 signal so a universe sums to unit gross exposure."""

    def run(prices: pd.DataFrame, **kw) -> pd.DataFrame:
        return fn(prices, **kw) / prices.shape[1]

    return run


STRATEGIES = {
    "sma_crossover": partial(equal_weight(ma_crossover), fast=20, slow=100),
    "tsmom": time_series_momentum,
    "mean_reversion": bollinger_mean_reversion,
    "xs_momentum": cross_sectional_momentum,
}


def run_all(prices: pd.DataFrame, cost_bps: float = 5.0) -> dict[str, BacktestResult]:
    """Backtest every registered strategy on one price panel."""
    return {n: backtest(prices, fn(prices), cost_bps=cost_bps) for n, fn in STRATEGIES.items()}


# Trend systems from the TREND study, keyed by name, as per-instrument +/-1 signals.
TREND_SYSTEMS = {
    "MA10/30": lambda p: ma_crossover(p, 10, 30),
    "MA30/100": lambda p: ma_crossover(p, 30, 100),
    "MA80/160": lambda p: ma_crossover(p, 80, 160),
    "Brk30": lambda p: breakout(p, 30),
}
