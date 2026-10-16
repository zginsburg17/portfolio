"""Market data: reproducible synthetic prices, plus a CSV loader for real data."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def synthetic_prices(
    n_assets: int = 8,
    n_days: int = 252 * 8,
    seed: int = 42,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    """Simulate daily close prices with a shared market factor and regime switching.

    Each asset's return = beta * market + idiosyncratic noise + a slow-moving
    per-asset drift (AR(1)), so trend-following and mean-reversion strategies
    both have something to find without the data being trivially exploitable.
    Volatility alternates between calm and turbulent regimes.
    """
    rng = np.random.default_rng(seed)

    # Two-state volatility regime (Markov chain): calm <-> turbulent.
    p_stay = np.array([0.995, 0.97])
    vol_scale = np.array([0.7, 1.8])
    state = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        state[t] = state[t - 1] if rng.random() < p_stay[state[t - 1]] else 1 - state[t - 1]
    regime_vol = vol_scale[state]

    market = rng.normal(0.0003, 0.008, n_days) * regime_vol
    betas = rng.uniform(0.6, 1.4, n_assets)
    idio_vol = rng.uniform(0.008, 0.016, n_assets)

    # Slowly varying drift gives persistent (momentum-like) behaviour.
    drift = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        drift[t] = 0.98 * drift[t - 1] + rng.normal(0, 0.00025, n_assets)

    idio = rng.normal(0, 1, (n_days, n_assets)) * idio_vol * regime_vol[:, None]
    rets = market[:, None] * betas + idio + drift

    dates = pd.bdate_range(start, periods=n_days)
    cols = [f"ASSET{i + 1:02d}" for i in range(n_assets)]
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=cols)


def load_prices_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load a wide CSV (one column per ticker, one row per date) of close prices."""
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col).sort_index()
    return df.dropna(how="all").ffill()


def synthetic_ohlc(n_days: int = 252 * 10, seed: int = 7, start: str = "2010-01-01") -> pd.DataFrame:
    """One instrument with open/high/low/close, for intraday-range strategies."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n_days)))
    prev = np.r_[close[0], close[:-1]]
    open_ = prev * np.exp(rng.normal(0, 0.003, n_days))
    spread = 0.006
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, spread, n_days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, spread, n_days)))
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def synthetic_pair(n_days: int = 252 * 6, seed: int = 11, start: str = "2015-01-01") -> pd.DataFrame:
    """Two prices that share a stochastic trend; their price difference is an OU process."""
    rng = np.random.default_rng(seed)
    common = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_days)))
    spread = np.zeros(n_days)
    for t in range(1, n_days):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 1.0)
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame({"WHITE": common + spread, "BLACK": common}, index=idx)
