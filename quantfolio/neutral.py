"""Sector- and beta-neutral cross-sectional equity long/short.

Pipeline (all signals use data through day t; the backtester trades them at t+1):

1. Score each stock: 12-1 momentum plus a short-term reversal term, each
   cross-sectionally z-scored.
2. Sector-neutralize: demean scores within each sector, so the portfolio bets on
   stock selection inside sectors, not on sectors themselves.
3. Beta-neutralize: shift weights along the sector-demeaned beta direction until
   ex-ante portfolio beta is zero. The shift sums to zero inside every sector, so
   sector neutrality is preserved exactly.
4. Scale to unit gross (0.5 long / 0.5 short) and rebalance every `rebalance` days.

Betas are rolling regressions on the universe's equal-weight return (or a supplied
market series such as SPY), estimated from past data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sector_matrix(columns: pd.Index, sectors: pd.Series) -> np.ndarray:
    """One-hot (names x sectors) membership matrix."""
    missing = [c for c in columns if c not in sectors.index]
    if missing:
        raise ValueError(f"no sector for: {missing}")
    codes = pd.Categorical(sectors.reindex(columns)).codes
    S = np.zeros((len(columns), codes.max() + 1))
    S[np.arange(len(columns)), codes] = 1.0
    return S


def _demean_by_sector(x: np.ndarray, S: np.ndarray, valid: np.ndarray, min_names: int):
    """Subtract each date's sector mean (over valid names). Sectors with fewer than
    `min_names` valid names on a date are dropped. Returns (demeaned, usable mask)."""
    count = valid.astype(float) @ S
    total = np.where(valid, x, 0.0) @ S
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    usable = valid & (((count >= min_names).astype(float) @ S.T) > 0)
    return np.where(usable, x - mean @ S.T, 0.0), usable


def rolling_betas(prices: pd.DataFrame, window: int = 252, market: pd.Series | None = None) -> pd.DataFrame:
    """Rolling beta of each stock to the market (default: equal-weight universe return)."""
    rets = prices.pct_change()
    m = rets.mean(axis=1) if market is None else market.reindex(rets.index)
    return rets.rolling(window).cov(m).div(m.rolling(window).var(), axis=0)


def _zscore_rows(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def neutral_weights(
    prices: pd.DataFrame,
    sectors: pd.Series,
    lookback: int = 252,
    skip: int = 21,
    reversal: int = 21,
    w_rev: float = 0.5,
    beta_window: int = 252,
    min_names: int = 3,
    beta_neutral: bool = True,
    rebalance: int = 21,
    market: pd.Series | None = None,
) -> pd.DataFrame:
    """Target weights (unit gross, zero net per sector, zero ex-ante beta if requested)."""
    mom = prices.shift(skip) / prices.shift(lookback) - 1
    rev = -prices.pct_change(reversal)
    score = _zscore_rows(mom) + w_rev * _zscore_rows(rev)
    beta = rolling_betas(prices, beta_window, market)

    S = _sector_matrix(prices.columns, sectors)
    valid = (score.notna() & beta.notna()).to_numpy()
    d, usable = _demean_by_sector(score.to_numpy(), S, valid, min_names)

    if beta_neutral:
        b0 = np.where(usable, beta.to_numpy(), 0.0)
        b, _ = _demean_by_sector(b0, S, usable, min_names)
        denom = (b * b0).sum(axis=1)
        k = np.divide((d * b0).sum(axis=1), denom, out=np.zeros(len(d)), where=denom > 1e-12)
        d = d - k[:, None] * b

    gross = np.abs(d).sum(axis=1, keepdims=True)
    w = pd.DataFrame(np.divide(d, gross, out=np.zeros_like(d), where=gross > 0),
                     index=prices.index, columns=prices.columns)

    return w.iloc[::rebalance].reindex(w.index).ffill().fillna(0.0)  # hold between rebalances


def exposure_check(weights: pd.DataFrame, prices: pd.DataFrame, sectors: pd.Series,
                   beta_window: int = 252, market: pd.Series | None = None) -> dict[str, float]:
    """Verify the constraints on the dates the portfolio was actually rebalanced."""
    changed = weights.diff().abs().sum(axis=1) > 0  # includes the first live row (prior row is flat)
    w = weights[changed & (weights.abs().sum(axis=1) > 0)]
    S = _sector_matrix(weights.columns, sectors)
    beta = rolling_betas(prices, beta_window, market).reindex(w.index).fillna(0.0)
    return {
        "rebalances": int(len(w)),
        "mean_gross": float(w.abs().sum(axis=1).mean()),
        "max_abs_net": float(w.sum(axis=1).abs().max()),
        "max_abs_sector_net": float(np.abs(w.to_numpy() @ S).max()),
        "max_abs_beta": float((w * beta).sum(axis=1).abs().max()),
    }
