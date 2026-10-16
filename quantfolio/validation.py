"""Is the backtest evidence of an edge, or of luck?

Tools to discount an in-sample Sharpe for sampling noise (bootstrap CI,
probabilistic Sharpe), for selection (deflated Sharpe, walk-forward), and for
fragility (parameter sensitivity).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import backtest
from .metrics import TRADING_DAYS, sharpe

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------- sampling noise
def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, block: int = 20,
                        alpha: float = 0.05, seed: int = 0) -> dict:
    """Circular block-bootstrap confidence interval for the annualised Sharpe.

    Blocks preserve short-range autocorrelation and volatility clustering that
    an i.i.d. bootstrap would destroy.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 2 * block:
        raise ValueError("series too short for the chosen block length")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    samples = r[idx]
    sd = samples.std(axis=1, ddof=1)
    sr = np.where(sd > 0, samples.mean(axis=1) / np.where(sd > 0, sd, 1), 0.0) * np.sqrt(TRADING_DAYS)
    lo, hi = np.quantile(sr, [alpha / 2, 1 - alpha / 2])
    return {"sharpe": sharpe(returns.dropna()), "lo": float(lo), "hi": float(hi),
            "p_le_zero": float((sr <= 0).mean()), "n_days": n}


def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark) given sample length, skew and kurtosis
    (Bailey & Lopez de Prado). `benchmark_sr` is annualised."""
    r = returns.dropna()
    n = len(r)
    sr = r.mean() / r.std()  # per-period
    skew, kurt = r.skew(), r.kurt() + 3.0
    bench = benchmark_sr / np.sqrt(TRADING_DAYS)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(_N.cdf((sr - bench) * np.sqrt(n - 1) / denom))


# ---------------------------------------------------------------- selection bias
def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Annualised Sharpe you'd expect the *best of n_trials* to show by luck alone,
    given the cross-trial variance of annualised Sharpes."""
    if n_trials < 2:
        return 0.0
    sd = np.sqrt(sr_variance)
    return float(sd * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                       + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_variance: float) -> float:
    """Probability the strategy's true Sharpe is positive *after* accounting for having
    picked it as the best of `n_trials` variants (Bailey & Lopez de Prado)."""
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sr_variance))


# ---------------------------------------------------------------- walk-forward
Factory = Callable[..., pd.DataFrame]


def _combos(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def trial_returns(prices: pd.DataFrame, factory: Factory, grid: dict[str, list],
                  cost_bps: float = 5.0) -> dict[tuple, pd.Series]:
    """Net return stream for every parameter combination, keyed by the params tuple."""
    out = {}
    for p in _combos(grid):
        pos = factory(prices, **p)
        out[tuple(p.items())] = backtest(prices, pos, cost_bps=cost_bps).returns
    return out


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    folds: pd.DataFrame  # one row per fold: chosen params, in-sample vs out-of-sample Sharpe
    trials: pd.DataFrame  # every combo's full-sample returns (for DSR variance)

    @property
    def is_sharpe(self) -> float:
        return float(self.folds["is_sharpe"].mean())

    @property
    def oos_sharpe(self) -> float:
        return sharpe(self.oos_returns)


def walk_forward(prices: pd.DataFrame, factory: Factory, grid: dict[str, list],
                 train_days: int = 756, test_days: int = 126, anchored: bool = False,
                 cost_bps: float = 5.0) -> WalkForwardResult:
    """Pick parameters on a training window, trade them on the next unseen window, roll.

    Strategies here are causal (signals use only past data), so each combo's
    return stream can be computed once on the full history and sliced per fold;
    that is equivalent to re-running the backtest per window, just faster.
    """
    trials = pd.DataFrame({k: v for k, v in trial_returns(prices, factory, grid, cost_bps).items()})
    n = len(prices)
    if train_days + test_days > n:
        raise ValueError("not enough history for one train+test fold")
    rows, oos = [], []
    start = train_days
    while start + test_days <= n:
        lo = 0 if anchored else start - train_days
        train = trials.iloc[lo:start]
        scores = train.apply(sharpe)
        best = scores.idxmax()
        test = trials.iloc[start:start + test_days][best]
        rows.append({"test_start": prices.index[start], "params": dict(best),
                     "is_sharpe": float(scores[best]), "oos_sharpe": sharpe(test)})
        oos.append(test)
        start += test_days
    return WalkForwardResult(pd.concat(oos), pd.DataFrame(rows), trials)


# ---------------------------------------------------------------- fragility
def sensitivity_grid(prices: pd.DataFrame, factory: Factory, x: tuple[str, list],
                     y: tuple[str, list], fixed: dict | None = None,
                     cost_bps: float = 5.0) -> pd.DataFrame:
    """Sharpe over a 2-D parameter grid (rows = y, columns = x). A robust edge shows
    a broad plateau; a single hot cell surrounded by cold ones is overfit."""
    (xn, xs), (yn, ys) = x, y
    grid = pd.DataFrame(index=pd.Index(ys, name=yn), columns=pd.Index(xs, name=xn), dtype=float)
    for yv in ys:
        for xv in xs:
            try:
                pos = factory(prices, **{**(fixed or {}), xn: xv, yn: yv})
            except ValueError:  # e.g. fast >= slow
                continue
            grid.loc[yv, xv] = sharpe(backtest(prices, pos, cost_bps=cost_bps).returns)
    return grid
