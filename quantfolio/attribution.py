"""Where do the returns come from? Alpha vs factor exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS
from .stats import ols_hac


def factor_regression(returns: pd.Series, factors: pd.DataFrame, rf_col: str | None = "RF",
                      lags: int | None = None) -> dict:
    """Regress excess strategy returns on factor returns (Newey-West t-stats).

    `factors` may contain an `rf_col` risk-free column, which is subtracted from the
    strategy and dropped from the regressors. Alpha is annualised (daily * 252).
    A strategy that's really just beta shows a high R^2 on the market and a
    near-zero or insignificant alpha.
    """
    f = factors.copy()
    y = returns.copy()
    if rf_col and rf_col in f:
        y = y - f[rf_col].reindex(y.index)
        f = f.drop(columns=rf_col)
    fit = ols_hac(y, f, lags)
    return {
        "alpha_ann": fit["coef"]["const"] * TRADING_DAYS,
        "alpha_t": fit["tstat"]["const"],
        "betas": {k: v for k, v in fit["coef"].items() if k != "const"},
        "beta_t": {k: v for k, v in fit["tstat"].items() if k != "const"},
        "r2": fit["r2"],
        "n": fit["n"],
    }


def beta_hedge(returns: pd.Series, market: pd.Series, window: int = 126) -> pd.Series:
    """Remove market exposure using a rolling beta estimated on *prior* data only."""
    beta = (returns.rolling(window).cov(market) / market.rolling(window).var()).shift(1)
    return (returns - beta * market).dropna()


def attribution_table(strategy_returns: pd.DataFrame, factors: pd.DataFrame,
                      rf_col: str | None = "RF") -> pd.DataFrame:
    rows = {}
    for c in strategy_returns:
        res = factor_regression(strategy_returns[c], factors, rf_col)
        rows[c] = {"alpha (ann)": res["alpha_ann"], "alpha t": res["alpha_t"], "R2": res["r2"],
                   **{f"beta {k}": v for k, v in res["betas"].items()}}
    return pd.DataFrame(rows).T
