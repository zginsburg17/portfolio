"""Statistical building blocks shared by the strategies and studies."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Dickey-Fuller critical values (constant, no trend, large sample).
DF_CRITICAL = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def zscore_trailing(x: pd.Series | pd.DataFrame, n: int, include_current: bool = False):
    """Z-score of x against its trailing n observations.

    With include_current=False (default) the window is the n values *prior* to
    today, i.e. Z(i) = (X(i) - mean(X(i-1..i-n))) / std(X(i-1..i-n)); no
    look-ahead and today's value cannot dampen its own signal.
    """
    base = x if include_current else x.shift(1)
    mu = base.rolling(n).mean()
    sd = base.rolling(n).std()
    return (x - mu) / sd.replace(0, np.nan)


def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """OLS with intercept. Returns coefficients, t-stats, R^2 and n."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    yv = df["y"].to_numpy()
    A = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ beta
    dof = len(yv) - A.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    names = ["const", *df.drop(columns="y").columns]
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    return {
        "coef": dict(zip(names, beta)),
        "tstat": dict(zip(names, beta / se)),
        "r2": float(1 - resid @ resid / ss_tot) if ss_tot > 0 else 0.0,
        "n": int(len(yv)),
    }


def adf_test(series: pd.Series) -> dict:
    """Dickey-Fuller test: regress delta_t on level_{t-1} (with intercept).

    Null: unit root (non-stationary). A t-stat on the lag coefficient more
    negative than the critical value rejects the null, i.e. the series is
    stationary / mean-reverting. For pairs, pass the *difference* of the two
    price series: that is the cointegration check.
    """
    s = series.dropna()
    res = ols(s.diff().dropna(), s.shift(1).dropna().rename("lag").to_frame())
    t = res["tstat"]["lag"]
    return {
        "coef": res["coef"]["lag"],
        "tstat": t,
        "critical": DF_CRITICAL,
        "stationary_5pct": bool(t < DF_CRITICAL["5%"]),
        "n": res["n"],
    }


def quintile_table(x: pd.Series, ys: dict[str, pd.Series], q: int = 5) -> pd.DataFrame:
    """Sort by x into q equal buckets (1 = lowest) and average x and each y per bucket."""
    df = pd.concat({"x": x, **ys}, axis=1).dropna()
    df["bucket"] = pd.qcut(df["x"].rank(method="first"), q, labels=range(1, q + 1))
    out = df.groupby("bucket", observed=True).mean()
    out["count"] = df.groupby("bucket", observed=True).size()
    return out
