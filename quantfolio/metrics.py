"""Performance and risk statistics on a daily return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0.0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sharpes(returns: pd.DataFrame) -> pd.Series:
    """Annualised Sharpe of every column at once (0 where volatility is zero)."""
    sd = returns.std()
    return (returns.mean() / sd.where(sd > 0) * np.sqrt(TRADING_DAYS)).fillna(0.0)


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0


def cagr(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    return float(equity_curve(returns).iloc[-1] ** (TRADING_DAYS / n) - 1)


def performance_summary(returns: pd.Series, rf: float = 0.0) -> dict[str, float]:
    returns = returns.dropna()
    mdd = float(drawdown(returns).min()) if len(returns) else 0.0
    c = cagr(returns)
    return {
        "CAGR": c,
        "Ann. Vol": float(returns.std() * np.sqrt(TRADING_DAYS)),
        "Sharpe": sharpe(returns, rf),
        "Sortino": sortino(returns, rf),
        "Max DD": mdd,
        "Calmar": float(c / abs(mdd)) if mdd < 0 else 0.0,
        "Hit Rate": float((returns > 0).mean()),
    }
