"""Counter-trend (retracement) entries on daily OHLC bars.

Long side: A = average high-low range over `window` days (as of the prior
close). Level L = previous day's HIGH - P*A. If the instrument trades down to
L during the day, buy at L (or the open, if it gapped through L) and exit at
the close. The short side is the mirror image.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sharpe


def countertrend(ohlc: pd.DataFrame, p: float = 2.2, window: int = 20, side: str = "long") -> pd.Series:
    """Daily strategy return; NaN on days with no signal (see `sharpe_of`)."""
    avg_range = (ohlc["high"] - ohlc["low"]).rolling(window).mean().shift(1)
    if side == "long":
        level = ohlc["high"].shift(1) - p * avg_range
        hit = ohlc["low"] <= level
        entry = np.minimum(ohlc["open"], level)
        ret = ohlc["close"] / entry - 1
    elif side == "short":
        level = ohlc["low"].shift(1) + p * avg_range
        hit = ohlc["high"] >= level
        entry = np.maximum(ohlc["open"], level)
        ret = entry / ohlc["close"] - 1
    else:
        raise ValueError("side must be 'long' or 'short'")
    return ret.where(hit)


def sharpe_of(returns: pd.Series, blanks: str = "zero") -> float:
    """Annualised Sharpe on signal days only ('blank') or with zeros filled in ('zero')."""
    return sharpe(returns.fillna(0.0) if blanks == "zero" else returns.dropna())


def sweep_p(ohlc: pd.DataFrame, ps=None, side: str = "long") -> pd.DataFrame:
    """Sharpe (both blank conventions) and signal count across retracement levels P."""
    ps = np.round(np.arange(2.2, 0.79, -0.2), 2) if ps is None else ps
    rows = []
    for p in ps:
        r = countertrend(ohlc, p, side=side)
        rows.append({"P": p, "signals": int(r.notna().sum()),
                     "Sharpe (zero)": sharpe_of(r, "zero"), "Sharpe (blank)": sharpe_of(r, "blank")})
    return pd.DataFrame(rows).set_index("P")


def market_vs_strategy(ohlc: pd.DataFrame, strat: pd.Series) -> pd.DataFrame:
    """2x2 of signal days by sign of market (close-to-close) vs strategy return,
    with the summed strategy return in each cell."""
    mkt = ohlc["close"].pct_change()
    df = pd.concat({"mkt": mkt, "strat": strat}, axis=1).dropna()
    rows = []
    for m_name, m_mask in (("mkt up", df.mkt > 0), ("mkt down", df.mkt <= 0)):
        for s_name, s_mask in (("strat up", df.strat > 0), ("strat down", df.strat <= 0)):
            sel = df[m_mask & s_mask]
            rows.append({"market": m_name, "strategy": s_name, "days": len(sel),
                         "total strat return": float(sel.strat.sum())})
    return pd.DataFrame(rows).set_index(["market", "strategy"])
