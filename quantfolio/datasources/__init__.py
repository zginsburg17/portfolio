"""Public data sources: Yahoo (prices), FRED (macro), Ken French (factors)."""

from __future__ import annotations

import pandas as pd

from .famafrench import fetch_factors
from .fred import fetch_series
from .http import DataSourceError
from .universe import BENCHMARKS, LARGE_CAP, SECTOR_OF, SECTORS
from .yahoo import fetch_bars, fetch_prices

__all__ = ["fetch_prices", "fetch_bars", "fetch_series", "fetch_factors", "clean_panel",
           "DataSourceError", "LARGE_CAP", "SECTORS", "SECTOR_OF", "BENCHMARKS"]


def clean_panel(prices: pd.DataFrame, min_history: float = 0.9, max_ffill: int = 3) -> pd.DataFrame:
    """Make a raw price panel strategy-ready.

    - drops tickers with less than `min_history` fraction of non-missing days
    - forward-fills short gaps (<= max_ffill days, e.g. holidays on one venue)
    - trims leading rows until every remaining ticker has a price
    """
    keep = prices.columns[prices.notna().mean() >= min_history]
    out = prices[keep].ffill(limit=max_ffill)
    first = out.apply(lambda c: c.first_valid_index()).max()
    return out.loc[first:].dropna(how="any")
