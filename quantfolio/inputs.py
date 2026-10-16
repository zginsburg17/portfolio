"""Resolve the data each study needs from one place, so no study silently uses the wrong data.

`Config` says where data comes from; `Inputs` lazily loads (and memoizes) each
kind of input. Asking for something a source cannot provide, such as OHLC bars
from a close-only CSV, raises DataSourceError instead of substituting synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from .backtest import BacktestResult
from .data import load_prices_csv, synthetic_ohlc, synthetic_pair, synthetic_prices
from .datasources import (LARGE_CAP, SECTOR_OF, DataSourceError, clean_panel, fetch_bars,
                          fetch_factors, fetch_prices)
from .strategies import run_all


@dataclass(frozen=True)
class Config:
    source: str = "synthetic"  # "synthetic" | "yahoo"
    csv: str | None = None  # wide close-price CSV; takes precedence over `source`
    tickers: tuple[str, ...] | None = None  # default: LARGE_CAP
    pair: tuple[str, str] = ("KO", "PEP")
    ohlc_ticker: str = "SPY"
    start: str = "2010-01-01"
    end: str | None = None
    seed: int = 42
    cost_bps: float = 5.0
    refresh: bool = False
    offline: bool = False

    @property
    def live(self) -> bool:
        """True when data comes from public providers (not a CSV, not synthetic)."""
        return self.csv is None and self.source == "yahoo"

    @property
    def _fetch_kw(self) -> dict:
        return {"refresh": self.refresh, "offline": self.offline}


class Inputs:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @cached_property
    def prices(self) -> pd.DataFrame:
        c = self.cfg
        if c.csv:
            return load_prices_csv(c.csv)
        if c.live:
            return clean_panel(fetch_prices(list(c.tickers or LARGE_CAP), c.start, c.end,
                                            **c._fetch_kw))
        return synthetic_prices(seed=c.seed)

    @cached_property
    def sectors(self) -> pd.Series:
        """Sector of every column in `prices`. Synthetic data gets arbitrary sectors of two."""
        cols = list(self.prices.columns)
        if not (self.cfg.live or self.cfg.csv):
            return pd.Series({c: f"S{i // 2}" for i, c in enumerate(cols)})
        missing = [c for c in cols if c not in SECTOR_OF]
        if missing:
            raise DataSourceError(f"no sector mapping for {missing}; the neutral study needs "
                                  "tickers from the built-in universe (datasources/universe.py)")
        return pd.Series({c: SECTOR_OF[c] for c in cols})

    @cached_property
    def pair(self) -> pd.DataFrame:
        c = self.cfg
        if c.live:
            return clean_panel(fetch_prices(list(c.pair), c.start, c.end, **c._fetch_kw))
        if c.csv:
            missing = [t for t in c.pair if t not in self.prices]
            if missing:
                raise DataSourceError(f"pair columns {missing} not found in {c.csv}; set --pair A,B")
            return self.prices[list(c.pair)]
        return synthetic_pair()

    @cached_property
    def ohlc(self) -> pd.DataFrame:
        """Daily OHLC, with Yahoo bars rescaled by adj_close/close so splits create no fake gaps."""
        c = self.cfg
        if c.live:
            bars = fetch_bars(c.ohlc_ticker, c.start, c.end, **c._fetch_kw)
            f = bars["adj_close"] / bars["close"]
            out = bars[["open", "high", "low"]].mul(f, axis=0)
            out["close"] = bars["adj_close"]
            return out.dropna()
        if c.csv:
            raise DataSourceError("the countertrend study needs OHLC bars; a close-only CSV "
                                  "cannot provide them (use --source yahoo)")
        return synthetic_ohlc()

    @cached_property
    def backtests(self) -> dict[str, BacktestResult]:
        return run_all(self.prices, self.cfg.cost_bps)

    @cached_property
    def strategy_returns(self) -> pd.DataFrame:
        return pd.DataFrame({n: r.returns for n, r in self.backtests.items()})

    @cached_property
    def factors(self) -> tuple[pd.DataFrame, str | None, str]:
        """(factor returns, risk-free column or None, provenance note)."""
        c = self.cfg
        if c.live:
            ff = fetch_factors("ff5", c.start, c.end, **c._fetch_kw).join(
                fetch_factors("mom", c.start, c.end, **c._fetch_kw), how="inner")
            return ff, "RF", "Fama-French 5 factors + momentum"
        proxy = pd.DataFrame({"MKT": self.prices.pct_change().mean(axis=1)})
        return proxy, None, ("proxy: equal-weight return of the loaded universe is the only "
                             "'market' factor; use --source yahoo for Fama-French factors")
