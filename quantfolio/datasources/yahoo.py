"""Yahoo Finance daily bars via the public chart endpoint (no API key).

This is an unofficial endpoint: fine for research prototypes, but pin your
results to the cache (or export a CSV) so they stay reproducible if it changes.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

import pandas as pd

from .cache import cached
from .http import DataSourceError, Transport, fetch

log = logging.getLogger("quantfolio.data")
BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def _epoch(d: str | date | None, default: int) -> int:
    if d is None:
        return default
    ts = pd.Timestamp(d)
    return int(datetime(ts.year, ts.month, ts.day, tzinfo=timezone.utc).timestamp())


def parse_chart(payload: bytes, symbol: str) -> pd.DataFrame:
    """Parse a chart-API JSON body into open/high/low/close/adj_close/volume."""
    data = json.loads(payload)
    chart = data.get("chart", {})
    if chart.get("error"):
        raise DataSourceError(f"yahoo error for {symbol}: {chart['error'].get('description')}")
    res = (chart.get("result") or [None])[0]
    if not res or "timestamp" not in res:
        raise DataSourceError(f"yahoo returned no data for {symbol}")
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose", q["close"])
    tz = res.get("meta", {}).get("exchangeTimeZoneName", "UTC")
    idx = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
           .tz_convert(tz).tz_localize(None).normalize())
    df = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
         "adj_close": adj, "volume": q["volume"]}, index=idx)
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"]).sort_index()
    df.index.name = "date"
    return df


def fetch_bars(symbol: str, start=None, end=None, transport: Transport | None = None,
               refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    """Daily OHLCV for one symbol. `adj_close` is split- and dividend-adjusted."""
    p1, p2 = _epoch(start, 0), _epoch(end, int(datetime.now(timezone.utc).timestamp()))

    def produce() -> pd.DataFrame:
        body = fetch(BASE.format(symbol=symbol),
                     {"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"},
                     transport)
        return parse_chart(body, symbol)

    return cached(f"yahoo|{symbol}|{start}|{end}", produce, refresh=refresh, offline=offline)


def fetch_prices(symbols: list[str], start=None, end=None, field: str = "adj_close",
                 transport: Transport | None = None, refresh: bool = False,
                 offline: bool = False) -> pd.DataFrame:
    """Wide frame of one price field (default adjusted close), one column per symbol.

    Symbols that fail are skipped with a warning; if all fail, raises.
    """
    cols, failed = {}, []
    for s in symbols:
        try:
            cols[s] = fetch_bars(s, start, end, transport, refresh, offline)[field]
        except DataSourceError as e:
            failed.append(s)
            log.warning("skipping %s: %s", s, e)
    if not cols:
        raise DataSourceError(f"no symbols could be fetched (failed: {failed})")
    return pd.DataFrame(cols).sort_index()
