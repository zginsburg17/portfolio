"""FRED macro series via the public graph CSV endpoint (no API key needed).

Handy for risk-free rates (DGS3MO), term structure (DGS10, T10Y2Y), VIX (VIXCLS),
credit spreads (BAMLH0A0HYM2), and so on.
"""

from __future__ import annotations

import io

import pandas as pd

from .cache import cached
from .http import DataSourceError, Transport, fetch

BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def parse_fred(payload: bytes, series_id: str) -> pd.Series:
    df = pd.read_csv(io.BytesIO(payload))
    if df.shape[1] < 2:
        raise DataSourceError(f"unexpected FRED response for {series_id}")
    date_col = df.columns[0]  # 'DATE' historically, 'observation_date' now
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # missing values are '.'
    s.index = pd.to_datetime(df[date_col])
    s.index.name = "date"
    return s.dropna().rename(series_id)


def fetch_series(series_ids: list[str] | str, start=None, end=None,
                 transport: Transport | None = None, refresh: bool = False,
                 offline: bool = False) -> pd.DataFrame:
    ids = [series_ids] if isinstance(series_ids, str) else list(series_ids)
    cols = {}
    for sid in ids:
        def produce(sid=sid) -> pd.DataFrame:
            params = {"id": sid}
            if start:
                params["cosd"] = str(pd.Timestamp(start).date())
            if end:
                params["coed"] = str(pd.Timestamp(end).date())
            return parse_fred(fetch(BASE, params, transport), sid).to_frame()

        cols[sid] = cached(f"fred|{sid}|{start}|{end}", produce, refresh=refresh,
                           offline=offline).iloc[:, 0]
    return pd.DataFrame(cols).sort_index()
