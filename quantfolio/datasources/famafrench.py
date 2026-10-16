"""Fama-French daily factor returns (Ken French data library) for factor attribution."""

from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from .cache import cached
from .http import DataSourceError, Transport, fetch

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_daily_CSV.zip"
DATASETS = {
    "ff3": "F-F_Research_Data_Factors",        # Mkt-RF, SMB, HML, RF
    "ff5": "F-F_Research_Data_5_Factors_2x3",  # + RMW, CMA
    "mom": "F-F_Momentum_Factor",              # Mom (UMD)
}
_ROW = re.compile(r"^\s*(\d{8})\s*,")


def parse_french_zip(payload: bytes) -> pd.DataFrame:
    """Parse the zipped CSV: free-text preamble, header row, data rows, then footer text.

    Values are published in percent; returned as decimal returns.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    first = next((i for i, ln in enumerate(lines) if _ROW.match(ln)), None)
    if first is None:
        raise DataSourceError("no data rows found in Fama-French file")
    header = lines[first - 1].split(",")
    cols = [c.strip() for c in header[1:]]
    rows = []
    for ln in lines[first:]:
        if not _ROW.match(ln):
            break
        parts = ln.split(",")
        rows.append([parts[0].strip(), *[float(x) for x in parts[1:]]])
    df = pd.DataFrame(rows, columns=["date", *cols])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date") / 100.0


def fetch_factors(kind: str = "ff3", start=None, end=None, transport: Transport | None = None,
                  refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    if kind not in DATASETS:
        raise ValueError(f"kind must be one of {sorted(DATASETS)}")

    def produce() -> pd.DataFrame:
        return parse_french_zip(fetch(BASE.format(name=DATASETS[kind]), transport=transport))

    df = cached(f"french|{kind}", produce, ttl_hours=24 * 7, refresh=refresh, offline=offline)
    return df.loc[start:end]
