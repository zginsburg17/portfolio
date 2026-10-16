"""On-disk CSV cache so research reruns are fast, reproducible, and survive outages."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .http import DataSourceError, log


def cache_dir() -> Path:
    d = Path(os.environ.get("QUANTFOLIO_CACHE", Path.home() / ".cache" / "quantfolio"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached(key: str, producer: Callable[[], pd.DataFrame], ttl_hours: float = 24.0,
           refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    """Return the cached frame for `key` if fresh, else call `producer` and store it.

    If the producer fails but a stale copy exists, the stale copy is returned
    with a warning, so a provider outage doesn't kill a research session.
    `offline=True` never calls the producer.
    """
    path = cache_dir() / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".csv")
    have = path.exists()
    fresh = have and (time.time() - path.stat().st_mtime) < ttl_hours * 3600

    def read() -> pd.DataFrame:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    if offline:
        if not have:
            raise DataSourceError(f"offline mode and nothing cached for {key!r}")
        return read()
    if fresh and not refresh:
        return read()
    try:
        df = producer()
    except DataSourceError as e:
        if have:
            log.warning("%s; using stale cache for %s", e, key)
            return read()
        raise
    df.to_csv(path)
    return df
