"""Provider tests run against canned payloads in each provider's documented format.

They verify parsing, caching and failure handling offline. They do NOT prove the
live endpoints still return this shape; run `python -m quantfolio --source yahoo`
on a machine with network access for that.
"""

import io
import json
import urllib.error
import zipfile

import pandas as pd
import pytest

from quantfolio.datasources import DataSourceError, clean_panel, fetch_factors, fetch_prices, fetch_series
from quantfolio.datasources.famafrench import parse_french_zip
from quantfolio.datasources.fred import parse_fred
from quantfolio.datasources.http import fetch
from quantfolio.datasources.universe import LARGE_CAP, SECTOR_OF
from quantfolio.datasources.yahoo import parse_chart


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTFOLIO_CACHE", str(tmp_path))


def chart_body(closes, adj=None, symbol="AAA"):
    ts = [1_700_000_000 + 86_400 * i for i in range(len(closes))]
    return json.dumps({"chart": {"error": None, "result": [{
        "meta": {"symbol": symbol, "exchangeTimeZoneName": "America/New_York"},
        "timestamp": ts,
        "indicators": {
            "quote": [{"open": closes, "high": closes, "low": closes, "close": closes,
                       "volume": [1000] * len(closes)}],
            "adjclose": [{"adjclose": adj or closes}],
        }}]}}).encode()


class Counting:
    def __init__(self, body):
        self.body, self.calls = body, 0

    def __call__(self, url, headers):
        self.calls += 1
        return self.body(url) if callable(self.body) else self.body


def test_parse_chart_uses_adjusted_close_and_drops_null_bars():
    body = json.loads(chart_body([10.0, None, 12.0], adj=[5.0, None, 6.0]))
    df = parse_chart(json.dumps(body).encode(), "AAA")
    assert list(df["adj_close"]) == [5.0, 6.0] and list(df["close"]) == [10.0, 12.0]
    assert df.index.is_monotonic_increasing


def test_parse_chart_error_payload():
    bad = json.dumps({"chart": {"result": None, "error": {"description": "Not Found"}}}).encode()
    with pytest.raises(DataSourceError, match="Not Found"):
        parse_chart(bad, "ZZZ")


def test_fetch_prices_caches_and_skips_failures():
    def body(url):
        if "BAD" in url:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        return chart_body([100.0, 101.0, 102.0])

    t = Counting(body)
    px = fetch_prices(["AAA", "BAD"], "2023-01-01", transport=t)
    assert list(px.columns) == ["AAA"] and len(px) == 3
    n = t.calls
    fetch_prices(["AAA"], "2023-01-01", transport=t)  # served from cache
    assert t.calls == n
    with pytest.raises(DataSourceError):
        fetch_prices(["BAD"], "2023-01-01", transport=t)


def test_stale_cache_survives_outage_and_offline_mode():
    ok = Counting(chart_body([1.0, 2.0, 3.0]))
    fetch_prices(["AAA"], "2023-01-01", transport=ok)

    def down(url, headers):
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

    again = fetch_prices(["AAA"], "2023-01-01", transport=down, refresh=True)
    assert len(again) == 3
    assert len(fetch_prices(["AAA"], "2023-01-01", offline=True)) == 3
    with pytest.raises(DataSourceError):
        fetch_prices(["NEVER"], "2023-01-01", offline=True)


def test_http_does_not_retry_policy_denial_but_retries_5xx():
    calls = []

    def denied(url, headers):
        calls.append(1)
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

    with pytest.raises(DataSourceError, match="example.com"):
        fetch("https://example.com/x", transport=denied, backoff=0)
    assert len(calls) == 1

    flaky = []

    def sometimes(url, headers):
        flaky.append(1)
        if len(flaky) < 3:
            raise urllib.error.HTTPError(url, 503, "busy", {}, None)
        return b"ok"

    assert fetch("https://example.com/x", transport=sometimes, backoff=0) == b"ok"


def test_parse_fred_handles_missing_dots_and_new_header():
    csv = b"observation_date,DGS10\n2024-01-01,.\n2024-01-02,4.05\n2024-01-03,4.10\n"
    s = parse_fred(csv, "DGS10")
    assert list(s) == [4.05, 4.10] and s.name == "DGS10"
    old = parse_fred(b"DATE,VIXCLS\n2024-01-02,13.1\n", "VIXCLS")
    assert old.iloc[0] == 13.1


def test_fetch_series_passes_dates_and_merges():
    seen = []

    def body(url):
        seen.append(url)
        sid = url.split("id=")[1].split("&")[0]
        return f"observation_date,{sid}\n2024-01-02,1.0\n2024-01-03,2.0\n".encode()

    df = fetch_series(["DGS10", "DGS2"], "2024-01-01", transport=Counting(body))
    assert list(df.columns) == ["DGS10", "DGS2"] and "cosd=2024-01-01" in seen[0]


def french_zip(with_footer=True):
    txt = ("This file was created by CMPT_ME_BEME_RETS_DAILY.\n\n"
           ",Mkt-RF,SMB,HML,RF\n"
           "20240102,  -0.50,  0.10, -0.20, 0.02\n"
           "20240103,   1.00, -0.30,  0.40, 0.02\n\n"
           "Copyright 2024 Kenneth R. French\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.CSV", txt)
    return buf.getvalue()


def test_parse_french_zip_converts_percent_and_stops_at_footer():
    df = parse_french_zip(french_zip())
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"] and len(df) == 2
    assert df.loc["2024-01-03", "Mkt-RF"] == pytest.approx(0.01)


def test_fetch_factors_validates_kind_and_slices():
    t = Counting(french_zip())
    assert len(fetch_factors("ff3", start="2024-01-03", transport=t)) == 1
    with pytest.raises(ValueError):
        fetch_factors("bogus")


def test_clean_panel_drops_sparse_names_and_fills_short_gaps():
    idx = pd.bdate_range("2024-01-01", periods=20)
    px = pd.DataFrame({"A": 100.0, "B": 50.0, "SPARSE": float("nan")}, index=idx)
    px.loc[idx[5], "A"] = float("nan")  # one-day gap gets forward-filled
    px.loc[idx[:3], "B"] = float("nan")  # late start gets trimmed
    out = clean_panel(px, min_history=0.8)
    assert list(out.columns) == ["A", "B"]
    assert out.notna().all().all() and out.index[0] == idx[3]


def test_universe_is_consistent():
    assert len(LARGE_CAP) == len(set(LARGE_CAP)) and set(LARGE_CAP) == set(SECTOR_OF)
