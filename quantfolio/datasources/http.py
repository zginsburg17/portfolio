"""Minimal HTTP transport (stdlib only) with retries and a swappable backend for tests."""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

log = logging.getLogger("quantfolio.data")

USER_AGENT = "Mozilla/5.0 (compatible; quantfolio/0.1; research)"


class DataSourceError(RuntimeError):
    """A data provider could not be reached or returned something unusable."""


# A transport maps (url, headers) -> response bytes. Tests inject a fake one.
Transport = Callable[[str, dict], bytes]


def urllib_transport(url: str, headers: dict, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # honors HTTPS_PROXY / SSL_CERT_FILE
        return resp.read()


def fetch(url: str, params: dict | None = None, transport: Transport | None = None,
          retries: int = 3, backoff: float = 1.0) -> bytes:
    """GET with exponential backoff on 429/5xx and transient network errors.

    403/407/404 are not retried: they mean "not permitted / not found", and
    hammering a policy denial helps nobody.
    """
    transport = transport or urllib_transport
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    host = urllib.parse.urlparse(url).netloc
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return transport(url, {"User-Agent": USER_AGENT})
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            if any(code in str(e) for code in ("403", "407")):  # proxy/policy denial
                break
        if attempt < retries:
            time.sleep(backoff * 2**attempt)
    raise DataSourceError(f"request to {host} failed: {last}") from last
