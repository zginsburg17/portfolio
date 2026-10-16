"""A liquid US large-cap research universe with GICS-style sectors.

WARNING (survivorship bias): this is a *current* snapshot of well-known names.
Backtesting it over past decades silently excludes every company that failed,
shrank or was delisted, which flatters results, especially for long-only and
momentum. Treat results as illustrative, and say so in any write-up. A
point-in-time constituent history would be needed to remove the bias.
"""

SECTORS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "INTC"],
    "Communication": ["GOOGL", "META", "NFLX", "DIS", "VZ", "T"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ"],
    "Health Care": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "AXP"],
    "Industrials": ["CAT", "BA", "HON", "UPS", "GE", "LMT"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Utilities": ["NEE", "DUK", "SO"],
    "Materials": ["LIN", "APD", "SHW"],
    "Real Estate": ["AMT", "PLD", "SPG"],
}

LARGE_CAP: list[str] = [t for ts in SECTORS.values() for t in ts]
SECTOR_OF: dict[str, str] = {t: s for s, ts in SECTORS.items() for t in ts}

# Benchmarks / cross-asset proxies for beta and regime work.
BENCHMARKS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
