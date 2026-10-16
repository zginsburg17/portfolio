"""Command line entry point: run every strategy, combine, and report."""

from __future__ import annotations

import argparse

import pandas as pd

from .backtest import backtest
from . import studies
from .datasources import LARGE_CAP, DataSourceError, clean_panel, fetch_factors, fetch_prices
from .data import load_prices_csv, synthetic_ohlc, synthetic_pair, synthetic_prices
from .metrics import performance_summary
from .portfolio import combine_inverse_vol, correlation_matrix
from .report import format_table, plot_report, summary_table
from .strategies import STRATEGIES


def load_universe(args: argparse.Namespace) -> pd.DataFrame:
    """Resolve the price panel from --csv, --source yahoo, or synthetic data."""
    if args.csv:
        return load_prices_csv(args.csv)
    if args.source == "yahoo":
        tickers = args.tickers.split(",") if args.tickers else LARGE_CAP
        raw = fetch_prices(tickers, args.start, args.end, refresh=args.refresh, offline=args.offline)
        return clean_panel(raw)
    return synthetic_prices(seed=args.seed)


def run_studies(args: argparse.Namespace) -> None:
    prices = load_universe(args)
    want = (lambda k: args.study in (k, "all"))
    if want("trend"):
        studies.trend_study(prices, cost_bps=0.0)
    if want("pairs"):
        studies.pairs_study(synthetic_pair(), cost_bps=args.cost_bps)
    if want("countertrend"):
        studies.countertrend_study(synthetic_ohlc())
    if want("risk"):
        studies.risk_study(prices.pct_change().dropna())
    if want("vol"):
        studies.vol_study(prices.mean(axis=1))
    if want("validation"):
        studies.validation_study(prices, cost_bps=args.cost_bps)
    if want("attribution"):
        rets = pd.DataFrame({n: backtest(prices, fn(prices), cost_bps=args.cost_bps).returns
                             for n, fn in STRATEGIES.items()})
        if args.source == "yahoo" and not args.csv:
            fac = fetch_factors("ff5", args.start, args.end, refresh=args.refresh, offline=args.offline)
            fac = fac.join(fetch_factors("mom", args.start, args.end, refresh=args.refresh,
                                         offline=args.offline), how="inner")
            studies.attribution_study(rets, fac, "RF")
        else:
            print("\n(synthetic/CSV data: using the universe's equal-weight return as the only "
                  "'market' factor; use --source yahoo for Fama-French factors)")
            studies.attribution_study(rets, pd.DataFrame({"MKT": prices.pct_change().mean(axis=1)}), None)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the quantfolio multi-strategy demo.")
    ap.add_argument("--csv", help="wide CSV of close prices (Date index); default: synthetic data")
    ap.add_argument("--source", choices=["synthetic", "yahoo"], default="synthetic",
                    help="price source (yahoo = public daily bars, cached on disk)")
    ap.add_argument("--tickers", help="comma-separated tickers for --source yahoo (default: large-cap universe)")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end")
    ap.add_argument("--refresh", action="store_true", help="ignore cache freshness and re-download")
    ap.add_argument("--offline", action="store_true", help="use only cached data, never the network")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--out", default="output/report.png")
    ap.add_argument("--study", choices=["trend", "pairs", "countertrend", "risk", "vol", "validation", "attribution", "all"],
                    help="run a technique study instead of the multi-strategy demo")
    args = ap.parse_args(argv)

    try:
        _run(args)
    except DataSourceError as e:
        ap.exit(2, f"data error: {e}\n")


def _run(args: argparse.Namespace) -> None:
    if args.study:
        return run_studies(args)

    prices = load_universe(args)
    print(f"Universe: {prices.shape[1]} assets, {len(prices)} days "
          f"({prices.index[0].date()} -> {prices.index[-1].date()}), cost {args.cost_bps} bps\n")

    results = {name: backtest(prices, fn(prices), cost_bps=args.cost_bps)
               for name, fn in STRATEGIES.items()}
    rets = pd.DataFrame({n: r.returns for n, r in results.items()})
    port = combine_inverse_vol(rets)

    stats = {n: r.summary() for n, r in results.items()}
    stats["PORTFOLIO"] = performance_summary(port)
    print(format_table(summary_table(stats)))
    print("\nStrategy return correlations:")
    print(correlation_matrix(rets).round(2).to_string())
    print(f"\nChart saved to {plot_report(rets, port, args.out)}")


if __name__ == "__main__":
    main()
