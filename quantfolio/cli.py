"""Command line entry point: run every strategy, combine, and report."""

from __future__ import annotations

import argparse

import pandas as pd

from .backtest import backtest
from . import studies
from .data import load_prices_csv, synthetic_ohlc, synthetic_pair, synthetic_prices
from .metrics import performance_summary
from .portfolio import combine_inverse_vol, correlation_matrix
from .report import format_table, plot_report, summary_table
from .strategies import STRATEGIES


def run_studies(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.csv) if args.csv else synthetic_prices(seed=args.seed)
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the quantfolio multi-strategy demo.")
    ap.add_argument("--csv", help="wide CSV of close prices (Date index); default: synthetic data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--out", default="output/report.png")
    ap.add_argument("--study", choices=["trend", "pairs", "countertrend", "risk", "vol", "all"],
                    help="run a technique study instead of the multi-strategy demo")
    args = ap.parse_args(argv)

    if args.study:
        return run_studies(args)

    prices = load_prices_csv(args.csv) if args.csv else synthetic_prices(seed=args.seed)
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
