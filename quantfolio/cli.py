"""Command line entry point: run the multi-strategy demo or any technique study."""

from __future__ import annotations

import argparse

import pandas as pd

from .datasources import DataSourceError
from .inputs import Config, Inputs
from .metrics import performance_summary
from .portfolio import combine_inverse_vol
from .report import format_table, plot_report, render, summary_table
from .studies import STUDIES


def demo(inp: Inputs, out: str) -> None:
    prices, rets = inp.prices, inp.strategy_returns
    print(f"Universe: {prices.shape[1]} assets, {len(prices)} days "
          f"({prices.index[0].date()} -> {prices.index[-1].date()}), cost {inp.cfg.cost_bps} bps\n")
    port = combine_inverse_vol(rets)
    stats = {n: r.summary() for n, r in inp.backtests.items()}
    stats["PORTFOLIO"] = performance_summary(port)
    print(format_table(summary_table(stats)))
    print("\nStrategy return correlations:")
    print(rets.corr().round(2).to_string())
    print(f"\nChart saved to {plot_report(rets, port, out)}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the quantfolio multi-strategy demo or a study.")
    ap.add_argument("--csv", help="wide CSV of close prices (Date index); overrides --source")
    ap.add_argument("--source", choices=["synthetic", "yahoo"], default="synthetic",
                    help="data source (yahoo = public daily bars, cached on disk)")
    ap.add_argument("--tickers", help="comma-separated tickers (default: large-cap universe)")
    ap.add_argument("--pair", default="KO,PEP", help="two tickers for the pairs study")
    ap.add_argument("--ohlc-ticker", default="SPY", help="instrument for the countertrend study")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end")
    ap.add_argument("--refresh", action="store_true", help="ignore cache freshness and re-download")
    ap.add_argument("--offline", action="store_true", help="use only cached data, never the network")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--out", default="output/report.png")
    ap.add_argument("--study", choices=[*STUDIES, "all"],
                    help="run a technique study instead of the multi-strategy demo")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    a = ap.parse_args(argv)
    pair = tuple(a.pair.split(","))
    if len(pair) != 2:
        ap.error("--pair needs exactly two tickers, e.g. KO,PEP")
    cfg = Config(source=a.source, csv=a.csv, pair=pair, ohlc_ticker=a.ohlc_ticker,
                 tickers=tuple(a.tickers.split(",")) if a.tickers else None, start=a.start,
                 end=a.end, seed=a.seed, cost_bps=a.cost_bps, refresh=a.refresh, offline=a.offline)
    inp = Inputs(cfg)
    try:
        if not a.study:
            demo(inp, a.out)
        for name in (STUDIES if a.study == "all" else [a.study] if a.study else []):
            print(render(STUDIES[name](inp)))
    except DataSourceError as e:
        ap.exit(2, f"data error: {e}\n")


if __name__ == "__main__":
    main()
