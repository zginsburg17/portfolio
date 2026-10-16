# quantfolio

A compact quantitative trading research project: a vectorized backtesting engine, four systematic strategies, risk/performance analytics, and a volatility-targeted multi-strategy portfolio. It runs out of the box on reproducible synthetic data, and also accepts real price CSVs.

## Quick start

```bash
pip install -e ".[dev]"
python -m quantfolio                     # synthetic data, prints stats, writes output/report.png
python -m quantfolio --csv prices.csv    # your own wide CSV: Date column + one close-price column per ticker
pytest
```

## What's inside

| Module | Purpose |
|---|---|
| `quantfolio/data.py` | Synthetic multi-asset prices (market factor, drifting alpha, calm/turbulent vol regimes) and a CSV loader |
| `quantfolio/strategies.py` | SMA crossover, time-series momentum (vol-targeted), Bollinger mean reversion, cross-sectional momentum (dollar-neutral) |
| `quantfolio/backtest.py` | Vectorized engine: 1-day signal lag, per-turnover transaction costs, gross-exposure cap |
| `quantfolio/metrics.py` | CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, hit rate |
| `quantfolio/portfolio.py` | Inverse-vol strategy blending with a lagged vol-target overlay |
| `quantfolio/report.py` | Equity-curve / drawdown chart and formatted tables |

## Design notes

- **No look-ahead.** Signals use data through day *t* and are executed on *t+1*; weights and leverage in the portfolio layer are likewise lagged. `tests/test_quantfolio.py::test_no_lookahead_bias` guards this.
- **Costs are on by default** (5 bps per unit of one-way turnover). Try `--cost-bps 25` and watch the high-turnover strategies degrade.
- **Honest results.** On the synthetic data the mean-reversion strategy loses money, because the simulated drift is persistent. That's expected and worth discussing: the diversification benefit comes from the negatively correlated sleeve even though it loses on its own.
- Synthetic results are illustrative, not evidence of real-world edge. Run on real data and add out-of-sample validation before drawing conclusions.

## Ideas to extend

Walk-forward parameter optimization, a regime filter, a proper slippage/market-impact model, `yfinance` data ingestion, a factor-neutralization step, and a Monte Carlo / bootstrap Sharpe confidence interval.
