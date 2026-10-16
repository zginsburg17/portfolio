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

## Public data sources

No API keys needed. Everything is cached on disk (`~/.cache/quantfolio`, override with `QUANTFOLIO_CACHE`), so reruns are fast and reproducible, and a provider outage falls back to the stale cache.

| Source | Module | Provides | Host |
|---|---|---|---|
| Yahoo Finance | `datasources.yahoo` | Daily OHLCV, split/dividend-adjusted close | `query1.finance.yahoo.com` |
| FRED | `datasources.fred` | Rates, VIX, credit spreads, any macro series | `fred.stlouisfed.org` |
| Ken French library | `datasources.famafrench` | Daily FF3 / FF5 / momentum factors for attribution | `mba.tuck.dartmouth.edu` |

```python
from quantfolio.datasources import fetch_prices, fetch_series, fetch_factors, clean_panel, LARGE_CAP
px = clean_panel(fetch_prices(LARGE_CAP, "2010-01-01"))      # adjusted closes, strategy-ready
rf = fetch_series("DGS3MO", "2010-01-01")                    # 3-month T-bill
fac = fetch_factors("ff5", "2010-01-01")                     # market, size, value, profitability, investment
```

```bash
python -m quantfolio --source yahoo --start 2012-01-01                      # demo on the large-cap universe
python -m quantfolio --source yahoo --tickers SPY,TLT,GLD,QQQ --study trend
python -m quantfolio --source yahoo --offline                               # cache only, never the network
```

`datasources/universe.py` ships a sector-tagged large-cap list. It is a *current* snapshot, so it carries survivorship bias; see the warning in that file before trusting long-horizon results.

## The flagship strategy: sector- and beta-neutral stock selection

`neutral.neutral_weights` builds a unit-gross long/short book whose net exposure to every sector is exactly zero and whose ex-ante beta to the market is exactly zero, so returns come from stock selection rather than sector or market bets. The beta step moves weights along the sector-demeaned beta direction, which sums to zero inside each sector, so the two constraints don't fight each other. `neutral.exposure_check` verifies gross, net, per-sector net and beta on the dates the book was rebalanced; the test suite also plants a known signal (strategy must find it), plants a trending market (realized beta must be removed), and mutation-checks that any look-ahead in the momentum, reversal or beta estimates fails the no-look-ahead test.

```bash
python -m quantfolio --study neutral                        # synthetic plumbing check
python -m quantfolio --source yahoo --start 2005-01-01 --study neutral   # the real run
```

On the built-in synthetic data the neutral book is weak (out-of-sample Sharpe about 0.4, interval spanning zero). That is the expected answer there, since the simulated drifts decay over weeks and a 12-1 signal shouldn't work, and it shows the validation tools refusing to be impressed. The real-data result is the one to write up, whichever way it comes out.

## Validation and attribution

Backtests flatter. These tools discount a result for noise, selection and fragility, and test whether the returns are alpha or just factor exposure.

| Question | Tool | Module |
|---|---|---|
| How uncertain is this Sharpe? | Circular block-bootstrap CI; probabilistic Sharpe ratio (skew/kurtosis-aware) | `validation.bootstrap_sharpe_ci`, `probabilistic_sharpe` |
| Did I just pick the luckiest of N variants? | Expected-max-Sharpe hurdle and deflated Sharpe ratio | `validation.expected_max_sharpe`, `deflated_sharpe` |
| Does it work on data it wasn't tuned on? | Rolling/anchored walk-forward with per-fold parameter selection | `validation.walk_forward` |
| Is the edge a plateau or a spike? | 2-D parameter sensitivity grid | `validation.sensitivity_grid` |
| Is it alpha or beta? | Factor regression (FF5 + momentum) with Newey-West t-stats; rolling-beta hedge | `attribution.factor_regression`, `beta_hedge` |

```bash
python -m quantfolio --study validation                                  # synthetic data
python -m quantfolio --source yahoo --study validation --start 2010-01-01
python -m quantfolio --source yahoo --study attribution                  # needs Ken French factors
```

The tests check that these tools reject the null: on a zero-edge random walk the walk-forward out-of-sample Sharpe is below the in-sample one, its bootstrap CI spans zero, and the probabilistic Sharpe of pure noise is spread across [0, 1] rather than clustered near 1. The default synthetic data has a planted trend, so the validation study "passing" there is only a plumbing check. The conclusions that matter come from real data.

## Technique studies (from the Systematic Investing coursework)

`python -m quantfolio --study {trend,pairs,countertrend,risk,vol,validation,attribution,neutral,all}` runs one study per technique family. Each assignment's setup is generalized into reusable library code:

| Study | Concepts implemented | Where |
|---|---|---|
| **trend** | MA crossovers 10/30, 30/100, 80/160, 30-day breakout, EMA vs SMA; best-per-instrument selection; equal-weight vs inverse-20d-vol combination; Sharpe/Sortino; max drawdown and its date | `strategies.ma_crossover`, `breakout`, `TREND_SYSTEMS`; `studies.trend_study` |
| **pairs** | Dickey-Fuller stationarity test on the price *difference*; z-scored N-day returns (5/10/20d, 60-day window); zdiff entry thresholds, profit targets, holding period; legs sized by inverse 20d vol | `stats.adf_test`, `strategies.pairs_spread`; `studies.pairs_study` |
| **countertrend** | Entry at `prev high - P*avg 20d range`, exit at close, gap-through fills at the open; P sweep 2.2 to 0.8; Sharpe with zero vs blank no-signal days; 2x2 market-vs-strategy matrix; short-side mirror | `countertrend.py`; `studies.countertrend_study` |
| **risk** | Pairwise correlation, equal vs inverse-vol combination, drawdown, lead-lag predictability via OLS with t-stats | `portfolio.combine_equal`, `stats.ols`; `studies.risk_study` |
| **neutral** | Sector- and beta-neutral long/short: 12-1 momentum + 1-month reversal, demeaned within sectors, then projected to zero ex-ante beta without breaking sector neutrality; monthly rebalance; compared against raw and sector-only versions; constraint check, market and Fama-French attribution, walk-forward validation | `neutral.py`; `studies.neutral_study` |
| **vol** | Trailing z-scores (excluding today, N=250), quintile buckets of zvol vs concurrent and forward normalized returns | `stats.zscore_trailing`, `quintile_table`; `studies.vol_study` |

**How studies get data.** `inputs.Inputs` resolves each study's data from one `Config` (synthetic, `--source yahoo`, or `--csv`) and memoizes it. The pairs study takes `--pair KO,PEP` and the countertrend study `--ohlc-ticker SPY`; Yahoo OHLC bars are rescaled by adj_close/close so splits create no fake gaps. A source that can't provide an input (for example OHLC from a close-only CSV) raises an error instead of substituting synthetic data. **What studies return.** Each study returns a `StudyResult` (titled sections of DataFrames or text, plus a `metrics` dict of headline numbers); `report.render` is the only place that formats it, so results can be tested, saved or plotted without parsing stdout. Adding a study is one function plus one line in `studies.STUDIES`.

Cross-cutting conventions that carry through every strategy: signals use information through day *t* and trade at *t+1*; z-scores exclude the current observation; position sizing is inversely proportional to trailing volatility; and performance is always judged net of costs, on Sharpe *and* drawdown, not CAGR alone.

Caveat on the countertrend "blank" Sharpe: annualizing with sqrt(252) over only the days a signal fired (a few dozen per decade) wildly overstates it. Compare it with the zero-filled Sharpe, which reflects capital actually deployed.

All studies run on synthetic data by default. Use `--source yahoo` (or `--csv` for close-only studies) for real conclusions.

## Design notes

- **No look-ahead.** Signals use data through day *t* and are executed on *t+1*; weights and leverage in the portfolio layer are likewise lagged. `tests/test_quantfolio.py::test_no_lookahead_bias` guards this.
- **Costs are on by default** (5 bps per unit of one-way turnover). Try `--cost-bps 25` and watch the high-turnover strategies degrade.
- **Honest results.** On the synthetic data the mean-reversion strategy loses money, because the simulated drift is persistent. That's expected and worth discussing: the diversification benefit comes from the negatively correlated sleeve even though it loses on its own.
- Synthetic results are illustrative, not evidence of real-world edge. Run on real data and add out-of-sample validation before drawing conclusions.

## Ideas to extend

Walk-forward parameter optimization, a regime filter, a proper slippage/market-impact model, `yfinance` data ingestion, a factor-neutralization step, and a Monte Carlo / bootstrap Sharpe confidence interval.
