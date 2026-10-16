# Methodology

How the sector- and beta-neutral strategy is built, why each choice was made, and how its
results are judged. Nothing here depends on any particular result; the results live in the
generated report (`python -m quantfolio --report output/report.md`).

## 1. Question

Cross-sectional momentum and short-term reversal are among the best-documented return
patterns in equities. A naive long/short implementation, though, mixes three things: genuine
stock selection, accidental sector bets (winners cluster in a few industries), and accidental
market exposure (winners are often higher-beta in a rising market). The question this project
asks is narrow:

> After removing sector and market bets by construction, is there a stock-selection return left
> that is statistically distinguishable from luck, net of trading costs, out of sample?

A negative answer is a legitimate outcome and is reported as such.

## 2. Timing and data conventions

- A signal computed from information through the close of day *t* earns the return from the
  close of *t* to the close of *t+1*. The backtester enforces the one-day lag; strategies are
  forbidden from peeking. Every signal, beta estimate and z-score uses data through *t* only.
- Prices are split- and dividend-adjusted closes. Weights are treated as targets each day
  (constant-weight between rebalances), a standard simplification.
- The built-in universe is a present-day snapshot of large caps with sector tags
  (`datasources/universe.py`). This introduces **survivorship bias**, discussed in section 8.

## 3. Signal

For stock *i* on day *t*:

- **Momentum (12-1):** `mom = P(t-21) / P(t-252) - 1`. The most recent month is skipped
  because one-month returns tend to reverse, which would contaminate the trend signal.
- **Reversal (1 month):** `rev = -( P(t) / P(t-21) - 1 )`.
- Each is **cross-sectionally z-scored** across stocks on each date, then combined:
  `score = z(mom) + w_rev * z(rev)` with `w_rev = 0.5` by default.

Cross-sectional z-scoring makes the signal a pure ranking of stocks against each other on the
same date, so a market-wide trend cannot masquerade as a stock signal.

## 4. Portfolio construction

**Sector neutrality.** On each date, subtract the sector mean score from every stock's score
(over stocks with valid data). The resulting vector `d` sums to zero inside every sector, so
the portfolio is long the best stocks and short the worst *within each industry*. Sectors with
fewer than `min_names` valid stocks (default 3) are excluded that day, because a two-stock
sector's "neutral" bet is just one coin flip.

**Beta neutrality.** Each stock's beta is a rolling 252-day regression slope on the universe's
equal-weight return, estimated from past data only. Let `beta` be the vector of betas and
`b` the same vector demeaned within sectors. Define

    k  = (d . beta) / (b . beta)
    w  = d - k * b

Two properties follow directly:

1. `w` is still sector-neutral, because `b` sums to zero inside every sector.
2. `w . beta = d . beta - k (b . beta) = 0`, so ex-ante portfolio beta is exactly zero.

This is also the *smallest possible change* to the scores: because `d` sums to zero inside
sectors, `d . beta = d . b` and `b . beta = b . b`, which makes `w` the Euclidean projection of
`d` onto the set of weights that are both sector-neutral and beta-neutral. The portfolio stays
as close to the raw signal as the constraints allow.

**Scaling and rebalancing.** Weights are divided by their absolute sum (unit gross: 0.5 long,
0.5 short) and updated every 21 trading days, then held. Monthly rebalancing limits turnover,
which matters because costs scale with it.

`neutral.exposure_check` measures gross, net, per-sector net and beta on the dates the book was
actually rebalanced; the report prints it, so the constraints are verified, not assumed.

## 5. Backtest mechanics

- Daily return = sum over stocks of (held weight x that day's stock return) minus costs.
- Costs: `cost_bps` per unit of one-way turnover (default 5 bps), meant to cover commission,
  half-spread and slippage for liquid large caps.
- Not modelled: market impact, borrow fees and short availability, financing spread, taxes,
  intraday execution risk. These all reduce real-world returns.

## 6. How results are judged

A backtest Sharpe ratio is a point estimate with a large error bar and a selection bias. The
judgement stack, in order:

1. **Walk-forward testing.** Parameters (lookback, reversal weight; nine combinations) are chosen
   on a trailing three-year window and traded on the next unseen six months, rolling forward.
   The stitched out-of-sample returns are the only returns used for inference. A large gap
   between mean in-sample and out-of-sample Sharpe measures how much parameter selection
   overstated performance.
2. **Block bootstrap interval.** The out-of-sample Sharpe's 95% interval comes from a circular
   block bootstrap (20-day blocks), which preserves short-range autocorrelation and volatility
   clustering that an i.i.d. bootstrap would destroy.
3. **Probabilistic Sharpe ratio (PSR).** The probability that the true Sharpe exceeds a
   benchmark, accounting for sample length, skewness and kurtosis (Bailey and Lopez de Prado):

       PSR = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )

   with per-period Sharpes. Non-normal returns with fat tails make a given Sharpe less
   convincing, and this formula says by how much.
4. **Deflated Sharpe ratio (DSR).** Trying N variants and reporting the best inflates the
   best Sharpe even with no edge. DSR is the PSR measured against the Sharpe the *best of N
   luck-only trials* would be expected to reach, using the cross-trial variance of Sharpes.
5. **Factor attribution.** Strategy returns are regressed on market (and, with live data,
   Fama-French five-factor plus momentum) returns with Newey-West standard errors, which are
   robust to the autocorrelation and heteroskedasticity of daily returns. Alpha is annualized.

**Pre-committed reading of the result** (coded in `writeup.verdict`, so it cannot be tuned after
seeing the answer):

- *Evidence of an edge:* out-of-sample Sharpe positive, bootstrap interval entirely above zero,
  and DSR at least 95%.
- *Suggestive only:* out-of-sample Sharpe positive with PSR at least 95%, but DSR below 95%.
- *Not distinguishable from luck:* anything else.

A realized market beta above 0.10 in absolute value is flagged as a failure of the neutral
construction out of sample.

## 7. How the code is tested

The properties that matter are asserted directly:

- exact gross, net, per-sector net and beta constraints on rebalance dates;
- **no look-ahead:** rewriting all prices after a rebalance date must leave weights up to that
  date unchanged;
- **power:** a planted persistent cross-sectional signal must be found, and a planted trending
  market must have its beta removed;
- **mutation checks:** deliberately inserting a one-day look-ahead into the momentum, reversal
  or beta estimates makes the suite fail (this caught a weakness in the first version of the
  look-ahead test, which has since been fixed);
- validation tools must reject the null: on zero-edge random walks, out-of-sample Sharpe falls
  below in-sample and the bootstrap interval spans zero.

## 8. Limitations

- **Survivorship bias.** The universe contains companies that are large *today*. Names that
  collapsed, were acquired, or fell out of the index are absent, which flatters the long side of
  a momentum strategy over long horizons. A point-in-time constituent history is required to
  remove it. Results should be read as upper bounds.
- **Idealized execution and costs** (section 5).
- **Small, single-market sample.** One universe, one history, one rebalance frequency. No
  claim is made about other markets, regimes or sub-periods.
- **Estimated betas.** Realized beta drifts between rebalances, and an equal-weight proxy for
  the market is not the cap-weighted index.
- **A search was run.** The DSR corrects for the nine variants evaluated here, not for ideas
  discarded earlier.

## 9. What would strengthen it

Point-in-time universes; a volatility-scaled cost model with market-impact estimates; sub-period
and cross-market stability tests; capacity analysis; sector-relative and industry-group
neutrality; and a comparison against commercial factor-model residuals.
