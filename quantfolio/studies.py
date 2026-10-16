"""One study per technique family. Each takes prices (or OHLC) and prints its findings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest
from .countertrend import countertrend, market_vs_strategy, sharpe_of, sweep_p
from .metrics import drawdown, equity_curve, performance_summary, sharpe, sortino
from .portfolio import combine_equal, combine_inverse_vol
from .stats import adf_test, ols, quintile_table, zscore_trailing
from .strategies import TREND_SYSTEMS, ma_crossover, pairs_spread


def _hdr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def trend_study(prices: pd.DataFrame, cost_bps: float = 0.0) -> None:
    """MA 10/30, 30/100, 80/160, 30d breakout + EMA 10/30, per instrument; then combos."""
    _hdr("TREND: Sharpe by instrument and system")
    systems = {**TREND_SYSTEMS, "EMA10/30": lambda p: ma_crossover(p, 10, 30, "ema")}
    start = prices.index[160]  # slowest system's first valid day: compare on common ground
    table, rets = {}, {}
    for inst in prices.columns:
        px = prices[[inst]]
        table[inst] = {}
        for name, fn in systems.items():
            r = backtest(px, fn(px), cost_bps=cost_bps, max_gross=None).returns.loc[start:]
            rets[(inst, name)] = r
            table[inst][name] = sharpe(r)
    tbl = pd.DataFrame(table).T
    print(tbl.round(2).to_string())
    base = list(TREND_SYSTEMS)
    best = tbl[base].idxmax(axis=1)
    print("\nBest of the four per instrument:", best.to_dict())
    print(f"Sortino of each best: { {i: round(sortino(rets[(i, best[i])]), 2) for i in best.index} }")
    ema_better = (tbl["EMA10/30"] > tbl["MA10/30"]).sum()
    print(f"EMA10/30 beats MA10/30 on {ema_better}/{len(tbl)} instruments")

    best_rets = pd.DataFrame({i: rets[(i, best[i])] for i in best.index})
    eq = combine_equal(best_rets)
    iv = combine_inverse_vol(best_rets, window=20, target_vol=None)
    mdd = drawdown(eq)
    print(f"\nEqual-weight best-of combo: Sharpe {sharpe(eq):.2f}, "
          f"max DD {mdd.min():.1%} on {mdd.idxmin().date()}")
    print(f"Inverse-20d-vol combo:      Sharpe {sharpe(iv):.2f}")


def pairs_study(pair: pd.DataFrame, cost_bps: float = 2.0) -> None:
    """Dickey-Fuller cointegration screen, then 5/10/20-day spread strategies."""
    _hdr("PAIRS: cointegration screen + spread strategies")
    a, b = pair.columns
    res = adf_test((pair[a] - pair[b]).iloc[:1000])
    print(f"Dickey-Fuller on {a}-{b} (first 1000 days): slope {res['coef']:.3f}, "
          f"t = {res['tstat']:.2f}  (5% critical {res['critical']['5%']})  ->  "
          f"{'stationary: promising pair' if res['stationary_5pct'] else 'cannot reject unit root'}")
    rets = {}
    for n in (5, 10, 20):
        pos = pairs_spread(pair, lookback=n)
        rets[f"spread{n}"] = backtest(pair, pos, cost_bps=cost_bps, max_gross=None).returns
    df = pd.DataFrame(rets)
    print(pd.DataFrame({c: performance_summary(df[c]) for c in df}).T[["CAGR", "Sharpe", "Max DD"]]
          .round(3).to_string())
    print(f"Equal-weight combo Sharpe: {sharpe(combine_equal(df)):.2f}")


def countertrend_study(ohlc: pd.DataFrame) -> None:
    _hdr("COUNTERTREND: retracement entries, exit at close")
    bh = ohlc["close"].pct_change().dropna()
    print(f"Buy & hold Sharpe: {sharpe(bh):.2f}")
    print("\nLONG side, P sweep:")
    print(sweep_p(ohlc).round(2).to_string())
    print("\nSHORT side, P sweep:")
    print(sweep_p(ohlc, side="short").round(2).to_string())
    print("\nLong, P=2.2: strategy vs market on signal days")
    print(market_vs_strategy(ohlc, countertrend(ohlc, 2.2)).round(4).to_string())


def risk_study(returns: pd.DataFrame) -> None:
    """Correlation, equal vs inverse-vol combination, and lead-lag predictability."""
    _hdr("RISK: correlation, combination, predictability")
    cols = list(returns.columns[:3])
    r = returns[cols]
    print("Pairwise correlations:\n" + r.corr().round(2).to_string())
    print("\nStandalone Sharpe:", {c: round(sharpe(r[c]), 2) for c in cols})
    for name, comb in (("Equal weight", combine_equal(r)),
                       ("Inverse 20d vol", combine_inverse_vol(r, 20, None))):
        print(f"{name}: Sharpe {sharpe(comb):.2f}, max DD {drawdown(comb).min():.1%}")
    tgt, feats = cols[2], cols[:2]
    X = r[feats].shift(1).add_suffix("_lag1")
    fit = ols(r[tgt], X)
    print(f"\nPredict {tgt} from lagged {feats}: R2 {fit['r2']:.4f}, "
          f"t-stats { {k: round(float(v), 2) for k, v in fit['tstat'].items()} }  (|t|>2 ~ significant)")


def vol_study(price: pd.Series, vol_window: int = 20, z_window: int = 250) -> None:
    """Quintiles of normalized volatility vs concurrent and forward normalized returns."""
    _hdr("VOL: volatility regime vs returns (quintiles of zvol)")
    ret = price.pct_change()
    vol = ret.rolling(vol_window).std()
    ret_n = price.pct_change(vol_window)
    fret_n = price.shift(-vol_window) / price - 1
    t = quintile_table(
        zscore_trailing(vol, z_window),
        {"zret": zscore_trailing(ret_n, z_window), "zfret": zscore_trailing(fret_n, z_window)},
    )
    t = t.rename(columns={"x": "zvol"})
    print(t.round(3).to_string())
    print("\nConcurrent: high vol coincides with weak returns if zret falls across buckets.\n"
          "Lead-lag: if zfret rises with zvol, high-vol periods preceded better forward returns.")


def validation_study(prices: pd.DataFrame, cost_bps: float = 5.0) -> None:
    """How much of a 'best parameters' backtest survives honest testing?"""
    from .strategies import ma_crossover
    from .validation import (bootstrap_sharpe_ci, deflated_sharpe, probabilistic_sharpe,
                             sensitivity_grid, walk_forward)

    _hdr("VALIDATION: in-sample optimum vs out-of-sample reality")

    def factory(p, fast, slow):
        return ma_crossover(p, fast, slow) / p.shape[1]

    grid = {"fast": [5, 10, 20, 30, 40], "slow": [60, 80, 100, 150, 200]}
    wf = walk_forward(prices, factory, grid, cost_bps=cost_bps)
    sharpes = wf.trials.apply(sharpe)
    n_trials = len(sharpes)
    best_key = sharpes.idxmax()
    best = wf.trials[best_key]
    print(f"Trials: {n_trials} parameter sets. Full-sample best {dict(best_key)} -> "
          f"Sharpe {sharpes.max():.2f} (this number is selection-biased).")
    print(f"Walk-forward over {len(wf.folds)} folds: mean in-sample Sharpe {wf.is_sharpe:.2f} "
          f"vs out-of-sample Sharpe {wf.oos_sharpe:.2f}")
    print(f"Parameters chosen per fold: {wf.folds['params'].astype(str).nunique()} distinct "
          f"(unstable choice = fragile edge)")

    ci = bootstrap_sharpe_ci(wf.oos_returns)
    print(f"\nOOS Sharpe 95% bootstrap CI: [{ci['lo']:.2f}, {ci['hi']:.2f}]  "
          f"P(Sharpe<=0) = {ci['p_le_zero']:.1%}  ({ci['n_days']} days)")
    print(f"Probabilistic Sharpe, OOS (P true SR > 0): {probabilistic_sharpe(wf.oos_returns):.1%}")
    var = float((sharpes ** 2).mean() - sharpes.mean() ** 2)
    print(f"Deflated Sharpe of the full-sample winner (P true SR > luck-adjusted hurdle): "
          f"{deflated_sharpe(best, n_trials, var):.1%}")

    hm = sensitivity_grid(prices, factory, ("slow", grid["slow"]), ("fast", grid["fast"]),
                          cost_bps=cost_bps)
    print("\nSharpe sensitivity (rows = fast MA, cols = slow MA):")
    print(hm.round(2).to_string())


def attribution_study(strategy_returns: pd.DataFrame, factors: pd.DataFrame,
                      rf_col: str | None = "RF") -> None:
    """Regress each strategy on factors: is there alpha beyond the exposures?"""
    from .attribution import attribution_table

    _hdr("ATTRIBUTION: alpha vs factor exposure (Newey-West t-stats)")
    tbl = attribution_table(strategy_returns, factors, rf_col)
    print(tbl.round(3).to_string())
    print("\nRead: |alpha t| > 2 suggests alpha beyond the listed factors; a high R2 with a large "
          "market beta means the strategy is mostly repackaged exposure.")
