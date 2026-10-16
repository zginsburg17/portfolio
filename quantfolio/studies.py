"""One study per technique family. Each takes data and returns a StudyResult (no printing)."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .attribution import attribution_table
from .backtest import backtest
from .countertrend import countertrend, market_vs_strategy, sharpe_of, sweep_p
from .inputs import Inputs
from .metrics import drawdown, performance_summary, sharpe, sharpes, sortino
from .portfolio import combine_equal, combine_inverse_vol
from .report import summary_table
from .results import StudyResult
from .stats import adf_test, ols, quintile_table, zscore_trailing
from .strategies import TREND_SYSTEMS, equal_weight, ma_crossover, pairs_spread
from .validation import (bootstrap_sharpe_ci, deflated_sharpe, probabilistic_sharpe,
                         sensitivity_grid, walk_forward)


def trend_study(prices: pd.DataFrame, cost_bps: float = 0.0) -> StudyResult:
    """MA 10/30, 30/100, 80/160, 30d breakout + EMA 10/30, per instrument; then combos."""
    res = StudyResult("TREND: Sharpe by instrument and system")
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
    res.add("Sharpe by instrument and system", tbl)

    best = tbl[list(TREND_SYSTEMS)].idxmax(axis=1)
    best_rets = pd.DataFrame({i: rets[(i, best[i])] for i in best.index})
    picks = pd.DataFrame({"best system": best,
                          "Sharpe": [sharpe(best_rets[i]) for i in best.index],
                          "Sortino": [sortino(best_rets[i]) for i in best.index]})
    res.add("Best of the four per instrument (selection-biased: see the validation study)", picks)

    eq = combine_equal(best_rets)
    iv = combine_inverse_vol(best_rets, window=20, target_vol=None)
    mdd = drawdown(eq)
    res.metrics.update(
        ema_beats_sma=int((tbl["EMA10/30"] > tbl["MA10/30"]).sum()), n_instruments=len(tbl),
        equal_sharpe=sharpe(eq), inv_vol_sharpe=sharpe(iv), max_dd=float(mdd.min()))
    res.add("Combination of the best systems",
            f"EMA10/30 beats MA10/30 on {res.metrics['ema_beats_sma']}/{len(tbl)} instruments\n"
            f"Equal weight:      Sharpe {sharpe(eq):.2f}, max DD {mdd.min():.1%} on "
            f"{mdd.idxmin().date()}\n"
            f"Inverse 20d vol:   Sharpe {sharpe(iv):.2f}")
    return res


def pairs_study(pair: pd.DataFrame, cost_bps: float = 2.0) -> StudyResult:
    """Dickey-Fuller cointegration screen, then 5/10/20-day spread strategies."""
    res = StudyResult("PAIRS: cointegration screen + spread strategies")
    a, b = pair.columns
    df_res = adf_test((pair[a] - pair[b]).iloc[:1000])
    res.metrics.update(adf_t=df_res["tstat"], stationary=float(df_res["stationary_5pct"]))
    res.add("Dickey-Fuller on the price difference (first 1000 days)",
            f"{a}-{b}: slope {df_res['coef']:.3f}, t = {df_res['tstat']:.2f} "
            f"(5% critical {df_res['critical']['5%']}) -> "
            f"{'stationary: promising pair' if df_res['stationary_5pct'] else 'cannot reject unit root'}")
    rets = pd.DataFrame({
        f"spread{n}": backtest(pair, pairs_spread(pair, lookback=n), cost_bps=cost_bps,
                               max_gross=None).returns for n in (5, 10, 20)})
    res.add("Spread strategies", summary_table(
        {c: performance_summary(rets[c]) for c in rets})[["CAGR", "Sharpe", "Max DD"]])
    res.metrics["combo_sharpe"] = sharpe(combine_equal(rets))
    res.add("Equal-weight combination", f"Sharpe {res.metrics['combo_sharpe']:.2f}")
    return res


def countertrend_study(ohlc: pd.DataFrame) -> StudyResult:
    res = StudyResult("COUNTERTREND: retracement entries, exit at close")
    res.metrics["buy_hold_sharpe"] = sharpe(ohlc["close"].pct_change().dropna())
    res.add("Buy & hold", f"Sharpe {res.metrics['buy_hold_sharpe']:.2f}")
    res.add("LONG side, P sweep", sweep_p(ohlc))
    res.add("SHORT side, P sweep", sweep_p(ohlc, side="short"))
    res.add("Long, P=2.2: strategy vs market on signal days",
            market_vs_strategy(ohlc, countertrend(ohlc, 2.2)))
    res.add("Note", "'blank' Sharpe annualizes over signal days only and overstates; "
                    "compare with the zero-filled Sharpe.")
    return res


def risk_study(returns: pd.DataFrame) -> StudyResult:
    """Correlation, equal vs inverse-vol combination, and lead-lag predictability."""
    res = StudyResult("RISK: correlation, combination, predictability")
    cols = list(returns.columns[:3])
    r = returns[cols]
    res.add("Pairwise correlations", r.corr())
    res.add("Standalone Sharpe", pd.DataFrame({"Sharpe": sharpes(r)}))
    combos = {"Equal weight": combine_equal(r), "Inverse 20d vol": combine_inverse_vol(r, 20, None)}
    res.add("Combinations", pd.DataFrame({
        n: {"Sharpe": sharpe(c), "Max DD": drawdown(c).min()} for n, c in combos.items()}).T)
    tgt, feats = cols[2], cols[:2]
    fit = ols(r[tgt], r[feats].shift(1).add_suffix("_lag1"))
    res.metrics["predict_r2"] = fit["r2"]
    res.add(f"Predict {tgt} from lagged {feats} (|t| > 2 ~ significant)",
            pd.DataFrame({"coef": fit["coef"], "t-stat": fit["tstat"]}))
    return res


def vol_study(price: pd.Series, vol_window: int = 20, z_window: int = 250) -> StudyResult:
    """Quintiles of normalized volatility vs concurrent and forward normalized returns."""
    res = StudyResult("VOL: volatility regime vs returns (quintiles of zvol)")
    vol = price.pct_change().rolling(vol_window).std()
    fret = price.shift(-vol_window) / price - 1
    t = quintile_table(
        zscore_trailing(vol, z_window),
        {"zret": zscore_trailing(price.pct_change(vol_window), z_window),
         "zfret": zscore_trailing(fret, z_window)},
    ).rename(columns={"x": "zvol"})
    res.add("Average zvol, zret (concurrent) and zfret (forward) by volatility quintile", t)
    res.add("Reading", "Concurrent: high vol coincides with weak returns if zret falls across "
                       "buckets.\nLead-lag: if zfret rises with zvol, high-vol periods preceded "
                       "better forward returns.")
    return res


def validation_study(prices: pd.DataFrame, cost_bps: float = 5.0) -> StudyResult:
    """How much of a 'best parameters' backtest survives honest testing?"""
    res = StudyResult("VALIDATION: in-sample optimum vs out-of-sample reality")
    grid = {"fast": [5, 10, 20, 30, 40], "slow": [60, 80, 100, 150, 200]}
    wf = walk_forward(prices, equal_weight(ma_crossover), grid, cost_bps=cost_bps)
    trial_sr = sharpes(wf.trials)
    n_trials = len(trial_sr)
    best_key = trial_sr.idxmax()
    ci = bootstrap_sharpe_ci(wf.oos_returns)
    psr = probabilistic_sharpe(wf.oos_returns)
    dsr = deflated_sharpe(wf.trials[best_key], n_trials, float(trial_sr.var(ddof=0)))
    res.metrics.update(best_full_sample=float(trial_sr.max()), is_sharpe=wf.is_sharpe,
                       oos_sharpe=wf.oos_sharpe, oos_ci_lo=ci["lo"], oos_ci_hi=ci["hi"],
                       psr=psr, dsr=dsr)
    res.add("Selection bias",
            f"{n_trials} parameter sets. Full-sample best {dict(best_key)} -> Sharpe "
            f"{trial_sr.max():.2f} (selection-biased).\n"
            f"Walk-forward over {len(wf.folds)} folds: mean in-sample Sharpe {wf.is_sharpe:.2f} "
            f"vs out-of-sample {wf.oos_sharpe:.2f}.\n"
            f"Distinct parameter sets chosen across folds: "
            f"{wf.folds['params'].astype(str).nunique()} (unstable choice = fragile edge).")
    res.add("Out-of-sample significance",
            f"95% bootstrap CI for OOS Sharpe: [{ci['lo']:.2f}, {ci['hi']:.2f}]  "
            f"P(Sharpe<=0) = {ci['p_le_zero']:.1%}  ({ci['n_days']} days)\n"
            f"Probabilistic Sharpe, OOS (P true SR > 0): {psr:.1%}\n"
            f"Deflated Sharpe of the full-sample winner (P true SR > luck-adjusted hurdle): "
            f"{dsr:.1%}")
    res.add("Sharpe sensitivity (rows = fast MA, cols = slow MA)",
            sensitivity_grid(trial_sr, x="slow", y="fast"))
    return res


def attribution_study(strategy_returns: pd.DataFrame, factors: pd.DataFrame,
                      rf_col: str | None = "RF", note: str = "") -> StudyResult:
    """Regress each strategy on factors: is there alpha beyond the exposures?"""
    res = StudyResult("ATTRIBUTION: alpha vs factor exposure (Newey-West t-stats)")
    if note:
        res.add("Factors", note)
    res.add("Alpha and exposures", attribution_table(strategy_returns, factors, rf_col))
    res.add("Reading", "|alpha t| > 2 suggests alpha beyond the listed factors; a high R2 with a "
                       "large market beta means the strategy is mostly repackaged exposure.")
    return res


# name -> adapter pulling exactly the inputs that study needs. Adding a study = one line here.
STUDIES: dict[str, Callable[[Inputs], StudyResult]] = {
    "trend": lambda i: trend_study(i.prices),
    "pairs": lambda i: pairs_study(i.pair, i.cfg.cost_bps),
    "countertrend": lambda i: countertrend_study(i.ohlc),
    "risk": lambda i: risk_study(i.prices.pct_change().dropna()),
    "vol": lambda i: vol_study(i.prices.mean(axis=1)),
    "validation": lambda i: validation_study(i.prices, i.cfg.cost_bps),
    "attribution": lambda i: attribution_study(i.strategy_returns, *i.factors[:2], note=i.factors[2]),
}
