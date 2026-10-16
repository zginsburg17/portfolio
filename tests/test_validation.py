import numpy as np
import pandas as pd
import pytest

from quantfolio.attribution import beta_hedge, factor_regression
from quantfolio.metrics import sharpes
from quantfolio.stats import ols, ols_hac
from quantfolio.strategies import equal_weight, ma_crossover
from quantfolio.validation import (bootstrap_sharpe_ci, deflated_sharpe, expected_max_sharpe,
                                   probabilistic_sharpe, sensitivity_grid, walk_forward)


def noise(n=2500, seed=0, mu=0.0, sigma=0.01):
    return pd.Series(np.random.default_rng(seed).normal(mu, sigma, n),
                     index=pd.bdate_range("2012-01-01", periods=n))


def random_walk_prices(n_assets=6, n=3000, seed=3):
    r = np.random.default_rng(seed).normal(0, 0.01, (n, n_assets))  # zero drift, no structure
    return pd.DataFrame(100 * np.exp(r.cumsum(0)), index=pd.bdate_range("2010-01-01", periods=n),
                        columns=[f"S{i}" for i in range(n_assets)])


def test_bootstrap_ci_brackets_zero_for_noise_and_excludes_it_for_edge():
    ci = bootstrap_sharpe_ci(noise(seed=1), n_boot=1000)
    assert ci["lo"] < 0 < ci["hi"] and ci["lo"] <= ci["sharpe"] <= ci["hi"]
    edge = bootstrap_sharpe_ci(noise(seed=1, mu=0.001), n_boot=1000)  # Sharpe ~1.6
    assert edge["lo"] > 0 and edge["p_le_zero"] < 0.01
    with pytest.raises(ValueError):
        bootstrap_sharpe_ci(noise(n=30))


def test_bootstrap_is_reproducible():
    a = bootstrap_sharpe_ci(noise(), seed=5, n_boot=500)
    assert a == bootstrap_sharpe_ci(noise(), seed=5, n_boot=500)


def test_probabilistic_sharpe_orders_sensibly():
    assert probabilistic_sharpe(noise(mu=0.001)) > 0.99
    # zero-edge series: PSR is ~uniform on [0, 1], so it averages ~0.5 and rarely looks significant
    vals = np.array([probabilistic_sharpe(noise(seed=s)) for s in range(60)])
    assert 0.35 < vals.mean() < 0.65 and (vals > 0.95).mean() < 0.15
    short = noise(n=120, mu=0.001)
    assert probabilistic_sharpe(short) < probabilistic_sharpe(noise(n=2500, mu=0.001))


def test_more_trials_raise_the_luck_hurdle_and_lower_deflated_sharpe():
    assert expected_max_sharpe(100, 0.25) > expected_max_sharpe(5, 0.25) > expected_max_sharpe(1, 0.25) == 0
    r = noise(mu=0.0004)
    assert deflated_sharpe(r, 200, 0.25) < deflated_sharpe(r, 2, 0.25) <= probabilistic_sharpe(r)


def test_walk_forward_exposes_overfitting_on_a_zero_edge_market():
    px = random_walk_prices()

    factory = equal_weight(ma_crossover)
    grid = {"fast": [5, 10, 20, 30], "slow": [60, 100, 150, 200]}
    wf = walk_forward(px, factory, grid, train_days=504, test_days=126)
    # picking the best-of-16 in sample must look better than it trades out of sample
    assert wf.is_sharpe > wf.oos_sharpe
    assert wf.is_sharpe > 0.3
    # stitched OOS stream: contiguous, non-overlapping, exactly the test windows
    assert len(wf.oos_returns) == len(wf.folds) * 126
    assert wf.oos_returns.index.is_unique and wf.oos_returns.index.is_monotonic_increasing
    # and the strategy has no true edge here
    ci = bootstrap_sharpe_ci(wf.oos_returns, n_boot=500)
    assert ci["lo"] < 0 < ci["hi"]


def test_walk_forward_rejects_short_history_and_supports_anchored():
    px = random_walk_prices(n=400)
    f = equal_weight(ma_crossover)
    with pytest.raises(ValueError):
        walk_forward(px, f, {"fast": [5], "slow": [20]}, train_days=504, test_days=126)
    wf = walk_forward(random_walk_prices(n=900), f, {"fast": [5, 10], "slow": [30, 60]},
                      train_days=300, test_days=100, anchored=True)
    assert len(wf.folds) == 6


def test_sensitivity_grid_reuses_trial_sharpes():
    px = random_walk_prices(n=800)
    f = equal_weight(ma_crossover)
    wf = walk_forward(px, f, {"fast": [10, 30], "slow": [60, 120]}, train_days=300, test_days=100)
    g = sensitivity_grid(sharpes(wf.trials), x="slow", y="fast")
    assert g.shape == (2, 2) and g.index.name == "fast" and g.columns.name == "slow"
    key = (("fast", 10), ("slow", 120))
    assert g.loc[10, 120] == pytest.approx(sharpes(wf.trials)[key])


def test_hac_matches_ols_coefficients_but_widens_errors_under_autocorrelation():
    rng = np.random.default_rng(0)
    n = 3000
    x = pd.Series(np.zeros(n))
    e = np.zeros(n)
    for t in range(1, n):  # persistent regressor and persistent errors
        x.iloc[t] = 0.9 * x.iloc[t - 1] + rng.normal()
        e[t] = 0.7 * e[t - 1] + rng.normal()
    y = pd.Series(2.0 * x.to_numpy() + e)
    plain, hac = ols(y, x.to_frame("x")), ols_hac(y, x.to_frame("x"))
    assert hac["coef"]["x"] == pytest.approx(plain["coef"]["x"])
    assert abs(hac["tstat"]["x"]) < abs(plain["tstat"]["x"])


def test_factor_regression_recovers_alpha_and_beta():
    rng = np.random.default_rng(4)
    n = 4000
    idx = pd.bdate_range("2005-01-01", periods=n)
    mkt = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    rf = pd.Series(0.0001, index=idx)
    strat = rf + 0.0004 + 0.6 * mkt + rng.normal(0, 0.004, n)
    res = factor_regression(strat, pd.DataFrame({"MKT": mkt, "RF": rf}))
    assert res["betas"]["MKT"] == pytest.approx(0.6, abs=0.05)
    assert res["alpha_ann"] == pytest.approx(0.0004 * 252, abs=0.05)
    assert res["alpha_t"] > 2 and res["r2"] > 0.6


def test_beta_hedge_removes_market_correlation():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    mkt = pd.Series(rng.normal(0, 0.01, 1500), index=idx)
    strat = 1.2 * mkt + pd.Series(rng.normal(0, 0.003, 1500), index=idx)
    hedged = beta_hedge(strat, mkt)
    assert abs(hedged.corr(mkt.loc[hedged.index])) < 0.1 < abs(strat.corr(mkt))
