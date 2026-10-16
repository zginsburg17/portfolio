import numpy as np
import pandas as pd
import pytest

from quantfolio.backtest import backtest
from quantfolio.data import synthetic_prices
from quantfolio.metrics import cagr, drawdown, performance_summary, sharpe
from quantfolio.portfolio import combine_inverse_vol
from quantfolio.strategies import STRATEGIES, sma_crossover


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(n_assets=5, n_days=800, seed=1)


def test_synthetic_data_is_reproducible_and_positive(prices):
    again = synthetic_prices(n_assets=5, n_days=800, seed=1)
    pd.testing.assert_frame_equal(prices, again)
    assert (prices > 0).all().all()


def test_no_lookahead_bias(prices):
    """A signal that is today's return sign must NOT earn today's return."""
    cheat = np.sign(prices.pct_change()).fillna(0.0)
    res = backtest(prices, cheat, cost_bps=0, max_gross=None)
    # With a proper 1-day lag, the cheat has no edge: Sharpe stays modest.
    assert abs(sharpe(res.returns)) < 1.5
    # And day-0 holds nothing.
    assert (res.positions.iloc[0] == 0).all()


def test_costs_reduce_returns(prices):
    pos = sma_crossover(prices, 10, 50)
    free = backtest(prices, pos, cost_bps=0)
    costly = backtest(prices, pos, cost_bps=25)
    assert costly.returns.sum() < free.returns.sum()
    assert costly.costs.sum() > 0
    np.testing.assert_allclose(free.returns, free.gross_returns)


def test_constant_long_matches_buy_and_hold():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({"A": [100, 110, 121, 121, 133.1]}, index=idx)
    res = backtest(px, pd.DataFrame(1.0, index=idx, columns=["A"]), cost_bps=0)
    # Held from day 1 onward => captures all returns after the first bar.
    assert (1 + res.returns).prod() == pytest.approx(133.1 / 100)


def test_gross_exposure_is_capped(prices):
    big = pd.DataFrame(5.0, index=prices.index, columns=prices.columns)
    res = backtest(prices, big, max_gross=2.0)
    assert res.positions.abs().sum(axis=1).max() <= 2.0 + 1e-9


def test_metrics_known_values():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    dd = drawdown(r)
    assert dd.min() == pytest.approx(-0.02)
    flat = pd.Series([0.0] * 10)
    assert sharpe(flat) == 0.0
    assert cagr(pd.Series([0.0] * 252)) == pytest.approx(0.0)
    assert set(performance_summary(r)) >= {"CAGR", "Sharpe", "Max DD", "Calmar"}


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_shape_and_finite(prices, name):
    pos = STRATEGIES[name](prices)
    assert pos.shape == prices.shape
    assert np.isfinite(pos.to_numpy()).all()


def test_xs_momentum_is_dollar_neutral(prices):
    pos = STRATEGIES["xs_momentum"](prices).iloc[200:]
    assert pos.sum(axis=1).abs().max() < 1e-9


def test_sma_rejects_bad_windows(prices):
    with pytest.raises(ValueError):
        sma_crossover(prices, fast=50, slow=10)


def test_portfolio_hits_target_vol_roughly(prices):
    rets = pd.DataFrame({n: backtest(prices, f(prices)).returns for n, f in STRATEGIES.items()})
    port = combine_inverse_vol(rets, target_vol=0.10)
    assert 0.04 < port.std() * np.sqrt(252) < 0.25


# ---- technique building blocks (trend / pairs / countertrend / risk / vol) ----
from quantfolio.countertrend import countertrend, market_vs_strategy, sharpe_of, sweep_p
from quantfolio.data import synthetic_ohlc, synthetic_pair
from quantfolio.portfolio import combine_equal
from quantfolio.stats import adf_test, ols, quintile_table, zscore_trailing
from quantfolio.strategies import breakout, ma_crossover, pairs_spread


def test_zscore_excludes_today():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    z = zscore_trailing(x, 4)
    # today's 100 is compared with mean/std of the prior four values only
    prior = x.iloc[:4]
    assert z.iloc[-1] == pytest.approx((100 - prior.mean()) / prior.std())


def test_adf_detects_stationary_vs_random_walk():
    rng = np.random.default_rng(0)
    walk = pd.Series(np.cumsum(rng.normal(size=1500)))
    ou = pd.Series(np.zeros(1500))
    for t in range(1, 1500):
        ou.iloc[t] = 0.8 * ou.iloc[t - 1] + rng.normal()
    assert adf_test(ou)["stationary_5pct"]
    assert not adf_test(walk)["stationary_5pct"]


def test_cointegrated_pair_difference_is_stationary():
    pair = synthetic_pair()
    assert adf_test(pair["WHITE"] - pair["BLACK"])["stationary_5pct"]


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=500)})
    y = 2.0 + 3.0 * X["a"] + rng.normal(scale=0.1, size=500)
    fit = ols(y, X)
    assert fit["coef"]["a"] == pytest.approx(3.0, abs=0.05)
    assert fit["r2"] > 0.99


def test_ma_crossover_signs_and_ema():
    idx = pd.bdate_range("2020-01-01", periods=200)
    up = pd.DataFrame({"A": np.linspace(100, 200, 200)}, index=idx)
    assert (ma_crossover(up, 10, 30).iloc[40:] == 1).all().all()
    assert (ma_crossover(up, 10, 30).iloc[:29] == 0).all().all()
    assert (ma_crossover(up[::-1].set_axis(idx), 10, 30, "ema").iloc[40:] == -1).all().all()


def test_breakout_holds_until_opposite_break():
    px = pd.DataFrame({"A": [10, 11, 12, 13, 12.5, 12.4, 5, 6]}, dtype=float)
    sig = breakout(px, 3)["A"].tolist()
    assert sig == [0, 0, 0, 1, 1, 1, -1, -1]


def test_pairs_positions_are_opposed_and_respect_hold():
    pair = synthetic_pair()
    pos = pairs_spread(pair, lookback=5, hold=5)
    nz = pos.dropna()
    assert (np.sign(nz["WHITE"]) == -np.sign(nz["BLACK"])).all()
    # a longer holding period keeps capital deployed for more days
    short = (pairs_spread(pair, hold=1, target=99)["WHITE"] != 0).mean()
    long_ = (pairs_spread(pair, hold=10, target=99)["WHITE"] != 0).mean()
    assert long_ > short
    with pytest.raises(ValueError):
        pairs_spread(pair.assign(C=1.0))


def test_countertrend_fill_logic():
    idx = pd.bdate_range("2020-01-01", periods=30)
    o = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0}, index=idx)
    o.loc[idx[25], ["open", "low", "close"]] = [100.0, 90.0, 99.0]  # dips through L
    r = countertrend(o, p=1.5, window=20)  # A=4, L=102-6=96 -> fill at 96
    assert r.notna().sum() == 1
    assert r.iloc[25] == pytest.approx(99 / 96 - 1)
    gap = o.copy()
    gap.loc[idx[25], "open"] = 95.0  # gap below L: fills at the open, not L
    assert countertrend(gap, 1.5).iloc[25] == pytest.approx(99 / 95 - 1)


def test_countertrend_signal_count_grows_as_p_falls():
    ohlc = synthetic_ohlc(n_days=1500)
    s = sweep_p(ohlc)  # P runs 2.2 -> 0.8, so signals must grow down the table
    assert s["signals"].is_monotonic_increasing
    r = countertrend(ohlc, 1.0)
    assert sharpe_of(r, "blank") != sharpe_of(r, "zero")
    mx = market_vs_strategy(ohlc, r)
    assert r.notna().sum() - 1 <= mx["days"].sum() <= r.notna().sum()


def test_quintile_table_buckets_are_equal_and_ordered():
    x = pd.Series(np.arange(100.0))
    t = quintile_table(x, {"y": -x})
    assert list(t["count"]) == [20] * 5
    assert t["x"].is_monotonic_increasing and t["y"].is_monotonic_decreasing


def test_equal_weight_combination_is_mean():
    df = pd.DataFrame({"a": [0.01, 0.02], "b": [0.03, 0.0]})
    assert combine_equal(df).tolist() == pytest.approx([0.02, 0.01])
