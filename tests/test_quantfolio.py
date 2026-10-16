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
