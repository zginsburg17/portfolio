import numpy as np
import pandas as pd
import pytest

import quantfolio.inputs as inputs_mod
from quantfolio.attribution import factor_regression
from quantfolio.backtest import backtest
from quantfolio.datasources import DataSourceError
from quantfolio.datasources.universe import SECTOR_OF
from quantfolio.inputs import Config, Inputs
from quantfolio.neutral import exposure_check, neutral_weights, rolling_betas


def make_market(n=1800, n_assets=12, seed=0, drift_scale=0.0, market_drift=None):
    """Assets with heterogeneous betas, 4 sectors of 3. `drift_scale` plants a fixed
    per-asset drift (a real, persistent cross-sectional signal)."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0, 0.01, n)
    if market_drift is not None:
        mkt = mkt + market_drift
    betas = rng.uniform(0.4, 1.6, n_assets)
    mu = rng.normal(0, drift_scale, n_assets)
    rets = mkt[:, None] * betas + rng.normal(0, 0.012, (n, n_assets)) + mu
    idx = pd.bdate_range("2012-01-02", periods=n)
    cols = [f"T{i:02d}" for i in range(n_assets)]
    sectors = pd.Series({c: f"S{i // 3}" for i, c in enumerate(cols)})
    return pd.DataFrame(100 * np.exp(rets.cumsum(0)), index=idx, columns=cols), sectors, mu


@pytest.fixture(scope="module")
def market():
    return make_market(seed=1)


def test_constraints_hold_exactly(market):
    px, sec, _ = market
    w = neutral_weights(px, sec)
    chk = exposure_check(w, px, sec)
    assert chk["rebalances"] > 50
    assert chk["mean_gross"] == pytest.approx(1.0)
    assert chk["max_abs_net"] < 1e-9 and chk["max_abs_sector_net"] < 1e-9
    assert chk["max_abs_beta"] < 1e-9
    # without the beta step, sector neutrality still holds but beta does not
    only = exposure_check(neutral_weights(px, sec, beta_neutral=False), px, sec)
    assert only["max_abs_sector_net"] < 1e-9 and only["max_abs_beta"] > 1e-3


def test_flat_before_warmup_and_held_between_rebalances(market):
    px, sec, _ = market
    w = neutral_weights(px, sec, rebalance=21)
    assert (w.iloc[:250] == 0).all().all()  # needs lookback + beta window before trading
    changes = w.diff().abs().sum(axis=1) > 1e-12
    assert changes.sum() > 0
    assert all(i % 21 == 0 for i in np.flatnonzero(changes.to_numpy()))  # only rebalance rows


def test_no_lookahead(market):
    px, sec, _ = market
    t0 = 21 * 57  # a rebalance row: weights only change there, so a leak would show up here
    w = neutral_weights(px, sec)
    scrambled = px.copy()
    rng = np.random.default_rng(9)
    scrambled.iloc[t0 + 1:] *= rng.uniform(0.5, 2.0, scrambled.iloc[t0 + 1:].shape)  # rewrite the future
    assert w.iloc[t0].abs().sum() > 0  # the cut lands on a live rebalance, not on warmup zeros
    pd.testing.assert_frame_equal(w.iloc[: t0 + 1], neutral_weights(scrambled, sec).iloc[: t0 + 1])


def test_finds_a_planted_persistent_signal():
    px, sec, mu = make_market(seed=3, drift_scale=0.0015)
    w = neutral_weights(px, sec, w_rev=0.0)
    late = w.iloc[-1]
    assert np.corrcoef(late, mu)[0, 1] > 0.3  # long the high-drift names, short the low-drift
    res = backtest(px, w, cost_bps=5)
    assert res.returns.iloc[300:].mean() > 0


def test_beta_step_removes_realized_market_exposure():
    # a trending market makes momentum load on high-beta names; neutralizing must undo that
    drift = np.r_[np.full(900, 0.002), np.full(900, -0.002)]
    px, sec, _ = make_market(seed=4, market_drift=drift)
    mkt = pd.DataFrame({"MKT": px.pct_change().mean(axis=1)})

    def realized_beta(beta_neutral):
        r = backtest(px, neutral_weights(px, sec, beta_neutral=beta_neutral, w_rev=0.0)).returns
        return factor_regression(r.iloc[260:], mkt.iloc[260:], rf_col=None)["betas"]["MKT"]

    assert abs(realized_beta(True)) < abs(realized_beta(False))
    assert abs(realized_beta(True)) < 0.05


def test_small_sectors_are_dropped_and_missing_sectors_rejected(market):
    px, sec, _ = market
    sec2 = sec.copy()
    sec2["T00"] = "TINY"  # leaves S0 with 2 names and TINY with 1
    w = neutral_weights(px, sec2, min_names=3)
    assert (w[["T00", "T01", "T02"]] == 0).all().all()  # S0 (2 names) and TINY (1) both skipped
    assert (w[[c for c in w if c not in ("T00", "T01", "T02")]].abs().sum(axis=1) > 0).any()
    with pytest.raises(ValueError, match="no sector"):
        neutral_weights(px, sec.drop("T05"))


def test_rolling_betas_recover_true_betas(market):
    px, _, _ = market
    b = rolling_betas(px, 252).iloc[-1]
    assert b.mean() == pytest.approx(1.0, abs=0.1)  # EW-market betas average to 1 by construction
    assert b.std() > 0.2


def test_inputs_sector_mapping(monkeypatch, tmp_path):
    syn = Inputs(Config())
    assert syn.sectors.nunique() == len(syn.prices.columns) // 2

    idx = pd.bdate_range("2020-01-01", periods=300)
    monkeypatch.setattr(inputs_mod, "fetch_prices", lambda tickers, *a, **k: pd.DataFrame(
        {t: np.linspace(100, 120, 300) for t in tickers}, index=idx))
    live = Inputs(Config(source="yahoo", tickers=("AAPL", "JPM", "XOM")))
    assert live.sectors.to_dict() == {t: SECTOR_OF[t] for t in ("AAPL", "JPM", "XOM")}
    with pytest.raises(DataSourceError, match="no sector mapping"):
        Inputs(Config(source="yahoo", tickers=("AAPL", "ZZZZ"))).sectors
