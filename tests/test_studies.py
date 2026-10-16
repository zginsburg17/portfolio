import pandas as pd
import pytest

import quantfolio.inputs as inputs_mod
from quantfolio.cli import main
from quantfolio.datasources import DataSourceError
from quantfolio.inputs import Config, Inputs
from quantfolio.report import render
from quantfolio.results import StudyResult
from quantfolio.studies import STUDIES


@pytest.fixture(scope="module")
def synth():
    return Inputs(Config())


@pytest.mark.parametrize("name", list(STUDIES))
def test_every_study_returns_structured_results_and_renders(synth, name):
    res = STUDIES[name](synth)
    assert isinstance(res, StudyResult) and res.sections
    text = render(res)
    assert res.title in text and "nan" not in text.lower().replace("nanosecond", "")


def test_study_results_are_machine_readable(synth):
    v = STUDIES["validation"](synth)
    assert {"is_sharpe", "oos_sharpe", "psr", "dsr"} <= set(v.metrics)
    assert v.table("Sharpe sensitivity (rows = fast MA, cols = slow MA)").shape == (5, 5)
    assert STUDIES["pairs"](synth).metrics["stationary"] == 1.0  # synthetic pair is cointegrated
    with pytest.raises(KeyError):
        v.table("no such section")


def test_inputs_are_memoized(synth):
    assert synth.prices is synth.prices and synth.backtests is synth.backtests
    assert list(synth.strategy_returns.columns) == list(synth.backtests)


def test_close_only_csv_cannot_silently_fall_back_to_synthetic_ohlc(tmp_path):
    csv = tmp_path / "px.csv"
    synth_prices = Inputs(Config()).prices[["ASSET01", "ASSET02"]]
    synth_prices.index.name = "Date"
    synth_prices.to_csv(csv)
    inp = Inputs(Config(csv=str(csv), pair=("ASSET01", "ASSET02")))
    assert list(inp.pair.columns) == ["ASSET01", "ASSET02"]  # pair comes from the CSV, not synthetic
    with pytest.raises(DataSourceError, match="OHLC"):
        inp.ohlc
    with pytest.raises(DataSourceError, match="not found"):
        Inputs(Config(csv=str(csv), pair=("KO", "PEP"))).pair


def test_live_source_routes_every_input_through_providers(monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=300)
    calls = []

    def fake_prices(tickers, *a, **kw):
        calls.append(("prices", tuple(tickers)))
        return pd.DataFrame({t: 100.0 + i + pd.Series(range(300), index=idx) * 0.1
                             for i, t in enumerate(tickers)})

    def fake_bars(ticker, *a, **kw):
        calls.append(("bars", ticker))
        close = pd.Series(range(100, 400), index=idx, dtype=float)
        return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                             "adj_close": close / 2, "volume": 1.0})  # a 2:1 adjustment

    def fake_factors(kind, *a, **kw):
        calls.append(("factors", kind))
        cols = {"ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], "mom": ["Mom"]}[kind]
        return pd.DataFrame(0.001, index=idx, columns=cols)

    monkeypatch.setattr(inputs_mod, "fetch_prices", fake_prices)
    monkeypatch.setattr(inputs_mod, "fetch_bars", fake_bars)
    monkeypatch.setattr(inputs_mod, "fetch_factors", fake_factors)
    inp = Inputs(Config(source="yahoo", tickers=("AAA", "BBB", "CCC")))
    assert inp.cfg.live
    assert list(inp.prices.columns) == ["AAA", "BBB", "CCC"]
    assert list(inp.pair.columns) == ["KO", "PEP"]
    ohlc = inp.ohlc
    assert ohlc["close"].iloc[0] == 50.0  # close becomes adj_close
    assert ohlc["high"].iloc[0] == 50.5  # high rescaled by adj_close/close, no fake split gap
    fac, rf, note = inp.factors
    assert rf == "RF" and "Mom" in fac and "Fama-French" in note
    assert {c[0] for c in calls} == {"prices", "bars", "factors"}


def test_csv_wins_over_source_and_is_not_live():
    assert not Config(source="yahoo", csv="x.csv").live and Config(source="yahoo").live


def test_cli_runs_a_study_and_reports_data_errors(capsys):
    main(["--study", "vol"])
    assert "VOL:" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["--source", "yahoo", "--offline", "--tickers", "ZZZ", "--study", "trend"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        main(["--pair", "ONLYONE", "--study", "pairs"])
