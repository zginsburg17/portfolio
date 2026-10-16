import numpy as np
import pandas as pd
import pytest

import quantfolio.inputs as inputs_mod
from quantfolio.cli import main
from quantfolio.inputs import Config, Inputs
from quantfolio.report import markdown_table
from quantfolio.writeup import BETA_FLAG, build_report, verdict

BASE = {"oos_sharpe": 1.0, "is_sharpe": 1.1, "oos_ci_lo": 0.2, "oos_ci_hi": 1.8,
        "psr": 0.99, "dsr": 0.97, "realized_beta": 0.02}


def test_verdict_has_three_distinct_conclusions():
    strong = verdict(BASE)
    assert "distinguishable from luck after accounting" in strong
    suggestive = verdict({**BASE, "oos_ci_lo": -0.1, "dsr": 0.80})
    assert "does not survive correction" in suggestive and "distinguishable from luck after" not in suggestive
    weak = verdict({**BASE, "oos_sharpe": 0.1, "oos_ci_lo": -0.6, "psr": 0.6, "dsr": 0.4})
    assert "cannot be distinguished from luck" in weak


def test_verdict_never_overclaims():
    # a positive Sharpe with a wide interval and low DSR must not be called an edge
    v = verdict({**BASE, "oos_sharpe": 2.0, "oos_ci_lo": -0.5, "psr": 0.9, "dsr": 0.5})
    assert "edge" not in v and "cannot be distinguished" in v


def test_verdict_flags_overfitting_and_residual_beta():
    v = verdict({**BASE, "is_sharpe": 2.5, "realized_beta": BETA_FLAG + 0.05})
    assert "overstated performance" in v and "did not fully hold" in v
    assert "overstated" not in verdict(BASE) and "did not fully hold" not in verdict(BASE)
    assert "not enough history" in verdict({"realized_beta": 0.0})


def test_markdown_table_formats_percent_and_count_columns():
    df = pd.DataFrame({"CAGR": [0.1234], "Sharpe": [1.239], "rebalances": [84.0]}, index=["a"])
    md = markdown_table(df)
    assert "12.3%" in md and "1.24" in md and "| 84 |" in md
    assert md.splitlines()[1].startswith("|---|---:|")


def test_synthetic_report_is_labelled_and_complete():
    text = build_report(Inputs(Config()), methodology_link="../docs/METHODOLOGY.md")
    assert "GENERATED ON SYNTHETIC DATA" in text and "Survivorship bias" not in text
    for section in ("## Summary", "## Data", "## Results", "### Constraint check",
                    "## Attribution", "## Validation", "## Limitations", "## Reproduce"):
        assert section in text
    assert text.index("## Summary") < text.index("## Data")  # headline first
    assert "(../docs/METHODOLOGY.md)" in text and "nan" not in text.lower().split("## limitations")[0]


def test_live_report_drops_synthetic_banner_and_warns_about_survivorship(monkeypatch):
    tickers = ("AAPL", "MSFT", "NVDA", "JPM", "BAC", "WFC", "XOM", "CVX", "COP", "JNJ", "UNH", "PFE")
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2012-01-02", periods=1800)
    px = pd.DataFrame(100 * np.exp(rng.normal(0, 0.012, (1800, len(tickers))).cumsum(0)),
                      index=idx, columns=tickers)
    fac = lambda cols: pd.DataFrame(rng.normal(0, 0.01, (1800, len(cols))), index=idx, columns=cols)
    monkeypatch.setattr(inputs_mod, "fetch_prices", lambda *a, **k: px)
    monkeypatch.setattr(inputs_mod, "fetch_factors", lambda kind, *a, **k: fac(
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"] if kind == "ff5" else ["Mom"]))
    text = build_report(Inputs(Config(source="yahoo", tickers=tickers)))
    assert "synthetic" not in text.lower()
    assert "Survivorship bias" in text and "Yahoo Finance" in text
    assert "Fama-French 5 factors + momentum" in text
    assert "12 stocks in 4 sectors" in text


def test_cli_writes_report_with_relative_methodology_link(tmp_path, capsys):
    out = tmp_path / "nested" / "report.md"
    main(["--report", str(out)])
    assert out.exists() and "Report written" in capsys.readouterr().out
    text = out.read_text()
    assert "METHODOLOGY.md" in text and "../" in text  # resolved relative to where the report lives
