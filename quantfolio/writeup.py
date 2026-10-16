"""Generate the research report (markdown) from an actual run.

Every number comes from `studies.neutral_study`; the verdict is worded from the
validation statistics by fixed rules, so the report cannot claim more than the
evidence supports. Synthetic data is labelled as such, prominently.
"""

from __future__ import annotations

import subprocess
from datetime import date

from .inputs import Inputs
from .report import markdown_table
from .results import StudyResult
from .studies import STUDIES

# Decision rules for the verdict (explicit so they can be argued with).
DSR_BAR = 0.95
PSR_BAR = 0.95
BETA_FLAG = 0.10


def verdict(m: dict[str, float]) -> str:
    """Conservative plain-language reading of the validation metrics."""
    if "oos_sharpe" not in m:
        return "There was not enough history to run walk-forward validation, so no conclusion is drawn."
    lines = []
    if m["oos_ci_lo"] > 0 and m["dsr"] >= DSR_BAR:
        lines.append(
            f"The out-of-sample Sharpe ({m['oos_sharpe']:.2f}) is positive with a bootstrap interval "
            f"above zero and a deflated Sharpe of {m['dsr']:.0%}, so the result is distinguishable "
            "from luck after accounting for the number of variants tried. This is evidence of an edge "
            "in this sample, not proof of one in future data.")
    elif m["oos_sharpe"] > 0 and m["psr"] >= PSR_BAR:
        lines.append(
            f"The out-of-sample Sharpe ({m['oos_sharpe']:.2f}) is positive and individually "
            f"significant (probabilistic Sharpe {m['psr']:.0%}), but the deflated Sharpe "
            f"({m['dsr']:.0%}) does not clear {DSR_BAR:.0%}: it does not survive correction for "
            "having tried several parameter sets. Treat it as suggestive.")
    else:
        lines.append(
            f"The out-of-sample Sharpe is {m['oos_sharpe']:.2f} with a 95% bootstrap interval of "
            f"[{m['oos_ci_lo']:.2f}, {m['oos_ci_hi']:.2f}]. The evidence cannot be distinguished "
            "from luck.")
    if m["is_sharpe"] > m["oos_sharpe"] + 0.3:
        lines.append(f"In-sample Sharpe ({m['is_sharpe']:.2f}) exceeds out-of-sample "
                     f"({m['oos_sharpe']:.2f}): selecting parameters in-sample overstated performance.")
    if abs(m["realized_beta"]) > BETA_FLAG:
        lines.append(f"Realized market beta is {m['realized_beta']:.2f}, above the {BETA_FLAG:.2f} "
                     "tolerance, so the neutralization did not fully hold out of sample.")
    return " ".join(lines)


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_report(inp: Inputs, command: str = "python -m quantfolio --study neutral",
                 methodology_link: str | None = None) -> str:
    cfg, px = inp.cfg, inp.prices
    res: StudyResult = STUDIES["neutral"](inp)
    live = cfg.live or bool(cfg.csv)
    source = ("public data via Yahoo Finance (split/dividend-adjusted closes)" if cfg.live
              else f"CSV file `{cfg.csv}`" if cfg.csv else "**synthetic, simulated data**")

    out = ["# Sector- and beta-neutral cross-sectional equity strategy: research report", ""]
    if not live:
        out += ["> **THIS REPORT WAS GENERATED ON SYNTHETIC DATA.** It demonstrates that the "
                "pipeline and its safeguards work. It is not evidence about any market. "
                "Regenerate with `--source yahoo` for a real result.", ""]
    out += [f"Generated {date.today().isoformat()} at commit `{_commit()}`.", "",
            "## Summary", "", verdict(res.metrics), "",
            "## Data", "",
            f"- Source: {source}",
            f"- Universe: {px.shape[1]} stocks in {inp.sectors.nunique()} sectors, "
            f"{len(px)} trading days ({px.index[0].date()} to {px.index[-1].date()})",
            f"- Costs: {cfg.cost_bps:g} bps per unit of one-way turnover", ""]
    if cfg.live:
        out += ["- **Survivorship bias:** the universe is a present-day snapshot of large, "
                "successful companies. Firms that failed or shrank are absent, which flatters "
                "long-horizon results, especially for momentum. Read every number below as an "
                "upper bound.", ""]

    out += ["## Strategy", "",
            "Stocks are scored on 12-1 month momentum plus a one-month reversal term, each "
            "cross-sectionally z-scored. Scores are demeaned within sectors, then weights are "
            "projected to zero ex-ante market beta without disturbing sector neutrality, scaled to "
            "unit gross, and rebalanced every 21 trading days. "
            + (f"Full details and the reasoning behind each choice are in "
               f"[the methodology note]({methodology_link})." if methodology_link else ""), "",
            "## Results", "",
            "Net of costs, with market beta and alpha measured against the universe's "
            "equal-weight return (Newey-West t-statistics).", "",
            markdown_table(res.table("Variants")), "",
            "The raw cross-sectional momentum row is the naive baseline; the neutral rows show what "
            "remains once sector and market bets are removed.", "",
            "### Constraint check", "",
            "Measured on actual rebalance dates. Gross should be 1; net, per-sector net and "
            "beta should be zero.", "",
            markdown_table(res.table("Constraint check"), nd=4), "",
            "## Attribution", "",
            markdown_table(res.table("Factor attribution")), ""]
    note = [s.heading for s in res.sections if s.heading.startswith("Factor attribution")][0]
    out += [f"_{note}_", "", "## Validation", "",
            "Parameters (lookback, reversal weight) are chosen on a trailing three-year window "
            "and traded on the next unseen six months, rolling forward.", ""]
    try:
        out += [res.text("Walk-forward").replace("\n", "  \n"), ""]
    except KeyError:
        out += [res.text("Validation"), ""]
    n = int(res.metrics.get("n_trials", 0))
    out += ["## Limitations", "",
            "- **Execution is idealized:** a signal computed from day t's close is assumed filled "
            "at that close and earns day t+1's return, with flat proportional costs. There is no "
            "market impact, no borrow cost or short-availability constraint, and no financing "
            "spread.",
            f"- **Small search, but a search:** {n} parameter sets were tried. The deflated Sharpe "
            "accounts for that; it cannot account for variants tried and discarded before this "
            "report.",
            "- **One market, one regime sample:** a single universe over one history. Stability "
            "across sub-periods and markets was not tested.",
            "- **Beta is estimated, not known:** rolling regressions on an equal-weight market "
            "proxy; true realized beta drifts between rebalances.", "",
            "## Reproduce", "",
            "```bash", command, "python -m quantfolio --report output/report.md   # this document",
            "```", ""]
    return "\n".join(out)
