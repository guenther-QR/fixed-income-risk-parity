"""FI_26 review document."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro.backtest import speclog  # noqa: E402
from macro.report import charts  # noqa: E402
from macro.report.builder import PhaseReport  # noqa: E402

P = ROOT / "data/processed"


def main() -> int:
    T = pd.read_parquet(P / "fi_predictability_risk.parquet")
    dev = pd.read_parquet(P / "fi_dev_table.parquet")
    oos = pd.read_parquet(P / "fi_oos_table.parquet")
    cmp = pd.read_parquet(P / "fi_rank_comparison.parquet")
    B = pd.read_parquet(P / "fi_bootstrap.parquet")
    spa = pd.read_parquet(P / "fi_spa.parquet")["value"]
    stats = pd.read_parquet(P / "fi_asset_stats.parquet")
    nets = pd.read_parquet(P / "fi_all_strategies.parquet")
    rho = cmp["dev_sharpe"].corr(cmp["oos_sharpe"], method="spearman")
    sp_dur = st.spearmanr(T["oos_r2"], T["duration"])

    r = PhaseReport(
        phase="FI_26", title="A Fixed Income Only Allocation Engine",
        summary=("Macro_26 found that macro signals forecast rates and credit far "
                 "better than equities, and that a multi-asset portfolio could not "
                 "use it because the forecastable assets carried none of the risk. "
                 "This project removes equities and asks whether the edge becomes "
                 "usable. It does not - and the reason turns out to be a property "
                 "of fixed income itself, measurable at a rank correlation of "
                 "-0.96."),
        status="complete")

    r.metrics([
        ("12", "assets, 1982-2026", None),
        ("9 of 12", "forecastable out of sample", "pass"),
        ("-0.96", "R2 vs duration rank corr.", "fail"),
        ("0.026", "Hansen SPA p", "pass"),
        ("+0.45", "dev-to-holdout rank corr.", "pass"),
    ])

    # ---------------------------------------------------------------- 1
    r.section("1. The universe", (
        "Twelve assets in three groups - Treasury constant-maturity holdings "
        "across the curve, investment grade and high yield credit, and "
        "securitized municipals and mortgages. The curve legs are built from the "
        "same bootstrapped zero curve as Macro_26, so they differ only in "
        "duration; the rest are total-return funds chosen for reaching 1982."))
    s = stats[["group", "cagr", "vol", "sharpe", "skew", "corr_ust10y"]]
    s.index.name = "asset"
    r.table(s.round(4), align_right=["cagr", "vol", "sharpe", "skew", "corr_ust10y"])
    r.prose(
        "The first principal component explains 65.5% of excess-return variance "
        "and the second 13.9%. That is a universe with real but limited internal "
        "structure: high yield is the only leg that decouples, correlating 0.18 "
        "with the ten-year against 0.9-plus for everything else.")

    # ---------------------------------------------------------------- 2
    r.section("2. The finding: predictability is inversely related to risk", (
        "Nine of twelve assets are forecastable out of sample - a better hit rate "
        "than Macro_26 achieved. But which nine is the whole story."))
    t = T[["oos_r2", "vol", "duration", "risk_share"]].copy()
    t.columns = ["OOS R2", "volatility", "duration (yrs)", "share of variance"]
    t.index.name = "asset"
    r.table(t.round(4), align_right=list(t.columns))

    fig, ax = charts.new_axes(8.0, 3.6)
    ax.scatter(T["duration"], T["oos_r2"] * 100, s=70,
               color=charts.SERIES[0], zorder=3)
    for a, row in T.iterrows():
        ax.annotate(a, (row["duration"], row["oos_r2"] * 100),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=charts.MUTED)
    ax.axhline(0, color=charts.MUTED, linewidth=1.0, linestyle="--")
    ax.set_xlabel("modified duration (years)")
    ax.set_ylabel("out-of-sample R-squared (%)")
    r.figure(charts.to_svg(fig),
             f"Spearman rank correlation {sp_dur.statistic:+.3f} "
             f"(p = {sp_dur.pvalue:.4f}). Every asset with duration above five "
             f"years is unforecastable.")

    r.prose(
        f"The rank correlation between out-of-sample R-squared and duration is "
        f"<strong>{sp_dur.statistic:+.3f}</strong> with p = {sp_dur.pvalue:.4f}; "
        f"against volatility it is {st.spearmanr(T['oos_r2'], T['vol']).statistic:+.3f}. "
        f"The four most forecastable assets hold "
        f"<strong>{T.nlargest(4, 'oos_r2')['risk_share'].sum():.1%}</strong> of "
        f"universe variance; the four least forecastable hold "
        f"<strong>{T.nsmallest(4, 'oos_r2')['risk_share'].sum():.1%}</strong>.")
    r.prose(
        "Macro_26 reached the same conclusion with equities in the book, and the "
        "obvious objection was that equities were doing the work - remove them and "
        "the mismatch goes away. It does not. Finding the relationship again, "
        "stronger, inside a universe built entirely from forecastable assets means "
        "it is a property of the assets rather than of the earlier universe. "
        "Short-dated instruments are predictable because their returns are "
        "dominated by carry, which is known in advance; long-dated instruments are "
        "unpredictable because their returns are dominated by yield changes, which "
        "are not. The same fact makes the first group low-risk and the second "
        "high-risk. Predictability and risk are two readings of one quantity.")

    # ---------------------------------------------------------------- 3
    r.section("3. Allocation against 1/N", (
        "1/N is a hard benchmark. DeMiguel, Garlappi and Uppal (2009) found no "
        "optimising model beat it out of sample across fourteen datasets, because "
        "estimation error swamps the optimisation gain."))
    d = dev[["cagr", "vol", "sharpe", "vs_1N", "max_drawdown"]]
    d.index.name = "strategy"
    r.table(d.round(4), align_right=list(d.columns),
            caption="Development sample, net of transaction costs.")

    r.prose(
        "On unlevered Sharpe the risk-based rules win clearly, and unlike anything "
        "in Macro_26 the margin survives inference: paired block bootstrap "
        "intervals exclude zero for risk parity (+0.132, CI [0.056, 0.216]), "
        "maximum diversification (+0.120) and inverse volatility (+0.099), and "
        f"Hansen's SPA over the family returns p = {float(spa.get('p_spa')):.3f}. "
        "That is a genuine result and it is the first one in either project to "
        "clear a multiple-testing correction.")
    b = B[["difference", "ci_lo", "ci_hi", "p_one_sided"]]
    b.index.name = "strategy"
    r.table(b.round(4), align_right=list(b.columns))

    r.prose(
        "It does not survive being paid for. Risk parity reaches its Sharpe at "
        "1.65% volatility by holding 56% of the three-month bill; minimum variance "
        "holds 87% of it. Levering either to 1/N's 4% risk level costs more than "
        "the edge is worth:")

    r.section("4. What happens once leverage is charged")
    r.prose(
        "At a common 4% risk target with financing at 50bp over the risk-free "
        "rate, the development ranking inverts. 1/N moves from fifth on unlevered "
        "Sharpe to first on levered Sharpe (0.8047 against risk parity's 0.7467), "
        "because it already runs at roughly the target and borrows nothing. The "
        "risk-based rules need 1.8x to 4.4x and pay for all of it.")
    r.prose(
        "This is the same lesson the project owner identified in Macro_26 - that "
        "comparing Sharpe ratios across portfolios of different volatility is only "
        "fair if the levering is priced - and it is worth more here than there, "
        "because fixed income volatilities differ by a factor of fifteen across "
        "the universe.")

    # ---------------------------------------------------------------- 5
    r.section("5. The holdout")
    o = oos[["cagr", "vol", "sharpe", "vs_1N", "max_drawdown"]]
    o.index.name = "strategy"
    r.table(o.round(4), align_right=list(o.columns),
            caption="2016-01 to 2026-08, opened once after the inference above.")

    fig, ax = charts.new_axes(9.0, 3.4)
    sub = nets.loc["2016-01":]
    order = oos["sharpe"].sort_values(ascending=False).index[:5]
    for i, name in enumerate(order):
        ax.plot(sub.index, (1 + sub[name]).cumprod(),
                color=charts.SERIES[i % len(charts.SERIES)],
                linewidth=2.0 if name == "1/N monthly" else 1.2, label=name)
    ax.set_ylabel("growth of 1")
    charts.legend(ax, loc="upper left")
    r.figure(charts.to_svg(fig),
             "The holdout contains the 2022 rate shock, which is why every "
             "strategy posts a Sharpe near zero.")

    r.prose(
        f"Rank correlation between development and holdout is "
        f"<strong>{rho:+.3f}</strong> - positive, and a marked improvement on "
        f"Macro_26's -0.333. The risk-based ordering broadly held. But the decade "
        f"was hostile to the whole asset class: 1/N returned 2.2% a year at a "
        f"Sharpe of 0.03, and the 2022 rate shock produced a 15.9% drawdown in "
        f"what is nominally a defensive book.")
    r.prose(
        "Charged for leverage on the holdout, 1/N annual is again first. Minimum "
        "variance, which looks respectable unlevered at 0.081, needs 4.5x to reach "
        "4% volatility and delivers -0.456 after paying for it.")

    # ---------------------------------------------------------------- 6
    r.section("6. Conclusion")
    r.prose(
        "A dynamic allocation engine does not beat 1/N in fixed income once "
        "leverage is priced. Risk-based rules beat it on raw Sharpe and the margin "
        "is statistically real, but the margin is entirely duration reduction "
        "wearing a diversification costume - an investor who wants less duration "
        "can simply hold less duration, without an optimiser. Forecast, carry and "
        "momentum tilts lose in both samples.")
    r.prose(
        "The reason is the same one Macro_26 found and this project measured "
        "precisely. Fixed income returns are forecastable in inverse proportion to "
        "how much risk they carry, at a rank correlation of -0.96 against duration. "
        "A forecast can only move a portfolio through the assets it forecasts, and "
        "those are structurally the assets that do not move portfolios.")
    r.prose(
        "That is a stronger and more general statement than either project set out "
        "to make, and it is the one worth carrying forward: the search for "
        "predictability and the search for return are looking in opposite "
        "directions.")

    r.checks([
        (True, "Nine of twelve assets forecastable out of sample",
         "ust3m +3.4%, ust2y +1.8%, hy +1.6%"),
        (True, "Risk-based rules beat 1/N on unlevered Sharpe",
         f"Hansen SPA p = {float(spa.get('p_spa')):.3f}, intervals exclude zero"),
        (False, "Any strategy beats 1/N once leverage is charged",
         "1/N first in both samples at a 4% risk target"),
        (False, "Forecast tilts beat 1/N",
         "-0.069 development, -0.021 holdout"),
        (True, "Development ranking survived into the holdout",
         f"rank correlation {rho:+.3f}"),
    ])

    out = r.render(ROOT / "reports/fi26.html")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
