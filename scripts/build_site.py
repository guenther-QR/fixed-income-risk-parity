"""Build the three phase reports and the index for the fixed income project."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro.report import charts  # noqa: E402
from macro.report.builder import PhaseReport  # noqa: E402

P = ROOT / "data/processed"
OUT = ROOT / "reports"
MACRO = ROOT.parent / "Macro_26/data/processed"

URL_MACRO = "https://guenther-qr.github.io/macro-portfolio-rebuild/"

# Every comparison in this project runs from here. It is the first month the
# covariance-based methods can trade after their 60-month estimation window,
# and the benchmarks are held to the same start so nothing gets a head start.
START = "1987-11"


def get(name):
    try:
        return pd.read_parquet(P / f"{name}.parquet")
    except Exception:
        return None


def macro(name):
    try:
        return pd.read_parquet(MACRO / f"{name}.parquet")
    except Exception:
        return None


def three_window(T, cols, labels):
    """Slice the aligned table down to one window's worth of columns."""
    keep = [c for c in cols if c in T.columns]
    t = T[keep].copy()
    t.columns = labels[:len(keep)]
    t.index.name = "strategy"
    return t


# ------------------------------------------------------------------ phase 1

def phase1():
    r = PhaseReport(
        phase="Phase 1", title="The Idea",
        summary=("A prior multi-asset study tested roughly 1,600 "
                 "specifications against a sealed holdout and none beat a 60/40 "
                 "benchmark. One relationship in that work held everywhere it "
                 "was tested, and it pointed at fixed income. This phase sets "
                 "out that evidence and the universe built on it."),
        status="complete")

    fipred = get("fi_predictability_risk")
    stats = get("fi_asset_stats")
    zoo = macro("daily_model_skill")

    r.metrics([
        ("11", "assets in the universe", None),
        ("2.13%", "best bond out of sample R squared", "pass"),
        ("0.05%", "equity out of sample R squared", "fail"),
        ("9 of 12", "fixed income assets forecastable", "pass"),
        ("1987-2026", "sample", None),
    ])

    r.section("Predictive skill by instrument and by model", (
        "The parent project fitted eight model families at daily frequency "
        "across a seven asset universe, five of which are fixed income. Out of "
        "sample R squared is measured against a rolling mean forecast, so a "
        "positive number means the model beats simply predicting the average."))
    if zoo is not None:
        fi_cols = [c for c in ["ig", "hy", "ust2y", "ust10y", "ust30y"]
                   if c in zoo.columns]
        eq_cols = [c for c in ["sp500", "gold"] if c in zoo.columns]
        t = zoo[fi_cols + eq_cols].copy()
        t.index.name = "model family"
        r.table(t.round(5), align_right=list(t.columns),
                caption="Out of sample R squared, daily. Left block is fixed "
                        "income, right block is equity and gold.")
    r.prose(
        "Two patterns run through the table. First, <strong>the fixed income "
        "columns are systematically better than the equity column</strong>. The "
        "two year Treasury is positive under six of eight model families; the "
        "S&amp;P 500 is positive under three, and never by more than two basis "
        "points of R squared.")
    r.prose(
        "Second, <strong>within fixed income the short maturities dominate the "
        "long ones</strong>. That ordering holds across model families sharing "
        "no functional form, which is the reason to take it seriously. A result "
        "only a random forest can see is a result about random forests. A result "
        "that a univariate regression, a shrinkage estimator and a tree ensemble "
        "all report is a result about the data.")
    r.prose(
        "The high yield column is the exception, and it is not a real one. The "
        "elastic net reports +16.95% and the principal components model +12.99% "
        "on high yield, which no credit model achieves. The fund marks illiquid "
        "bonds with a lag, so its daily returns autocorrelate at 0.29, and a "
        "flexible model learns to predict a move that had already happened. It "
        "is flagged here because it recurs later.")

    r.section("The relationship that motivated this project", (
        "Extending the same measurement to twelve fixed income instruments makes "
        "the ordering explicit."))
    if fipred is not None:
        cols = [c for c in ["oos_r2", "vol", "duration", "risk_share"]
                if c in fipred.columns]
        t = fipred[cols]
        t.index.name = "asset"
        r.table(t.round(4), align_right=cols,
                caption="Out of sample R squared, annualised volatility, "
                        "modified duration, and share of universe variance.")

        fig, ax = charts.new_axes(8.0, 3.6)
        ax.scatter(fipred["duration"], fipred["oos_r2"] * 100, s=70,
                   color=charts.SERIES[0], zorder=3)
        for a, row in fipred.iterrows():
            ax.annotate(a, (row["duration"], row["oos_r2"] * 100),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=charts.MUTED)
        ax.axhline(0, color=charts.MUTED, linewidth=1.0, linestyle="--")
        ax.set_xlabel("modified duration, years")
        ax.set_ylabel("out of sample R squared")
        charts.percent_axis(ax, decimals=1)
        r.figure(charts.to_svg(fig),
                 "Rank correlation negative 0.958, p below 0.001. Every asset "
                 "above five years of duration sits below the line.")

    r.prose(
        "The rank correlation between out of sample R squared and modified "
        "duration is <strong>negative 0.958</strong>, with p below 0.001. "
        "Against volatility it is negative 0.818. The four most forecastable "
        "assets carry 12.8% of universe variance; the four least forecastable "
        "carry 67.9%.")
    r.prose(
        "The mechanism is in the return decomposition. A constant maturity bond "
        "return splits into carry, rolldown, duration and convexity, and the "
        "first two are known at purchase. A short bond's return is mostly those "
        "two. A long bond's return is mostly the yield change, which is not "
        "known. The same fact that makes short bonds forecastable also makes "
        "them low risk.")
    r.prose(
        "The first test of this included equities, so the obvious objection was "
        "that equities were doing the work. Removing them strengthened the "
        "relationship rather than weakening it. It is a property of bonds.")

    r.section("What this implies for portfolio construction", (
        "The finding cuts two ways, and both directions shaped the design of "
        "this project."))
    r.table(pd.DataFrame([
        ("Fixed income is the better place to forecast",
         "Out of sample R squared reaches 2.13% on the two year Treasury and "
         "1.81% on high yield, against 0.05% on equities. Nine of twelve fixed "
         "income assets are forecastable out of sample; two of seven multi "
         "asset ones are."),
        ("But forecasting cannot carry the portfolio",
         "Skill is concentrated in the assets that carry almost none of the "
         "risk. A forecast can only move the book through positions too small "
         "to matter. That predicts, in advance, that forecast driven tilts will "
         "fail here, and Phase 2 confirms it."),
        ("So the covariance matrix is where to work",
         "If expected returns cannot be estimated usefully but the risk "
         "structure can, then a method using only the second moment should "
         "outperform one that needs the first. That is the hypothesis this "
         "project tests."),
    ], columns=["implication", "reasoning"]).set_index("implication"))

    r.section("The universe", (
        "Eleven assets, monthly. All comparisons run from November 1987, the "
        "first month the covariance-based methods can trade after their "
        "estimation window."))
    r.table(pd.DataFrame([
        ("Treasuries", "2, 5, 10 and 30 year constant maturity holdings, built "
         "from a bootstrapped zero curve so they differ only in maturity"),
        ("Corporate credit", "Short, intermediate and long investment grade, "
         "plus high yield"),
        ("Securitized and municipal", "GNMA agency mortgages, intermediate "
         "municipals, high yield municipals"),
    ], columns=["group", "what is in it"]).set_index("group"))
    if stats is not None:
        cols = [c for c in ["group", "cagr", "vol", "sharpe", "corr_ust10y"]
                if c in stats.columns]
        t = stats[cols]
        t.index.name = "asset"
        r.table(t.round(4), align_right=[c for c in cols if c != "group"])
    r.prose(
        "The three month bill is excluded from the portfolio universe. It is a "
        "cash proxy at 0.9% volatility, and every risk minimizing method piles "
        "into it if allowed. A portfolio that wants less risk should hold less "
        "of the portfolio, not relabel cash as an asset. It is retained in the "
        "predictability table above because it is the cleanest illustration of "
        "the duration relationship.")
    r.prose(
        "<strong>This universe contains 1.92 independent bets out of 11.</strong> "
        "The first principal component explains 70.4% of the variance. Fixed "
        "income is close to a single factor, and adding more bonds does not fix "
        "that. Extending the universe from 12 assets to 26 reduced measured "
        "independence rather than raising it, because the additions were "
        "redundant: interpolated curve points, a second mortgage fund correlated "
        "0.98 with the first, duplicate managers in the same sector.")

    r.next_up("Phase 2 - Strategies", [
        "Forecast driven tilts: carry, momentum and regression",
        "Why they fail, and why the failure was predictable",
        "Risk parity, and Lopez de Prado's hierarchical extension",
    ])
    return r.render(OUT / "phase1_idea.html")


# ------------------------------------------------------------------ phase 2

def phase2():
    r = PhaseReport(
        phase="Phase 2", title="Strategies",
        summary=("Nine allocation methods on one universe, in two groups: those "
                 "that require a forecast of returns and those that require only "
                 "a covariance matrix. Every method in the first group loses to "
                 "equal weight. Almost every method in the second group beats "
                 "it."),
        status="complete")

    dev = get("fi_dev_table")
    oos = get("fi_oos_table")
    boot = get("fi_bootstrap")
    skill = get("fi_forecast_skill")

    r.metrics([
        ("9", "allocation methods tested", None),
        ("-0.07", "best forecast tilt vs 1/N", "fail"),
        ("+0.07", "risk parity vs 1/N", "pass"),
        ("+0.13", "hierarchical risk parity vs 1/N", "pass"),
        ("0", "forecasts used by the winner", None),
    ])

    r.section("Forecast driven methods", (
        "Four constructions that tilt the portfolio toward assets expected to "
        "outperform. Two of them use no statistical estimate at all."))
    r.table(pd.DataFrame([
        ("Carry tilt", "Overweight by yield plus rolldown. Fixed income's "
         "native signal, model free, and the most robust cross sectional "
         "predictor documented in the literature."),
        ("Momentum tilt", "Overweight by trailing return. Also model free."),
        ("Forecast tilt", "A univariate regression forecast per asset, combined "
         "across signals, converted into a tilt away from equal weight sized by "
         "forecast strength."),
        ("Forecast tilt, strong", "The same construction with the tilt cap "
         "relaxed, so the forecast is allowed to move the portfolio further."),
    ], columns=["method", "what it does"]).set_index("method"))
    if skill is not None:
        t = skill.copy()
        t.index.name = "asset"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Out of sample R squared per asset from the combined "
                        "regression forecast. Nine of twelve are positive.")
    r.prose(
        "<strong>The forecasts have genuine skill and the portfolios still "
        "lose.</strong> Nine of twelve assets are forecastable out of sample, a "
        "better hit rate than the parent project's multi asset universe managed. "
        "On development the carry tilt scores 0.734 against equal weight's "
        "0.805, momentum 0.684 and the regression tilt 0.736. Relaxing the tilt "
        "cap makes it worse rather than better, at 0.677.")
    r.prose(
        "Carry is the informative failure. It requires no statistical estimate, "
        "it is the signal the fixed income literature is most confident about, "
        "and it still gives up seven basis points of Sharpe to equal weight on "
        "this universe. That is not an estimation problem, so a better estimator "
        "will not fix it.")

    r.section("Regime conditioning", (
        "The other forecast-like approach, imported from the parent project "
        "where it was tested at length."))
    r.prose(
        "On the parent project a 1,296 specification grid over regime "
        "definition, persistence, confirmation delay, probability treatment, "
        "shrinkage and optimiser produced a positive edge on 85% of "
        "specifications in sample, a multiple testing p value of 1.000, and zero "
        "of eight development winners positive on the holdout.")
    r.prose(
        "The narrower version was tested here: condition only the "
        "<em>covariance</em> matrix on the regime, leaving expected returns "
        "alone. That is a much weaker claim, since a covariance matrix is far "
        "easier to estimate than a mean. It made a risk parity book "
        "<strong>worse by 0.11 Sharpe</strong>. Splitting the sample by regime "
        "costs more in estimation error than the regime-specific structure is "
        "worth.")

    r.section("Why the failure was predictable", (
        "Phase 1 said this would happen. Stating it explicitly matters, because "
        "it is what makes the positive result in the next section credible "
        "rather than lucky."))
    r.prose(
        "Forecastability runs inverse to risk at a rank correlation of negative "
        "0.958. The assets a forecast can actually call are the two year "
        "Treasury, the three month bill and short investment grade, which "
        "together carry about 2.5% of universe variance. The assets that carry "
        "the risk, the 30 year Treasury and long credit, have negative out of "
        "sample R squared.")
    r.prose(
        "So a forecast driven portfolio faces a choice with no good branch. Size "
        "positions by conviction and the book barely moves, because conviction "
        "sits in assets with no risk in them. Size them large enough to matter "
        "and the portfolio's risk is dominated by assets the forecast cannot "
        "call. The relaxed tilt cap is that second branch, and it performed "
        "worse than the constrained version.")
    r.prose(
        "<strong>If expected returns cannot be used but the risk structure can, "
        "the natural response is to build the portfolio from the covariance "
        "matrix alone.</strong>")

    r.section("Risk parity", (
        "Equal Risk Contribution. It uses no forecast of any kind."))
    r.prose(
        "Instead of putting equal money in each asset, put equal risk in each "
        "asset. A 30 year Treasury is about eight times more volatile than a two "
        "year, so it receives roughly one eighth the weight. Formally, solve for "
        "the weights at which every asset's risk contribution "
        "<code>w<sub>i</sub> (&Sigma;w)<sub>i</sub> / &sigma;<sub>p</sub></code> "
        "is equal.")
    r.prose(
        "The covariance matrix is re-estimated every month on all data available "
        "to that point, using Ledoit-Wolf shrinkage toward a constant "
        "correlation target. The expected return vector never enters the "
        "objective. That is the entire method, and it is the reason it has "
        "nothing to decay.")
    r.prose(
        "What it harvests is the low beta anomaly documented by Frazzini and "
        "Pedersen (2014): low volatility assets have historically delivered "
        "better risk adjusted returns than high volatility ones. Their "
        "explanation is that the anomaly persists <em>because</em> investors are "
        "leverage constrained. Anyone who can borrow can hold the low volatility "
        "asset at scale; anyone who cannot must reach for volatility to hit a "
        "return target. That is also why a strategy harvesting it runs at low "
        "volatility and has to be levered to compete on returns, and why the "
        "leverage accounting in Phase 3 is not a technicality.")

    r.section("Lopez de Prado's hierarchical extension", (
        "Hierarchical Risk Parity, Journal of Portfolio Management, 2016. It "
        "addresses a specific weakness in standard risk parity."))
    r.prose(
        "Risk parity works with the full covariance matrix. When assets are "
        "highly correlated, which bonds are, that matrix is close to singular "
        "and the weights it produces become unstable. It also treats every asset "
        "as a peer, so four nearly identical Treasury maturities register as "
        "four separate bets rather than one bet held four ways.")
    r.table(pd.DataFrame([
        ("1. Cluster", "Convert the correlation matrix into a distance measure, "
         "then build a tree by joining the closest assets first. Treasuries end "
         "up next to Treasuries, municipals next to municipals."),
        ("2. Quasi-diagonalize", "Reorder the matrix so similar assets sit next "
         "to each other. The result is close to block diagonal, which makes the "
         "structure usable without inverting anything."),
        ("3. Recursive bisection", "Split the tree in two, allocate between the "
         "halves in inverse proportion to their cluster variance, then repeat "
         "inside each half down to individual assets."),
    ], columns=["step", "what happens"]).set_index("step"))
    r.prose(
        "<strong>The key property is that HRP never inverts the covariance "
        "matrix.</strong> It uses it only to measure distance between assets and "
        "to compute cluster variance. Inversion is where estimation error gets "
        "amplified, and a near singular matrix amplifies it violently. Avoiding "
        "the inversion is what makes HRP stable in exactly the situation this "
        "universe presents.")
    r.prose(
        "The second property matters as much. Because capital is allocated "
        "between clusters before it is allocated between assets, adding a "
        "redundant bond does not dilute the rest of the portfolio. In a universe "
        "where the first principal component explains 70% of variance and "
        "several instruments are near duplicates, that is the difference between "
        "measuring diversification and assuming it.")

    r.section("All nine methods, side by side")
    for tbl, cap in [(dev, "Development, November 1987 to December 2015, 338 "
                           "months, net of costs."),
                     (oos, "Holdout, 2016 to 2026, 128 months. Nothing was "
                           "fitted on this period.")]:
        if tbl is not None:
            cols = [c for c in ["cagr", "vol", "sharpe", "max_drawdown", "vs_1N"]
                    if c in tbl.columns]
            t = tbl[cols]
            t.index.name = "strategy"
            r.table(t.round(4), align_right=cols, caption=cap)
    r.prose(
        "The split is clean. <strong>Every method that needs a return forecast "
        "sits below equal weight; every method built from the covariance matrix "
        "sits above it</strong>, with one exception. Minimum variance uses only "
        "covariance and still loses badly at 0.581, because it concentrates the "
        "book into the single lowest volatility asset and gives up too much "
        "return to be worth it. Risk parity is what minimum variance should have "
        "been: it uses the same information without collapsing the portfolio "
        "into one corner of it.")
    if boot is not None:
        cols = [c for c in ["sharpe_strategy", "difference", "ci_lo", "ci_hi",
                            "p_one_sided"] if c in boot.columns]
        t = boot[cols]
        t.index.name = "strategy"
        r.table(t.round(4), align_right=cols,
                caption="Stationary block bootstrap against equal weight, "
                        "development sample, twelve month expected blocks.")

    r.next_up("Phase 3 - Results and Holdout", [
        "Development, holdout and full sample against three benchmarks",
        "Matched on start date, on leverage and on duration",
        "Turnover, costs, borrowing, and what is not proven",
    ])
    return r.render(OUT / "phase2_strategies.html")


# ------------------------------------------------------------------ phase 3

def phase3():
    r = PhaseReport(
        phase="Phase 3", title="Results, Holdout and Next Steps",
        summary=("Risk parity beats equal weight, the Bloomberg Aggregate proxy "
                 "and a 2s10s barbell on the development sample, and the margin "
                 "survives matching on start date, on duration, on leverage and "
                 "on trading costs. It stays positive on the holdout but is not "
                 "statistically significant there."),
        status="complete")

    T = get("fi_aligned_table")
    L = get("fi_aligned_levered")
    Bt = get("fi_aligned_bootstrap")
    C = get("fi_aligned_curves")
    DT = get("fi_rp_duration_test")
    TS = get("fi_paper_turnover_sharpe")
    TO = get("fi_paper_turnover")
    RB = get("fi_rp_robustness")

    r.metrics([
        ("0.933", "hierarchical RP Sharpe, development", "pass"),
        ("0.805", "equal weight Sharpe, development", None),
        ("+0.147", "edge after duration matching", "pass"),
        ("0.008", "development p-value", "pass"),
        ("+0.122", "full sample edge vs equal weight", "pass"),
    ])

    r.section("A correction to how these were measured", (
        "Two problems with the earlier version of this table, both of which "
        "distorted the comparison for reasons unrelated to the strategies."))
    r.prose(
        "<strong>Start dates.</strong> Equal weight and the 2s10s barbell need "
        "no covariance estimate, so they ran from November 1982. Risk parity, "
        "hierarchical risk parity and inverse volatility need one, so a sixty "
        "month estimation window pushed them to November 1987. Comparing series "
        "measured over different windows is not a comparison, whichever way the "
        "bias runs.")
    r.prose(
        "<strong>And the bias does not run the way you would guess.</strong> "
        "Those five extra years contain the highest absolute bond returns "
        "anywhere in the sample: equal weight compounded at 11.63% a year over "
        "them, against 6.87% for the period that follows. The instinct is that "
        "dropping them should make equal weight look worse. It does the "
        "opposite.")
    r.table(pd.DataFrame([
        ("1982-11 to 1987-10, dropped", "11.63%", "7.45%", "4.18%", "6.64%",
         "0.575"),
        ("1987-11 to 2015-12, kept", "6.87%", "3.22%", "3.66%", "4.41%",
         "0.805"),
        ("1982-11 to 2015-12, as first run", "7.58%", "3.85%", "3.72%", "4.82%",
         "0.744"),
    ], columns=["equal weight over", "return", "cash rate", "excess", "vol",
                "Sharpe"]).set_index("equal weight over"),
        align_right=["return", "cash rate", "excess", "vol", "Sharpe"])
    r.prose(
        "Cash paid <strong>7.45%</strong> over those five years. An 11.63% bond "
        "return is a large number and a mediocre one at the same time: it is "
        "4.18% of excess return, earned at 6.64% volatility in the violently "
        "unstable rate environment that followed the Volcker disinflation. That "
        "is a Sharpe of 0.575, well below the 0.805 of the period that follows. "
        "The extra window was <em>dragging equal weight's Sharpe down</em>, not "
        "lifting it.")
    r.prose(
        "So aligning the start dates makes the benchmark harder rather than "
        "easier, and every edge measured against it shrinks by roughly six basis "
        "points of Sharpe.")
    r.table(pd.DataFrame([
        ("Hierarchical RP", "+0.190", "+0.129"),
        ("Risk parity (ERC)", "+0.140", "+0.079"),
        ("Inverse volatility", "+0.130", "+0.069"),
        ("Agg index (VBMFX)", "-0.012", "+0.019"),
    ], columns=["strategy", "edge as first reported",
                "edge on a common start"]).set_index("strategy"),
        align_right=["edge as first reported", "edge on a common start"])
    r.prose(
        "Everything on this page now starts in November 1987, benchmarks "
        "included. The result survives the correction but it is a third smaller "
        "than it was, and the Aggregate proxy moves from behind equal weight to "
        "slightly ahead of it. This is also a reminder that a high-rate decade "
        "flatters nominal returns and punishes Sharpe ratios, which is worth "
        "holding onto when reading any bond result that spans the early 1980s.")
    r.prose(
        "<strong>Volatility.</strong> A growth-of-1 chart puts the lowest "
        "volatility line at the bottom, which inverts the Sharpe ranking "
        "whenever the low volatility strategy is the better one. Hierarchical "
        "risk parity runs at 2.8% volatility against equal weight's 4.7%, so "
        "comparing their cumulative returns directly compares a smaller position "
        "to a larger one rather than comparing two strategies. Every chart and "
        "headline comparison below is levered to a common volatility target, "
        "with a 50 basis point financing spread charged on the borrowed "
        "portion.")

    r.section("Results against three benchmarks", (
        "Equal weight, the Bloomberg Aggregate as proxied by Vanguard Total "
        "Bond, and a 50/50 two year and ten year Treasury barbell."))
    if T is not None:
        for tag, cap in [("dev", "Development, November 1987 to December 2015."),
                         ("oos", "Holdout, 2016 to 2026. Nothing was fitted on "
                                 "this period."),
                         ("full", "Full sample, November 1987 to August 2026.")]:
            t = three_window(
                T, [f"{tag}_cagr", f"{tag}_vol", f"{tag}_sharpe", f"{tag}_dd",
                    f"{tag}_vs_1N"],
                ["return", "volatility", "Sharpe", "worst drawdown",
                 "vs equal weight"])
            r.table(t.round(4), align_right=list(t.columns), caption=cap)

    if L is not None:
        t = three_window(
            L, ["full_leverage", "full_lev_cagr", "full_lev_vol",
                "full_lev_sharpe", "full_lev_dd", "full_lev_vs_1N"],
            ["leverage", "return", "volatility", "Sharpe", "worst drawdown",
             "vs equal weight"])
        r.table(t.round(4), align_right=list(t.columns),
                caption="Full sample, every strategy levered to equal weight's "
                        "own 4.7% volatility with 50bp financing charged. This "
                        "is the apples-to-apples comparison.")
        r.prose(
            "Levered to the same risk, hierarchical risk parity returns 5.75% a "
            "year against equal weight's 5.57%, at the same volatility and with "
            "a smaller drawdown. Note that leverage <em>costs</em> it: an "
            "unlevered Sharpe of 0.659 falls to 0.598 once 1.68 times leverage "
            "is financed at 50 basis points. The ranking holds anyway, which is "
            "the point of charging for it rather than assuming it away.")

    if C is not None:
        roles = {c: ("benchmark" if c == "1/N" else
                     "hero" if "Hierarchical" in c else "strategy")
                 for c in C.columns}
        fig, ax = charts.new_axes(9.0, 4.0)
        charts.growth(ax, C, roles=roles)
        ax.axvline(pd.Timestamp("2016-01-01"), color=charts.MUTED,
                   linestyle=":", linewidth=1.2, zorder=1)
        ax.annotate("holdout begins", xy=(pd.Timestamp("2016-03-01"), 0.04),
                    xycoords=("data", "axes fraction"), fontsize=8,
                    color=charts.MUTED)
        charts.legend(ax, loc="upper left")
        r.figure(charts.to_svg(fig),
                 "Growth of one dollar with every series levered to equal "
                 "weight's volatility and financing charged, so vertical "
                 "position corresponds to risk-adjusted performance. Marked "
                 "line is hierarchical risk parity; dashed grey is equal "
                 "weight.")

    r.section("Statistical significance", (
        "Monthly returns are serially dependent, so a textbook standard error on "
        "a Sharpe ratio is too narrow. A stationary block bootstrap with twelve "
        "month expected blocks, paired so the common market move is differenced "
        "away rather than adding noise to the comparison."))
    if Bt is not None:
        for tag, cap in [("dev", "Development."), ("oos", "Holdout."),
                         ("full", "Full sample.")]:
            t = three_window(
                Bt, [f"{tag}_edge", f"{tag}_lo", f"{tag}_hi", f"{tag}_p"],
                ["edge vs equal weight", "CI low", "CI high", "p"])
            r.table(t.round(4), align_right=list(t.columns), caption=cap)
    r.prose(
        "These edges differ slightly from the <em>vs equal weight</em> column in "
        "the performance tables above, by one or two basis points of Sharpe. The "
        "performance tables annualise a geometric return over volatility; the "
        "bootstrap works on arithmetic excess returns, because that is what can "
        "be resampled in paired blocks. The two conventions agree on ranking and "
        "on sign, and the bootstrap number is the one each p-value refers to.")
    r.prose(
        "<strong>Development and full sample are significant; the holdout is "
        "not.</strong> Hierarchical risk parity is +0.150 on development with a "
        "95% interval of [+0.034, +0.271] and p = 0.009, and +0.122 on the full "
        "sample at p = 0.010. On the holdout alone it is +0.012 with an interval "
        "of [-0.085, +0.166] and p = 0.389. Risk parity is the stronger of the "
        "two out of sample, at +0.030 with p = 0.178, and that is still not "
        "significant.")
    r.prose(
        "Both benchmarks go the other way with more confidence. The Aggregate "
        "proxy is significantly <em>worse</em> than equal weight over the "
        "holdout, at -0.103 with p = 0.001, and the barbell at -0.170 with "
        "p = 0.042.")

    r.section("Why the holdout is hard to read", (
        "The decade the strategies were tested on contained the worst bond "
        "market in forty years."))
    r.table(pd.DataFrame([
        ("2021", "-0.66%", "-1.75%", "-0.29%", "-0.32%"),
        ("2022", "-13.53%", "-13.24%", "-7.59%", "-9.95%"),
        ("2023", "+6.22%", "+5.62%", "+5.63%", "+6.06%"),
        ("2021-2023 cumulative", "-8.75%", "-9.96%", "-2.67%", "-4.80%"),
        ("Holdout worst drawdown", "-17.3%", "-17.5%", "-10.3%", "-13.0%"),
    ], columns=["", "equal weight", "Agg proxy", "Hierarchical RP",
                "Risk parity"]).set_index(""),
        align_right=["equal weight", "Agg proxy", "Hierarchical RP",
                     "Risk parity"])
    r.prose(
        "Equal weight returned 2.2% a year over the holdout at a Sharpe of "
        "0.026. The Aggregate proxy was negative on a risk-adjusted basis at "
        "-0.072, and the barbell at -0.152. When the benchmark itself is at "
        "zero, there is very little for a strategy to separate on. A positive "
        "but small margin is exactly what a real edge looks like under those "
        "conditions, and it is also exactly what noise looks like, which is why "
        "this page does not claim significance.")
    r.prose(
        "What the drawdown row does show is that the risk-based methods took "
        "materially less damage through the rate shock: -7.6% for hierarchical "
        "risk parity in 2022 against -13.5% for equal weight and -13.2% for the "
        "Aggregate proxy. Underweighting duration is the mechanism, which raises "
        "the obvious question addressed next.")

    r.section("Is it just a bet on shorter bonds?", (
        "This is the objection that should kill the result if anything does. "
        "Risk parity underweights volatile assets, volatility in bonds is "
        "duration, so it naturally holds less duration than equal weight."))
    r.prose(
        "If that is the whole story, an investor who wants less duration can "
        "hold less duration and skip the covariance matrix entirely. So the "
        "benchmark was rebuilt: equal weight, scaled up or down month by month "
        "to match each strategy's own portfolio duration, with the difference "
        "held in cash at the risk free rate. That benchmark carries the same "
        "duration exposure as the strategy and none of its structure.")
    if DT is not None:
        d = DT.copy()
        d.index.name = "strategy"
        r.table(d.round(4), align_right=list(d.columns))
    r.prose(
        "<strong>The edge does not change.</strong> Hierarchical risk parity "
        "goes from +0.151 against plain equal weight to +0.147 against the "
        "duration matched version, at p = 0.008. Risk parity goes from +0.087 to "
        "+0.088 at p = 0.012. All intervals exclude zero. It is not a duration "
        "bet, and the group risk budget variant behaves the same way.")

    r.section("Costs, turnover and borrowing")
    if TS is not None and TO is not None:
        both = pd.concat([TS.add_suffix(" Sharpe"), TO.add_suffix(" turnover")],
                         axis=1)
        both.index.name = "strategy"
        r.table(both.round(4), align_right=list(both.columns),
                caption="By rebalancing frequency.")
    r.prose(
        "Turnover runs 5% to 14% a year. Risk parity trades <em>less</em> than "
        "equal weight, because bond correlations are stable while equal weight "
        "has to trade back against price drift every month. Annual rebalancing "
        "is marginally best, which says the covariance estimate is stable enough "
        "that monthly re-optimization is mostly noise.")
    if RB is not None:
        g = RB.groupby(["cov", "lookback"])[["dev_edge", "oos_edge"]].mean()
        r.table(g.round(4), align_right=["dev_edge", "oos_edge"],
                caption="27 combinations of covariance estimator, lookback "
                        "window and rebalancing frequency, averaged.")
        pos_dev = int((RB["dev_edge"] > 0).sum())
        pos_oos = int((RB["oos_edge"] > 0).sum())
        r.prose(
            f"<strong>{pos_dev} of 27 combinations are positive on development "
            f"and {pos_oos} of 27 on the holdout.</strong> The pattern in the "
            "failures is informative: every expanding-window specification is "
            "positive on both, and the losses concentrate in the short sixty "
            "month lookback and the EWMA estimator, both of which discard the "
            "long history that makes the covariance estimate stable. That is the "
            "same lesson as the annual rebalancing result.")

    r.section("What I would and would not claim")
    r.checks([
        (True, "Risk parity beats equal weight on this universe",
         "0.933 and 0.884 against 0.805 on development, both bootstrap "
         "intervals exclude zero"),
        (True, "The edge is not explained by duration",
         "unchanged against a duration matched benchmark, p = 0.008 and 0.012"),
        (True, "It beats a 2s10s barbell and the Aggregate proxy",
         "barbell 0.655, Vanguard Total Bond 0.824, on the same window"),
        (True, "It survives leverage and trading costs",
         "still ahead levered to equal weight's volatility at 50bp; turnover "
         "under 15% a year"),
        (True, "It holds over the full sample",
         "+0.122 against equal weight from 1987 to 2026, p = 0.010"),
        (False, "It is proven out of sample",
         "2016 to 2026 is positive against all three benchmarks but p = 0.389"),
        (False, "Any forecast is involved",
         "the covariance matrix only, no return prediction anywhere"),
    ])
    r.prose(
        "The honest summary is that the development and full sample results are "
        "solid and the holdout result is directionally right but unproven. A "
        "decade in which the benchmark returned a 0.03 Sharpe is a poor "
        "environment for demonstrating anything, in either direction.")

    r.section("Next steps")
    r.table(pd.DataFrame([
        ("Replace funds with indices or futures",
         "The universe is mutual funds, which charge 20 to 80 basis points and "
         "carry manager decisions. High yield's daily returns autocorrelate at "
         "0.29 from stale pricing. Index or futures data removes both problems "
         "and would make daily work trustworthy."),
        ("Add genuinely different exposures",
         "Adding more US bonds reduced independence rather than raising it. "
         "TIPS, international sovereigns, emerging market debt and bank loans "
         "are different factors, not more of the same one."),
        ("Test the result on a second market",
         "If the duration and forecastability relationship is structural it "
         "should appear in gilts and bunds too. That would be much stronger "
         "evidence than another robustness check on the same data."),
        ("Cost aware optimisation",
         "On the parent project, trading a fraction of the way toward the target "
         "each period instead of fully rebalancing raised Sharpe from 0.06 to "
         "0.23 with turnover cut ninefold. Turnover is already low here, so the "
         "gain would be smaller, but the method carries over."),
    ], columns=["step", "why"]).set_index("step"))

    r.section("Limitations")
    r.table(pd.DataFrame([
        ("Mutual funds, not indices",
         "Fees of 20 to 80 basis points and manager idiosyncrasy. Daily returns "
         "autocorrelate, high yield at 0.29, because illiquid bonds are priced "
         "with a lag."),
        ("The sample starts in 1987",
         "Five years of available data are discarded so every strategy and "
         "benchmark shares a start date. That is the right trade, but it costs "
         "sample size."),
        ("Out of sample is not significant",
         "Positive against all three benchmarks with intervals spanning zero."),
        ("The holdout was opened once for the parent project",
         "This holdout is clean of fitting but not of the author having seen "
         "that decade."),
    ], columns=["limitation", "what it means"]).set_index("limitation"))

    r.prose(
        "Reference: Lopez de Prado, M. (2016). Building Diversified Portfolios "
        "that Outperform Out of Sample. <em>Journal of Portfolio Management</em>, "
        "42(4), 59-69.")
    return r.render(OUT / "phase3_results.html")


def index():
    T = get("fi_aligned_table")
    Bt = get("fi_aligned_bootstrap")

    body = ""
    if T is not None:
        for name in T.index:
            cells = []
            for tag in ["dev", "oos", "full"]:
                sh = T.loc[name, f"{tag}_sharpe"]
                # Edge and p-value both come from the bootstrap, so the number
                # shown is the one the p-value actually refers to.
                has_b = Bt is not None and name in Bt.index
                sub = "benchmark"
                if has_b:
                    sub = (f"{Bt.loc[name, f'{tag}_edge']:+.3f}"
                           f" &middot; p {Bt.loc[name, f'{tag}_p']:.3f}")
                cells.append(f'{sh:.3f}<span class="d">{sub}</span>')
            cls = ' class="me"' if ("RP" in name or "parity" in name) else ""
            body += (f"<tr{cls}><th>{name}</th>"
                     + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    rows = [
        ("1", "The Idea", "phase1_idea.html",
         "Predictive skill by instrument and by model family. Fixed income is "
         "more forecastable than equity, short maturities more than long, and "
         "the ordering holds across model families that share no functional "
         "form."),
        ("2", "Strategies", "phase2_strategies.html",
         "Nine allocation methods. Carry, momentum and regression tilts all lose "
         "to equal weight; risk parity and Lopez de Prado's hierarchical version "
         "both beat it. Why the split was predictable."),
        ("3", "Results and Holdout", "phase3_results.html",
         "Development, holdout and full sample against three benchmarks, matched "
         "on start date, on leverage and on duration. Turnover, costs, "
         "borrowing, and what is not proven."),
    ]
    cards = "".join(
        f'<a class="card" href="{href}"><span class="num">Phase {n}</span>'
        f'<span class="ttl">{title}</span><span class="dsc">{desc}</span></a>'
        for n, title, href, desc in rows)

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Fixed Income Risk Parity</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{--bg:#f6f7f8;--fg:#16202b;--mut:#5d6b78;--rule:#dde3e8;--card:#fff;--acc:#12556b}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1417;--fg:#e6ecef;--mut:#93a3ad;--rule:#232d34;--card:#161d22;--acc:#5cb6c8}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.6}}
.wrap{{max-width:52rem;margin:0 auto;padding:4rem 1.5rem 5rem}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--mut)}}
h1{{font-size:2.1rem;margin:.5rem 0 1.2rem;letter-spacing:-.02em}}
h2{{font-size:.78rem;font-family:"IBM Plex Mono",monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);margin:2.4rem 0 .8rem}}
.intro p{{margin:0 0 1rem;max-width:42rem}}
.scroll{{overflow-x:auto;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
table{{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}}
th,td{{padding:.6rem .7rem;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:top}}
thead th{{font-weight:500;font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);line-height:1.4}}
tbody th{{text-align:left;font-weight:400}}
tbody tr:last-child td,tbody tr:last-child th{{border-bottom:none}}
tr.me th,tr.me td{{font-weight:600}}
.d{{display:block;font-size:.74rem;color:var(--mut);font-weight:400}}
.note{{font-size:.8rem;color:var(--mut);margin:.7rem 0 2.4rem;max-width:42rem}}
.card{{display:block;background:var(--card);border:1px solid var(--rule);border-radius:4px;padding:1.1rem 1.3rem;margin-bottom:.7rem;text-decoration:none;color:inherit}}
.card:hover{{border-color:var(--acc)}}
.num{{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--acc)}}
.ttl{{display:block;font-weight:600;font-size:1.05rem;margin:.15rem 0 .3rem}}
.dsc{{display:block;color:var(--mut);font-size:.92rem}}
footer{{margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid var(--rule);font-size:.85rem;color:var(--mut)}}
a.plain{{color:var(--acc)}}
</style>
<div class="wrap">
<span class="eyebrow">Gus Guenther &middot; UCLA Anderson MFE</span>
<h1>Fixed Income Risk Parity</h1>
<div class="intro">
<p>This project builds on my <a class="plain" href="{URL_MACRO}">macro portfolio
rebuild</a>, which showed how difficult equity returns are to predict but also
produced real evidence that fixed income returns are predictable. The goal here
was to take that evidence seriously and build a bond-only portfolio that beats
its benchmark.</p>

<p>The universe extends to eleven fixed income assets: four constant maturity
Treasury holdings at 2, 5, 10 and 30 years, four corporate credit sleeves running
from short investment grade through to high yield, and three securitized and
municipal positions covering agency mortgages, intermediate municipals and high
yield municipals. It uses the same underlying data and the same walk-forward
backtesting framework as the prior project, so results across the two are
directly comparable.</p>

<p>I developed a range of predictive models here, using regression forecasts,
carry and momentum signals, and machine learning. Those results were
inconclusive, for a reason that turned out to be structural rather than
technical: in fixed income, the assets you can forecast are the ones carrying
almost none of the risk. Along the way it became clear that a <strong>risk parity
framework works considerably better</strong>, because it never needs an expected
return at all.</p>

<p>I built and tested both classic risk parity and Marcos Lopez de Prado's
Hierarchical Risk Parity, and both show a clear edge over the benchmarks in
development. Against equal weight, hierarchical risk parity delivers a
<strong>0.933 Sharpe against 0.805</strong>, an edge of +0.150 with a 95%
confidence interval of [+0.034, +0.271] and p = 0.009. Classic risk parity
delivers 0.884, an edge of +0.085 at p = 0.015. Both also beat the Bloomberg
Aggregate proxy at 0.824 and a 2s10s Treasury barbell at 0.655.</p>

<p>The results continue to perform out of sample, though not at a level I would
call significant: hierarchical risk parity is +0.012 against equal weight over
the holdout and classic risk parity +0.030, with confidence intervals spanning
zero. Some of that is the environment. The holdout decade contained the worst
bond market in forty years, with equal weight down 13.5% in 2022 and returning
just 2.2% a year across the period at a 0.03 Sharpe. There was very little for
anything to separate on. <strong>Over the full sample the edge is significant
again</strong>, at +0.122 for hierarchical risk parity (p = 0.010) and +0.077 for
classic risk parity (p = 0.006).</p>

<p>Because risk parity naturally holds less duration, the obvious objection is
that this is simply a bet on shorter bonds. It is not. Measured against an equal
weight benchmark rescaled month by month to match each strategy's own portfolio
duration, the edge is essentially unchanged: <strong>+0.147 for hierarchical risk
parity at p = 0.008</strong>, and +0.088 for classic risk parity at p = 0.012.</p>
</div>

<h2>Sharpe ratio, and edge against equal weight</h2>
<div class="scroll">
<table>
<thead><tr><th>Strategy</th><th>Development<br>1987-2015</th><th>Holdout<br>2016-2026</th><th>Full sample<br>1987-2026</th></tr></thead>
<tbody>{body}</tbody>
</table>
</div>
<p class="note">Every series starts in November 1987, the first month the
covariance-based methods can trade after their estimation window, so no strategy
or benchmark gets a head start. p-values are one-sided, from a stationary block
bootstrap with twelve month expected blocks.</p>

{cards}
<footer>
The project this grew out of:
<a class="plain" href="{URL_MACRO}">macro-portfolio-rebuild</a>.
&nbsp;&middot;&nbsp;
<a class="plain" href="https://github.com/guenther-QR/fixed-income-risk-parity">Source on GitHub</a>
</footer>
</div>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    return OUT / "index.html"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in [phase1, phase2, phase3, index]:
        try:
            p = fn()
            print(f"  wrote {Path(p).name}")
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
