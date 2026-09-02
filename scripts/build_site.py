"""Build the three phase reports, two appendices, and the index."""
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
PROJECT = "Fixed Income Risk Parity"

DUR = {"ust2y": 1.9, "ust5y": 4.6, "ust10y": 8.4, "ust30y": 18.5,
       "ig_short": 2.5, "ig": 4.2, "ig_long": 12.0, "hy": 4.0, "mbs": 4.5,
       "muni": 5.0, "muni_hy": 7.5}

PAGES = [
    ("phase1_idea", "Phase 1", "The Idea"),
    ("phase2_strategies", "Phase 2", "Strategies"),
    ("phase3_results", "Phase 3", "Results"),
    ("appendix_universe", "Appendix A", "Universe Structure"),
    ("appendix_method", "Appendix B", "Implementation"),
]

DROP_ROWS = ["Inverse volatility"]


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


def trim(df):
    if df is None:
        return None
    return df.drop(index=[i for i in DROP_ROWS if i in df.index],
                   errors="ignore")


def paginate(r, stem):
    i = [p[0] for p in PAGES].index(stem)
    back = (f"{PAGES[i-1][0]}.html", PAGES[i-1][2]) if i > 0 else None
    fwd = (f"{PAGES[i+1][0]}.html", PAGES[i+1][2]) if i < len(PAGES) - 1 else None
    return r.nav(index=("index.html", "All sections"), prev=back, next=fwd)


def window(T, tag, cols, labels):
    keep = [f"{tag}_{c}" for c in cols if f"{tag}_{c}" in T.columns]
    t = T[keep].copy()
    t.columns = labels[:len(keep)]
    t.index.name = "strategy"
    return t


KIND = {"risk based": "k-risk", "regime conditional": "k-risk",
        "benchmark": "k-bench"}


def row_kinds(T):
    """Colour the row rail by what the model has to estimate."""
    if T is None or "class" not in T.columns:
        return {}
    return {i: KIND.get(c, "k-return") for i, c in T["class"].items()}


LEGEND = ('<div class="legend"><span class="l-risk">risk based, no return '
          'forecast</span><span class="l-return">needs a return '
          'forecast</span><span class="l-bench">benchmark</span></div>')


# ------------------------------------------------------------------ phase 1

def phase1():
    r = PhaseReport(
        phase="Phase 1", title="The Idea",
        summary=("A prior multi-asset study tested roughly 1,600 "
                 "specifications against a sealed holdout and none beat a "
                 "60/40 benchmark. Its one durable result pointed at bonds. "
                 "This phase sets out that evidence, the universe built to "
                 "follow it, and what this half adds."),
        status="complete", project=PROJECT)

    stats = get("fi_asset_stats")
    zoo = macro("daily_model_skill")

    r.metrics([
        ("11", "assets in the universe", None),
        ("1982-2026", "data available", None),
        ("1987-2026", "evaluated, after burn-in", None),
        ("2.13%", "best bond out of sample R squared", "pass"),
        ("0.05%", "equity out of sample R squared", "fail"),
    ])

    r.section("Where this comes from", (
        "The parent project rebuilt a 2025 macro allocation study, corrected "
        "its accounting, and gave it the out of sample test it never had."))
    r.prose(
        "Roughly 1,600 specifications were tested there: regime allocation "
        "across 1,296 design combinations, return regression on 182 signals, "
        "eight machine learning families, recession timing and cross sectional "
        "ranking, at monthly and daily frequency, on universes from 7 to 59 "
        "assets. None beat a 60/40 benchmark on both the development sample and "
        "the held-out decade. One thing in that work did hold up everywhere it "
        "was tested, and it was about bonds rather than about the method.")

    r.section("Predictive skill by instrument and by model", (
        "Eight model families fitted at daily frequency across a seven asset "
        "universe, five of which are fixed income. Out of sample R squared is "
        "measured against a rolling mean forecast, so positive means the model "
        "beats predicting the average."))
    if zoo is not None:
        fi_cols = [c for c in ["ig", "hy", "ust2y", "ust10y", "ust30y"]
                   if c in zoo.columns]
        eq_cols = [c for c in ["sp500", "gold"] if c in zoo.columns]
        t = (zoo[fi_cols + eq_cols] * 100).copy()
        t.index.name = "model family"
        r.table(t.round(2), align_right=list(t.columns), heat=list(t.columns),
                caption="Out of sample R squared, percent. Green positive, red "
                        "negative, shading scaled within each column. Left "
                        "block is fixed income, right block equity and gold.")
    r.prose(
        "Two patterns. <strong>The fixed income columns are systematically "
        "better than the equity column.</strong> The two year Treasury is "
        "positive under six of eight model families; the S&amp;P 500 under "
        "three, and never by more than two basis points of R squared. And "
        "<strong>within fixed income the short maturities beat the long "
        "ones</strong>, an ordering that holds across model families sharing no "
        "functional form.")
    r.prose(
        "High yield reads differently because of how the fund is priced. Its "
        "daily returns autocorrelate at <strong>0.29</strong>, against 0.03 for "
        "investment grade, 0.01 for the two year Treasury and -0.08 for the "
        "S&amp;P 500. Illiquid bonds are marked with a lag, so part of today's "
        "move is yesterday's already-known move and a flexible model can "
        "predict it. The elastic net's +16.95% on high yield is that "
        "autocorrelation, and every daily credit result in this project carries "
        "the same caveat.")

    r.section("The universe", (
        "Eleven assets, monthly. Data begins November 1982 and every model "
        "burns in for sixty months, so the first weight is formed in November "
        "1987 and every comparison in this project starts there."))
    r.table(pd.DataFrame([
        ("Treasuries", "4", "2, 5, 10 and 30 year constant maturity holdings, "
         "built from a bootstrapped zero curve so they differ only in "
         "maturity"),
        ("Corporate credit", "4", "Short, intermediate and long investment "
         "grade, plus high yield"),
        ("Securitized and municipal", "3", "GNMA agency mortgages, "
         "intermediate municipals, high yield municipals"),
    ], columns=["group", "n", "what is in it"]).set_index("group"))
    if stats is not None:
        cols = [c for c in ["group", "cagr", "vol", "sharpe", "corr_ust10y"]
                if c in stats.columns]
        t = stats[cols]
        t.index.name = "asset"
        r.table(t.round(4), align_right=[c for c in cols if c != "group"],
                caption="All twelve series are complete from November 1982 with "
                        "no gaps. Nothing enters the panel late.")
    r.prose(
        "The three month bill is excluded from the portfolio universe. It is a "
        "cash proxy at 0.9% volatility and every risk minimising method piles "
        "into it if allowed. A portfolio that wants less risk should hold less "
        "of the portfolio, not relabel cash as an asset.")

    r.section("What this half adds", (
        "Beyond the change of universe, three things are new relative to the "
        "parent study."))
    r.table(pd.DataFrame([
        ("A bond-specific signal set", "Carry and rolldown by maturity, "
         "modified duration, forward rates at every curve node, and the "
         "Cochrane-Piazzesi forward rate factor, alongside the macro and credit "
         "signals carried over."),
        ("Regime terms inside the models", "Regime dummies and regime "
         "interactions enter the signal panel directly, so a regime can change "
         "a signal's slope rather than only its intercept."),
        ("One estimation window", "Every model class burns in for the same "
         "sixty months and is scored on the same dates, so a comparison across "
         "classes is not a comparison of sample lengths."),
    ], columns=["addition", "what it means"]).set_index("addition"))
    r.prose(
        "The walk-forward backtesting framework is carried over unchanged, "
        "which is what makes results across the two projects directly "
        "comparable.")

    r.next_up("Phase 2 - Strategies", [
        "Five model classes and how each forms its weights",
        "Machine learning tuned on development before it is scored",
        "Development results, and what goes forward",
    ])
    paginate(r, "phase1_idea")
    return r.render(OUT / "phase1_idea.html")


# ------------------------------------------------------------------ phase 2

def phase2():
    r = PhaseReport(
        phase="Phase 2", title="Strategies",
        summary=("Five classes of model on one universe, one estimation window "
                 "and one benchmark. Everything on this page is measured on the "
                 "development sample only; the holdout is opened in Phase 3."),
        status="complete", project=PROJECT)

    T = get("fi_uni_summary")
    ML = get("fi_uni_ml_chosen")
    MLS = get("fi_uni_ml_skill")
    UNI = get("fi_uni_regression_skill")

    bd = None
    if T is not None and {"dev_sharpe", "dev_vs_agg"} <= set(T.columns):
        bd = float((T["dev_sharpe"] - T["dev_vs_agg"]).median())

    r.metrics([
        ("5", "model classes", None),
        ("2", "ways of forming weights", None),
        (f"{bd:.3f}" if bd is not None else "0.807", "Agg Sharpe, development",
         None),
        ("60", "month burn-in, every model", None),
        ("development", "this page only", None),
    ])

    r.section("What each class has to estimate", (
        "The classes are grouped by what a model needs to get right, because "
        "that determines how it fails."))
    r.table(pd.DataFrame([
        ("Technical", "Nothing", "Carry is the yield plus rolldown implied by "
         "today's curve. Momentum is the trailing twelve month return skipping "
         "the most recent month. Both are arithmetic on observed data, with no "
         "fitted parameter to overfit."),
        ("Regression", "A conditional mean", "One expanding-window univariate "
         "least squares fit per signal per asset, averaged across signals."),
        ("Machine learning", "A conditional mean, flexibly", "Elastic net, "
         "ridge, random forest and gradient boosting on the same panel, each "
         "tuned on development before being scored."),
        ("Regime conditional", "A mean or a covariance, within state", "The "
         "sample is split by macro regime and the estimate formed inside it."),
        ("Risk based", "A covariance matrix", "Equal risk contribution and "
         "hierarchical risk parity. No expected return enters the objective."),
    ], columns=["class", "what it estimates", "how"]).set_index("class"))

    r.section("The regression, in detail", (
        "This is the workhorse of the return-forecasting side, so it is worth "
        "setting out precisely."))
    r.prose(
        "For each asset and each signal independently, fit an expanding window "
        "univariate ordinary least squares regression of next month's excess "
        "return on the signal:")
    r.formula(
        "<span class='t t1'>y<sub>t</sub></span> &nbsp;=&nbsp; "
        "&alpha; &nbsp;+&nbsp; &beta; &middot; "
        "<span class='t t3'>x<sub>t-1</sub></span> &nbsp;+&nbsp; "
        "&epsilon;<sub>t</sub>",
        "one regression per signal, per asset")
    r.prose(
        "Then average the forecasts across signals into one number per asset "
        "per month. This is the Goyal-Welch combination, and it is deliberately "
        "the least flexible construction in the project: a single signal on a "
        "single asset is a coin flip with a standard error attached, and "
        "averaging cancels most of the noise while leaving whatever common "
        "component exists.")
    r.table(pd.DataFrame([
        ("Estimator", "Ordinary least squares, computed in closed form from "
         "cumulative moments rather than refitted each month. NumPy only, no "
         "estimation package."),
        ("Window", "Expanding, minimum sixty months, matching every other "
         "class."),
        ("Timing", "Coefficients applied at t are estimated only from outcomes "
         "realised through t-1, and every signal carries its own publication "
         "lag before it enters."),
        ("Combination", "Equal-weighted mean across signals."),
        ("Regime terms", "Four regime dummies enter the panel directly. Regime "
         "interaction terms let a regime change a signal's slope rather than "
         "only its intercept; the signals interacted are selected on "
         "development by the magnitude of their standardised slope."),
    ], columns=["choice", "what was done"]).set_index("choice"))
    if UNI is not None:
        t = (UNI * 100).copy()
        t.index.name = "asset"
        t.columns = [c.replace("_r2", " R squared, %") for c in t.columns]
        r.table(t.round(3), align_right=list(t.columns), heat=list(t.columns),
                caption="Out of sample R squared from the combined regression, "
                        "against a rolling mean forecast.")

    r.section("Machine learning, and how it was tuned", (
        "Four families on the same signal panel. Hyperparameters are selected "
        "on the development sample only, then the chosen setting is scored."))
    r.prose(
        "This matters more than it sounds. An untuned model reported as a "
        "failure is evidence about default settings, not about the model class. "
        "Each family is run across a grid, the configuration with the best "
        "development R squared is selected, and only that one is carried "
        "forward. The holdout is never consulted in the choice.")
    if ML is not None:
        t = ML.copy()
        t.index.name = "family"
        if "dev_r2" in t.columns:
            t["dev_r2"] = pd.to_numeric(t["dev_r2"], errors="coerce") * 100
        t = t.rename(columns={"params": "chosen setting",
                              "dev_r2": "development R squared, %"})
        r.table(t.round(3),
                align_right=[c for c in t.columns if c != "chosen setting"],
                caption="Selected on development. Grids covered regularisation "
                        "strength and mix for the linear models, depth for the "
                        "forest, and depth and learning rate for the boosted "
                        "trees.")
    if MLS is not None:
        t = (MLS * 100).copy()
        t.columns = [c.replace("_r2", " R squared, %") for c in t.columns]
        r.table(t.round(2), align_right=list(t.columns), heat=list(t.columns),
                caption="The tuned model in each family, by asset.")

    r.section("Two ways of turning a forecast into a portfolio", (
        "Every return signal above is run both ways, because a bounded tilt can "
        "hide how wrong a signal is."))
    r.prose(
        "<strong>Tilt.</strong> Start from equal weight, deviate in proportion "
        "to the cross-sectionally standardised forecast, and cap the deviation "
        "per asset:")
    r.formula(
        "w &nbsp;=&nbsp; clip( <span class='t t1'>w<sub>eq</sub></span> "
        "&nbsp;+&nbsp; <span class='t t2'>&Lambda;</span> &middot; "
        "<span class='t t3'>z</span> , &nbsp;bounds )",
        "bounded tilt, lambda = 0.5, cap 15 points per asset")
    r.prose(
        "This is what a desk would run. If the forecast is worthless the "
        "portfolio stays near equal weight and loses only transaction costs; if "
        "it carries information the tilt captures part of it. The downside is "
        "bounded by construction rather than by luck, which is the response "
        "DeMiguel, Garlappi and Uppal give to estimation error of this size.")
    r.prose(
        "<strong>Base.</strong> The signal alone sets the weights, long only "
        "and normalised to sum to one, with no base allocation underneath. "
        "Nobody would run this. It is included because it shows what the signal "
        "actually wants to hold, and a tilt that barely moves can make a badly "
        "wrong signal look merely disappointing.")

    r.section("Development results", (
        "Every model, the full development sample, net of per-asset "
        "transaction costs, measured against the Aggregate."))
    if T is not None:
        t = window(T, "dev", ["sharpe", "vol", "vs_agg", "p"],
                   ["Sharpe", "volatility", "vs the Agg", "p"])
        t.insert(0, "class", T["class"])
        r.table(t.round(4),
                align_right=[c for c in t.columns if c != "class"],
                stars=["p"], row_class=row_kinds(T),
                caption="Sorted by development Sharpe. Stars mark one-sided "
                        "bootstrap significance: *** below 0.01, ** below 0.05, "
                        "* below 0.10.")
        r._blocks.append(LEGEND)
    r.prose(
        "Read the rail on the left rather than the ordering. Everything that "
        "needs a return forecast carries one colour and everything built from "
        "the covariance matrix another, and the separation between those two "
        "groups is cleaner than the separation between any two individual "
        "models.")
    r.prose(
        "<strong>Every base variant is far worse than its tilt.</strong> Carry "
        "goes from 0.734 as a tilt to 0.581 as a standalone portfolio, the "
        "regression from 0.714 to 0.542, momentum from 0.686 to 0.405. The "
        "volatility column explains it: the base portfolios run at 7% to 9% "
        "against 5% for the tilts, because a signal-weighted book concentrates "
        "into whichever assets the signal likes this month and those are "
        "usually the long ones. This is the clearest evidence on the page that "
        "the bounded tilt was doing the work rather than the forecast. Given "
        "room, the same signals actively destroy risk-adjusted return.")
    r.prose(
        "One row is worth explaining rather than ranking. <strong>The ridge "
        "tilt scores 0.910 on development despite the worst out of sample R "
        "squared of any model tested.</strong> Ridge at alpha 100 shrinks its "
        "forecasts so hard toward the sample mean that the tilt barely moves, "
        "so the portfolio is close to equal weight with a small amount of "
        "residual variation. It scores well by not really doing anything, its "
        "development margin is not significant at p = 0.13, and it does not "
        "survive the holdout. A model can look good in a portfolio for reasons "
        "that have nothing to do with its forecasts being right.")

    r.section("What goes forward", (
        "Two strategies are carried into the holdout."))
    r.prose(
        "<strong>Equal risk contribution and hierarchical risk parity, both on "
        "the Ledoit-Wolf covariance.</strong> Neither uses a return forecast.")
    r.prose(
        "The regime-conditional variants are not carried, on the evidence "
        "rather than on principle. Conditioning the covariance on the regime "
        "takes hierarchical risk parity from 0.932 to 0.909 and equal risk "
        "contribution from 0.884 to 0.888: worse on one, a rounding error on "
        "the other. The hierarchical version already clusters the correlation "
        "matrix, so cutting its estimation window into four states costs "
        "estimation precision without adding structure it did not have.")
    r.prose(
        "DCC-GARCH is not carried either, for the mirror reason. It takes "
        "hierarchical risk parity from 0.932 to 0.800 and equal risk "
        "contribution from 0.884 to 0.865. A time-varying correlation estimator "
        "picks up variation that Ledoit-Wolf shrinkage deliberately suppresses, "
        "and on eleven highly correlated bond funds that variation is mostly "
        "noise.")
    r.prose(
        "Nothing from the return-forecasting side is carried. Out of sample R "
        "squared near one percent is real but too small to size a position on, "
        "and the base-weighted versions show what happens when it is allowed "
        "to try.")

    r.next_up("Phase 3 - Results", [
        "The held-out decade, opened once",
        "Constant risk, duration and financing",
        "What the portfolio actually holds",
    ])
    paginate(r, "phase2_strategies")
    return r.render(OUT / "phase2_strategies.html")


# ------------------------------------------------------------------ phase 3

def phase3():
    r = PhaseReport(
        phase="Phase 3", title="Results",
        summary=("The held-out decade, 2016 to 2026, opened once after "
                 "development was finished. Risk parity clears the Aggregate "
                 "and the margin survives matching on risk, on duration and on "
                 "financing. Every model that needed a return forecast does "
                 "not."),
        status="complete", project=PROJECT)

    T = get("fi_uni_summary")
    L = trim(get("fi_aligned_levered"))
    Bt = trim(get("fi_aligned_bootstrap"))
    C = get("fi_aligned_curves")
    DT = trim(get("fi_rp_duration_test"))
    TS = trim(get("fi_paper_turnover_sharpe"))
    TO = trim(get("fi_paper_turnover"))
    RB = get("fi_rp_robustness")
    FBE = trim(get("fi_financing_breakeven"))
    HW = get("fi_uni_hrp_weights")

    r.metrics([
        ("128", "holdout months", None),
        ("-0.072", "Agg Sharpe on the holdout", "fail"),
        ("+0.115", "hierarchical RP edge", "pass"),
        ("0.042", "holdout p-value", "pass"),
        ("+0.147", "edge after duration matching", "pass"),
    ])

    r.section("The holdout", (
        "Every model from Phase 2 on data none of them was fitted to."))
    if T is not None:
        t = window(T, "oos", ["sharpe", "vs_agg", "p"],
                   ["Sharpe", "vs the Agg", "p"])
        t.insert(0, "class", T["class"])
        t.insert(1, "dev Sharpe", T["dev_sharpe"].round(3))
        r.table(t.round(4),
                align_right=[c for c in t.columns if c != "class"],
                stars=["p"], row_class=row_kinds(T),
                caption="Holdout, 2016 to 2026, with the development Sharpe "
                        "alongside for reference.")
        r._blocks.append(LEGEND)
    r.prose(
        "The Aggregate returned a Sharpe of <strong>-0.072</strong> across the "
        "holdout, so the bar is low in absolute terms and a great many things "
        "clear it, equal weight included at p = 0.003. Clearing the index over "
        "this particular decade is a weak claim on its own.")
    r.prose(
        "The stronger reading is in the two columns together. <strong>The "
        "risk-based methods are the only ones that score well on development "
        "and hold it.</strong> Hierarchical risk parity is top of the "
        "development table and still clears the index out of sample; equal risk "
        "contribution does the same at p = 0.008. Everything in the "
        "forecast-driven group either scored poorly on development, or scored "
        "well and gave it back.")
    r.prose(
        "One row shows why development ranking has to come first. <strong>The "
        "gradient boosting tilt clears the Aggregate on the holdout at "
        "p = 0.014, better than either risk parity method</strong>, having "
        "ranked eighth on development with a p-value of 0.47. Nothing about it "
        "was promising when the choice had to be made. With twenty-one "
        "candidates and a benchmark at a negative Sharpe, some of them will "
        "clear the index out of sample by chance, and picking them after the "
        "fact is the error the development sample exists to prevent.")

    r.section("Comparing at constant risk", (
        "A Sharpe ratio earned at 2.8% volatility and one earned at 4.2% are "
        "not the same claim. Every comparison below puts the strategies on the "
        "index's own risk before ranking them."))
    r.prose(
        "The Aggregate runs at 4.17% annualised volatility over the full sample "
        "and hierarchical risk parity at 2.82%, so the latter is scaled up by "
        "1.48 times to sit at the same risk, with financing charged on the "
        "borrowed portion at the per-asset rates in Appendix B. Anything "
        "already running hotter than the index is scaled down instead, with the "
        "balance held in cash.")
    if L is not None:
        keep = [c for c in ["full_leverage", "full_lev_cagr", "full_lev_vol",
                            "full_lev_sharpe", "full_lev_dd", "full_lev_vs_agg"]
                if c in L.columns]
        t = L[keep].copy()
        t.columns = ["scaling", "return", "volatility", "Sharpe",
                     "worst drawdown", "vs the Agg"][:len(keep)]
        t.index.name = "strategy"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Full sample, every series at the Aggregate's own 4.17% "
                        "volatility, financing charged.")
    if C is not None:
        roles = {c: ("benchmark" if c in ("1/N", "Agg index (VBMFX)",
                                          "2s10s barbell 50/50")
                     else "hero" if "Hierarchical" in c else "strategy")
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
                 "Growth of one dollar at constant risk, so vertical position "
                 "corresponds to risk-adjusted performance. Marked line is "
                 "hierarchical risk parity; reference portfolios are dotted and "
                 "dashed.")

    r.section("Significance", (
        "Monthly returns are serially dependent, so a textbook standard error "
        "on a Sharpe ratio is too narrow. A stationary block bootstrap with "
        "twelve month expected blocks, paired so the common market move is "
        "differenced away."))
    if Bt is not None:
        for tag, win in [("dev", "Development"), ("oos", "Holdout"),
                         ("full", "Full sample")]:
            t = window(Bt, tag, ["edge", "lo", "hi", "p"],
                       ["edge vs the Agg", "CI low", "CI high", "p"])
            r.table(t.round(4), align_right=list(t.columns), stars=["p"],
                    caption=win)
    r.prose(
        "Hierarchical risk parity clears the Aggregate in all three windows: "
        "+0.122 on development, +0.115 on the holdout and +0.153 over the full "
        "sample at p = 0.002. Equal risk contribution is not significant on "
        "development alone but is on both the holdout and the full sample.")
    r.prose(
        "Worth holding alongside that: <strong>equal weight also beat the "
        "Aggregate over the holdout</strong>, at +0.103. Over that decade "
        "almost any diversified bond book beat the index, so clearing it out of "
        "sample is a real result against the benchmark a mandate uses and a "
        "weaker claim about skill than it first appears. Measured against equal "
        "weight instead, the holdout margin is +0.012 with an interval spanning "
        "zero.")

    r.section("Conditions over the holdout", (
        "The decade the strategies were tested on contained the worst bond "
        "market in forty years, which constrains what any result measured over "
        "it can establish."))
    r.table(pd.DataFrame([
        ("2021", "-1.75%", "-0.66%", "-0.29%", "-0.32%"),
        ("2022", "-13.24%", "-13.53%", "-7.59%", "-9.95%"),
        ("2023", "+5.62%", "+6.22%", "+5.63%", "+6.06%"),
        ("2021-2023 cumulative", "-9.96%", "-8.75%", "-2.67%", "-4.80%"),
        ("Worst drawdown", "-17.5%", "-17.3%", "-10.3%", "-13.0%"),
    ], columns=["", "Agg index", "equal weight", "Hierarchical RP",
                "Risk parity"]).set_index(""),
        align_right=["Agg index", "equal weight", "Hierarchical RP",
                     "Risk parity"])
    r.prose(
        "The risk-based methods took materially less damage through the rate "
        "shock, losing 7.6% in 2022 against the index's 13.2%. Underweighting "
        "duration is the mechanism, which raises the obvious objection.")

    r.section("Is it a duration bet?", (
        "Risk parity underweights volatile assets, volatility in bonds is "
        "duration, so it holds less duration than the index. If that is the "
        "whole story an investor could hold shorter bonds and skip the "
        "covariance matrix."))
    r.prose(
        "The benchmark was rebuilt to test exactly that: equal weight, scaled "
        "month by month to match each strategy's own portfolio duration, with "
        "the difference held in cash. It carries the same interest rate "
        "exposure as the strategy and none of its structure.")
    if DT is not None:
        d = DT.copy()
        d.index.name = "strategy"
        r.table(d.round(4), align_right=list(d.columns), stars=["dm p"])
    r.prose(
        "<strong>The edge is unchanged.</strong> Hierarchical risk parity goes "
        "from +0.151 against plain equal weight to +0.147 against the duration "
        "matched version at p = 0.008, and equal risk contribution from +0.087 "
        "to +0.088 at p = 0.012. All intervals exclude zero.")

    r.section("What the portfolio actually holds", (
        "A risk parity book is usually described and rarely shown. These are "
        "the realised weights, month by month."))
    if HW is not None and len(HW):
        W = HW[[c for c in HW.columns]].copy()
        W = W[list(W.mean().sort_values(ascending=False).index)]
        fig, ax = charts.new_axes(9.0, 3.9)
        ax.stackplot(W.index, [W[c].to_numpy() * 100 for c in W.columns],
                     labels=list(W.columns),
                     colors=[charts.SERIES[i % len(charts.SERIES)]
                             for i in range(len(W.columns))],
                     edgecolor="none")
        ax.set_ylim(0, 100)
        ax.set_ylabel("weight")
        charts.percent_axis(ax, decimals=0)
        charts.legend(ax, loc="upper center", ncol=6)
        r.figure(charts.to_svg(fig),
                 "Hierarchical risk parity weights over time, largest average "
                 "holding at the bottom.")
        m = pd.DataFrame({
            "average weight": W.mean(),
            "minimum": W.min(),
            "maximum": W.max(),
            "modified duration": pd.Series(DUR).reindex(W.columns),
        })
        m["duration contribution"] = m["average weight"] * m["modified duration"]
        m.index.name = "asset"
        r.table(m.round(4), align_right=list(m.columns),
                caption="Realised weights across the full sample. The last "
                        "column is average weight times modified duration, and "
                        "sums to the portfolio's duration.")
        short = [a for a in ("ust2y", "ig_short") if a in W.columns]
        r.prose(
            f"The book concentrates in the short end: "
            f"<strong>{W[short].mean().sum() * 100:.0f}%</strong> of the "
            f"portfolio sits in the two year Treasury and short investment "
            f"grade combined, against "
            f"{W[[a for a in ('ust30y', 'ig_long') if a in W.columns]].mean().sum() * 100:.0f}% "
            "in the 30 year Treasury and long credit. That is what equalising "
            "risk contributions does when the long end is roughly eight times "
            "more volatile than the short end. Portfolio duration works out "
            f"around {m['duration contribution'].sum():.1f} years, well inside "
            "the index. The weights are also stable, which is why turnover is "
            "low.")

    r.section("Costs, turnover and robustness")
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
        "is marginally best, which says the covariance estimate is stable "
        "enough that monthly re-optimisation is mostly noise.")
    if RB is not None:
        g = RB.groupby(["cov", "lookback"])[["dev_edge", "oos_edge"]].mean()
        r.table(g.round(4), align_right=["dev_edge", "oos_edge"],
                heat=["dev_edge", "oos_edge"],
                caption="27 combinations of covariance estimator, lookback "
                        "window and rebalancing frequency, averaged.")
        pos_dev = int((RB["dev_edge"] > 0).sum())
        pos_oos = int((RB["oos_edge"] > 0).sum())
        r.prose(
            f"<strong>{pos_dev} of 27 combinations are positive on development "
            f"and {pos_oos} of 27 on the holdout.</strong> Every "
            "expanding-window specification is positive on both, and the losses "
            "concentrate in the short sixty month lookback and the EWMA "
            "estimator, which discard the long history that makes the "
            "covariance estimate stable.")
    if FBE is not None:
        keep = [c for c in ["leverage", "pays_bp", "breakeven_bp",
                            "headroom_bp"] if c in FBE.columns]
        t = FBE[keep].copy()
        t.columns = ["scaling", "pays, bp", "breakeven, bp",
                     "headroom, bp"][:len(keep)]
        t.index.name = "strategy"
        r.table(t.round(1), align_right=list(t.columns),
                caption="Financing, with the full per-asset detail in "
                        "Appendix B.")
    r.prose(
        "Financing would have to be several times more expensive than an "
        "institutional cost of funds before the constant-risk comparison "
        "reverses.")

    r.section("Conclusions", (
        "What the evidence supports, and what it does not."))
    r.checks([
        (True, "Risk parity outperforms the Aggregate index",
         "significant in development, holdout and full sample; full sample "
         "p = 0.002 for the hierarchical version and 0.007 for equal risk "
         "contribution"),
        (True, "The margin is not explained by duration",
         "unchanged against a duration matched benchmark, p = 0.008 and 0.012"),
        (True, "It survives costs and institutional financing",
         "turnover under 15% a year, and 90 to 252bp of headroom to the "
         "financing breakeven"),
        (True, "It uses no return forecast",
         "the covariance matrix only, at every point in the construction"),
        (False, "Return forecasting adds anything on this universe",
         "every forecast-driven class fails on the holdout, including the ones "
         "that scored best on development"),
        (False, "The holdout margin demonstrates skill rather than "
                "diversification",
         "equal weight also beat the index over the same decade, and against "
         "equal weight the holdout margin is +0.012 with an interval spanning "
         "zero"),
    ])

    r.section("Next steps")
    r.table(pd.DataFrame([
        ("Replace funds with indices or futures",
         "The universe is mutual funds, which charge 20 to 80 basis points and "
         "carry manager decisions. High yield's daily returns autocorrelate at "
         "0.29 from stale pricing. Index or futures data removes both."),
        ("Add genuinely different exposures",
         "Adding more US bonds reduced measured independence rather than "
         "raising it. TIPS, international sovereigns, emerging market debt and "
         "bank loans are different factors, not more of the same one."),
        ("Test on a second market",
         "If the duration and forecastability relationship is structural it "
         "should appear in gilts and bunds. That is stronger evidence than "
         "another robustness check on the same data."),
        ("Cost aware optimisation",
         "On the parent project, trading a fraction of the way toward the "
         "target each period rather than fully rebalancing raised Sharpe from "
         "0.06 to 0.23 with turnover cut ninefold."),
    ], columns=["step", "why"]).set_index("step"))

    r.section("Limitations")
    r.table(pd.DataFrame([
        ("Mutual funds, not indices",
         "Fees of 20 to 80 basis points and manager idiosyncrasy. Daily returns "
         "autocorrelate, high yield at 0.29, because illiquid bonds are priced "
         "with a lag."),
        ("Eleven assets is a small universe",
         "Appendix A measures how much independent variation it contains, and "
         "the answer is less than the count suggests."),
        ("The holdout was opened once on the parent project",
         "It is clean of any model being fitted to it, but not of having been "
         "seen once before this half began."),
    ], columns=["limitation", "what it means"]).set_index("limitation"))
    paginate(r, "phase3_results")
    return r.render(OUT / "phase3_results.html")


# ---------------------------------------------------------------- appendix A

def appendix_universe():
    r = PhaseReport(
        phase="Appendix A", title="Universe Structure",
        summary=("How much independent variation eleven bond funds actually "
                 "contain, and how forecastability is distributed across them. "
                 "Neither result builds a portfolio. Together they explain why "
                 "the portfolio that works is built from the covariance "
                 "matrix."),
        status="complete", project=PROJECT)

    fipred = get("fi_predictability_risk")
    ranks = get("fi_rank_evidence")
    tests = get("fi_rank_tests")
    eig = get("fi_breadth_evidence")
    breadth = get("fi_breadth_stats")

    r.section("How much of this universe is distinct", (
        "Eleven tickers is not eleven decisions. The question has a defined "
        "answer rather than a rule of thumb."))
    if eig is not None:
        t = eig.copy()
        t.index.name = "component"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Eigenvalues of the correlation matrix of monthly "
                        "excess returns. Eleven assets, so they sum to 11.")
    r.prose(
        "The standard summary of that spectrum is the <strong>participation "
        "ratio</strong>:")
    r.formula(
        "N<sub>eff</sub> &nbsp;=&nbsp; "
        "<span class='t t1'>(&Sigma; &lambda;<sub>i</sub>)&sup2;</span>"
        " &nbsp;/&nbsp; "
        "<span class='t t3'>&Sigma; &lambda;<sub>i</sub>&sup2;</span>",
        "effective number of independent assets")
    r.prose(
        "It has two fixed points that make it readable. Eleven uncorrelated "
        "assets give eleven equal eigenvalues and a ratio of "
        "<code>11&sup2;/11 = 11</code>. Eleven copies of one asset give a "
        "single eigenvalue of 11 and a ratio of <code>11&sup2;/11&sup2; = 1</code>. "
        "Here the eigenvalues sum to 11 and their squares to 62.94, so the "
        "ratio is <code>121 / 62.94 = 1.92</code>.")
    if breadth is not None:
        r.table(breadth, align_right=["value"])
    r.prose(
        "<strong>Read this as a description of the covariance structure, not a "
        "test.</strong> It carries no p-value and it depends on the sample used "
        "to estimate the correlation matrix. Two other measures point the same "
        "way: the first principal component holds 70.4% of the variance and the "
        "average pairwise correlation is 0.66. The practical consequence is "
        "that adding more bonds adds little, and extending the universe from 12 "
        "assets to 26 lowered this measure rather than raising it.")

    r.section("Forecastability against risk", (
        "The relationship the parent project found, measured directly on this "
        "universe."))
    if ranks is not None:
        keep = [c for c in ["oos_r2", "duration", "vol", "rank_r2",
                            "rank_duration", "rank_vol"] if c in ranks.columns]
        t = ranks[keep]
        t.index.name = "asset"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Rank 1 is the most forecastable asset, the longest "
                        "duration and the highest volatility respectively.")
    r.prose(
        "The two rank columns are close to mirrors. The three most forecastable "
        "assets rank 12th, 11th and 9th by duration; the three least "
        "forecastable rank 1st, 2nd and 3rd.")
    if tests is not None:
        t = tests.copy()
        t.index.name = "predictor"
        r.table(t.round(4), align_right=list(t.columns),
                stars=["spearman p", "pearson p"])
    r.prose(
        "Spearman gives -0.958 against duration and -0.818 against volatility. "
        "The parametric versions agree rather than depending on the rank "
        "transform: Pearson is -0.78 on both, and an ordinary least squares fit "
        "of R squared on duration gives a slope of -0.0016 per year with a t "
        "statistic of -3.96. Twelve assets is a small sample and they are not "
        "independent draws, so these p-values are descriptive rather than a "
        "clean hypothesis test.")
    if fipred is not None:
        t = fipred
        fig, ax = charts.new_axes(8.0, 3.6)
        ax.scatter(t["duration"], t["oos_r2"] * 100, s=70,
                   color=charts.SERIES[0], zorder=3)
        for a, row in t.iterrows():
            ax.annotate(a, (row["duration"], row["oos_r2"] * 100),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=charts.MUTED)
        ax.axhline(0, color=charts.MUTED, linewidth=1.0, linestyle="--")
        ax.set_xlabel("modified duration, years")
        ax.set_ylabel("out of sample R squared")
        charts.percent_axis(ax, decimals=1)
        r.figure(charts.to_svg(fig),
                 "Every asset above five years of duration sits below the "
                 "line.")

    r.section("Why this favours a risk-based portfolio", (
        "The two results above combine into the reason risk parity works here "
        "and forecasting does not."))
    r.prose(
        "Skill is concentrated in assets that carry almost none of the risk. "
        "The most forecastable holdings account for a small fraction of "
        "universe variance, while the 30 year Treasury and long credit, which "
        "dominate it, have negative out of sample R squared. A forecast-driven "
        "portfolio therefore faces a choice with no good branch: size positions "
        "by conviction and the book barely moves, or size them to matter and "
        "the risk is dominated by assets the forecast cannot call.")
    r.prose(
        "The mechanism is in the return decomposition. A bond's return splits "
        "into carry, rolldown, duration and convexity, and the first two are "
        "known when you buy it. A short bond's return is mostly those two; a "
        "long bond's is mostly the yield change, which is not known. The same "
        "fact that makes short bonds forecastable makes them low risk.")
    r.prose(
        "If expected returns cannot be used but the risk structure can, the "
        "portfolio should be built from the covariance matrix alone. That is "
        "the argument for the approach in Phase 3, and it is why it sits here "
        "as justification rather than being presented as the origin of the "
        "idea.")
    paginate(r, "appendix_universe")
    return r.render(OUT / "appendix_universe.html")


# ---------------------------------------------------------------- appendix B

def appendix_method():
    r = PhaseReport(
        phase="Appendix B", title="Implementation",
        summary=("Financing charged per asset, the estimation window every "
                 "model shares, and the constant-risk convention. Details that "
                 "change the numbers but would interrupt the argument."),
        status="complete", project=PROJECT)

    FDET = get("fi_financing_detail")
    FROUTE = get("fi_financing_routes")
    FBE = trim(get("fi_financing_breakeven"))
    SENS = trim(get("fi_aligned_spread_sensitivity"))

    r.section("One estimation window", (
        "Every model class burns in for sixty months from the first month of "
        "data."))
    r.prose(
        "Data begins November 1982, so the first weight any model forms is for "
        "November 1987 and every comparison in this project starts there. All "
        "twelve series are complete from the first date with no gaps, so "
        "nothing enters the panel late and no strategy gets a head start.")
    r.prose(
        "This matters because the project previously carried four different "
        "estimation windows across its scripts, which meant the "
        "return-forecasting classes were being judged on a five-year shorter "
        "sample than the risk-based ones. A comparison across model classes has "
        "to hold the sample fixed or it is measuring the sample.")

    r.section("Financing, charged per asset", (
        "What it costs to lever a position depends on what instrument carries "
        "it, the same way transaction costs do. The assumption throughout is an "
        "institutional book with access to repo, listed futures and cleared "
        "swaps."))
    r.table(pd.DataFrame([
        ("Treasuries, 2 to 30 year", "3bp", "Repo or the futures basis. SOFR is "
         "constructed from Treasury general collateral repo, so this is close "
         "to definitionally flat; the GC versus non-GC component of the fixing "
         "has averaged about 3bp since 2018."),
        ("Agency mortgages", "15bp", "TBA dollar rolls and agency repo, a few "
         "basis points wide of Treasury general collateral."),
        ("Investment grade credit", "50bp", "Total return swap at SOFR + 30 to "
         "75bp, plus a 10 to 25bp agent fee where one applies."),
        ("High yield", "65bp", "Same structure, priced wider."),
        ("Municipals", "110bp", "No municipal futures contract and no liquid "
         "municipal total return swap, so the only route is a margin loan "
         "against the fund at SOFR + 50 to 150bp."),
    ], columns=["holding", "over the risk free rate", "route"]).set_index(
        "holding"), align_right=["over the risk free rate"])
    if FDET is not None:
        t = FDET.drop(columns=[c for c in DROP_ROWS if c in FDET.columns])
        t.index.name = "asset"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Per-asset rate and each strategy's average weight. The "
                        "blended cost is the product summed down each column.")
    r.prose(
        "<strong>The blended rates land between 39 and 42 basis points and "
        "barely differ across strategies</strong>, which is not the intuition. "
        "Risk parity tilts hard toward the two year Treasury at 3bp, so it "
        "ought to finance far more cheaply than equal weight. It does not, "
        "because it tilts just as hard toward short investment grade at 50bp "
        "and because none of these methods materially underweight the "
        "municipals that cost 110bp. Municipals are not the risky assets here: "
        "they run at 4.0% and 5.5% volatility against the 30 year Treasury's "
        "13.4%, so equalising risk contributions leaves them near their equal "
        "weight.")
    if FROUTE is not None:
        t = FROUTE.drop(index=[i for i in DROP_ROWS if i in FROUTE.index],
                        errors="ignore").copy()
        t.columns = ["proportional, bp", "overlay, bp"][:t.shape[1]]
        t.index.name = "strategy"
        r.table(t.round(1), align_right=list(t.columns))
    r.prose(
        "Scaling every position proportionally means margin-lending against the "
        "municipal fund, which no desk would do. The alternative is an overlay: "
        "hold the sleeve with no derivative at its cash weight and take the "
        "borrowed exposure only through instruments that can be replicated "
        "synthetically. That costs about 28 basis points rather than 40. "
        "<strong>Every number in this project uses the proportional figure, the "
        "more expensive of the two</strong>, so the financing drag reported is "
        "an upper bound.")

    r.section("What the assumption is worth")
    r.prose(
        "The unlevered comparison does not depend on it at all: hierarchical "
        "risk parity scores 0.659 against the Aggregate's 0.518 over the full "
        "sample holding no leverage, and every significance test runs on the "
        "unlevered series. The constant-risk comparison does depend on it, and "
        "it has a breakeven.")
    if FBE is not None:
        keep = [c for c in ["leverage", "pays_bp", "breakeven_bp",
                            "headroom_bp"] if c in FBE.columns]
        t = FBE[keep].copy()
        t.columns = ["scaling", "pays, bp", "breakeven, bp",
                     "headroom, bp"][:len(keep)]
        t.index.name = "strategy"
        r.table(t.round(1), align_right=list(t.columns))
    if SENS is not None:
        t = SENS.copy()
        t.index.name = "strategy"
        r.table(t.round(3), align_right=list(t.columns),
                caption="Sharpe at constant risk under a flat spread applied to "
                        "every asset, so the per-asset result can be located "
                        "against the simpler assumption.")
    r.prose(
        "Equal risk contribution pays 41 basis points against a breakeven of "
        "294, so financing would have to be roughly seven times more expensive "
        "before the result reverses. The hierarchical version has less headroom "
        "because it needs more scaling to reach the index's risk.")
    r.prose(
        "<strong>The more leverage a strategy needs, the more of its edge "
        "belongs to whoever finances it.</strong> That restates what produces "
        "the result rather than undermining it. Frazzini and Pedersen's account "
        "of the low beta anomaly is that it persists <em>because</em> most "
        "investors cannot lever cheaply, so a strategy harvesting it should "
        "work for an institution financing at repo and stop working for someone "
        "paying 150 over.")

    r.section("References")
    r.prose(
        "Lopez de Prado, M. (2016). Building Diversified Portfolios that "
        "Outperform Out of Sample. <em>Journal of Portfolio Management</em>, "
        "42(4), 59-69.")
    r.prose(
        "Frazzini, A. and Pedersen, L. H. (2014). Betting Against Beta. "
        "<em>Journal of Financial Economics</em>, 111(1), 1-25.")
    r.prose(
        "DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal Versus Naive "
        "Diversification: How Inefficient is the 1/N Portfolio Strategy? "
        "<em>Review of Financial Studies</em>, 22(5), 1915-1953.")
    r.prose(
        "Ledoit, O. and Wolf, M. (2004). Honey, I Shrunk the Sample Covariance "
        "Matrix. <em>Journal of Portfolio Management</em>, 30(4), 110-119.")
    paginate(r, "appendix_method")
    return r.render(OUT / "appendix_method.html")


# ------------------------------------------------------------------- index

def index():
    A = trim(get("fi_aligned_table"))
    Bt = trim(get("fi_aligned_bootstrap"))

    body = ""
    if A is not None:
        for name in A.index:
            cells = []
            for tag in ["dev", "oos", "full"]:
                sh = A.loc[name, f"{tag}_sharpe"]
                sub = "benchmark"
                if Bt is not None and name in Bt.index:
                    e = Bt.loc[name, f"{tag}_edge"]
                    p = Bt.loc[name, f"{tag}_p"]
                    mk = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
                    sub = f"{e:+.3f}{mk}"
                cells.append(f'{sh:.3f}<span class="d">{sub}</span>')
            cls = ' class="me"' if ("RP" in name or "parity" in name) else ""
            body += (f"<tr{cls}><th>{name}</th>"
                     + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    descs = [
        "Where the project comes from, the eleven asset universe, and what this "
        "half adds to the parent study.",
        "Five model classes, how each forms its weights, and how they score on "
        "the development sample.",
        "The held-out decade, the constant-risk comparison, the duration check, "
        "and what the portfolio actually holds.",
        "How much independent variation the universe contains, and how "
        "forecastability is distributed across it.",
        "Per-asset financing, the shared estimation window, and the "
        "constant-risk convention.",
    ]
    cards = "".join(
        f'<a class="card" href="{stem}.html"><span class="num">{label}</span>'
        f'<span class="ttl">{title}</span><span class="dsc">{d}</span></a>'
        for (stem, label, title), d in zip(PAGES, descs))

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
.intro p{{margin:0 0 1.15rem;max-width:42rem}}
.intro b{{color:var(--fg)}}
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
<p><b>The starting point.</b> This project builds on my
<a class="plain" href="{URL_MACRO}">macro portfolio rebuild</a>, which showed how
difficult equity returns are to predict but also produced real evidence that
fixed income returns are predictable. The goal here was to take that evidence
seriously and build a bond-only portfolio that beats its benchmark.</p>

<p><b>The universe.</b> Eleven fixed income assets: four constant maturity
Treasury holdings at 2, 5, 10 and 30 years, four corporate credit sleeves running
from short investment grade through to high yield, and three securitized and
municipal positions covering agency mortgages, intermediate municipals and high
yield municipals. It uses the same walk-forward backtesting framework as the
prior project, so results across the two are directly comparable.</p>

<p><b>What was tested.</b> Carry and momentum signals, univariate regression
forecasts including regime interactions, four machine learning families tuned on
the development sample, regime-conditional estimation, and risk-based
construction. Each return signal was run both as a bounded tilt and as a
standalone portfolio.</p>

<p><b>What worked.</b> Risk parity, which never needs an expected return at all.
I built and tested both classic risk parity and Marcos Lopez de Prado's
Hierarchical Risk Parity, benchmarked against the Bloomberg Aggregate with equal
weight as a second reference.</p>

<p><b>How performance is measured.</b> A Sharpe ratio earned at 2.8% volatility
and one earned at 4.2% are not the same claim, so <b>every comparison is made at
constant risk</b>: each strategy is scaled to the index's own volatility, with
financing charged per asset at an institutional cost of funds.</p>

<p><b>The duration question.</b> Risk parity naturally holds less duration, so
the obvious objection is that this is a bet on shorter bonds. It is not, and
Phase 3 sets out the duration-matched test.</p>
</div>

<h2>Sharpe ratio, and edge against the Aggregate index</h2>
<div class="scroll">
<table>
<thead><tr><th>Strategy</th><th>Development<br>1987-2015</th><th>Holdout<br>2016-2026</th><th>Full sample<br>1987-2026</th></tr></thead>
<tbody>{body}</tbody>
</table>
</div>
<p class="note">Stars mark one-sided bootstrap significance against the
Aggregate: *** below 0.01, ** below 0.05, * below 0.10.</p>

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
    for stale in ["fi26.html", "final_paper.html"]:
        f = OUT / stale
        if f.exists():
            f.unlink()
    for fn in [phase1, phase2, phase3, appendix_universe, appendix_method,
               index]:
        try:
            p = fn()
            print(f"  wrote {Path(p).name}")
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
