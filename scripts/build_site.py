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
    ("appendix_method", "Appendix A", "Implementation"),
    ("appendix_universe", "Appendix B", "Market Dynamics"),
    ("appendix_eval", "Appendix C", "Evaluating Results"),
]

DROP_ROWS = ["Inverse volatility", "2s10s barbell 50/50"]

# Coded signal names carry the construction, which is useful in the data and
# unreadable on a page. This turns them back into English.
ASSET_NAME = {
    "ust2y": "2y Treasury", "ust5y": "5y Treasury", "ust10y": "10y Treasury",
    "ust30y": "30y Treasury", "ig_short": "short IG", "ig": "IG credit",
    "ig_long": "long IG", "hy": "high yield", "mbs": "agency MBS",
    "muni": "municipals", "muni_hy": "high yield municipals",
    "ust3m": "3m bill",
}


def signal_name(code: str) -> str:
    """Translate a coded signal into something readable."""
    import re as _re
    c = code
    if c.startswith("regime_"):
        return f"{c.replace('regime_', '').title()} regime"
    if " x " in c:
        a, b = c.split(" x ", 1)
        return f"{signal_name(a)} in {b.title()}"
    m = _re.match(r"d_rev(\d+)_(.+)$", c)
    if m:
        return f"{m.group(1)}-day reversal, {ASSET_NAME.get(m.group(2), m.group(2))}"
    m = _re.match(r"d_mom(\d+)s(\d+)_(.+)$", c)
    if m:
        return (f"{int(m.group(1)) // 21}-month momentum, "
                f"{ASSET_NAME.get(m.group(3), m.group(3))}")
    m = _re.match(r"d_mom(\d+)_(.+)$", c)
    if m:
        d = int(m.group(1))
        span = f"{d}-day" if d < 21 else f"{d // 21}-month"
        return f"{span} momentum, {ASSET_NAME.get(m.group(2), m.group(2))}"
    m = _re.match(r"d_ma(\d+)_(\d+)_(.+)$", c)
    if m:
        return (f"{m.group(1)} vs {m.group(2)} day moving average, "
                f"{ASSET_NAME.get(m.group(3), m.group(3))}")
    m = _re.match(r"d_rvol(\d+)_(.+?)(_z)?$", c)
    if m:
        z = ", z-score" if m.group(3) else ""
        return (f"{m.group(1)}-day realized volatility, "
                f"{ASSET_NAME.get(m.group(2), m.group(2))}{z}")
    named = {
        "d_vix_chg5": "VIX, 5-day change",
        "d_vix_chg1": "VIX, 1-day change",
        "d_vix_z": "VIX, z-score",
        "d_vix": "VIX, level",
        "d_nfci_chg": "Chicago Fed financial conditions, change",
        "d_nfci": "Chicago Fed financial conditions, level",
        "d_slope_2s10s_chg5": "2s10s curve slope, 5-day change",
        "d_slope_10y3m_chg5": "10y minus 3m slope, 5-day change",
        "d_slope_2s10s": "2s10s curve slope",
        "d_slope_10y3m": "10y minus 3m slope",
        "d_cp_factor": "Cochrane-Piazzesi forward rate factor",
    }
    if c in named:
        return named[c]
    return c.replace("d_", "").replace("_", " ")


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
        "benchmark": "k-bench", "regression": "k-reg",
        "machine learning": "k-ml", "technical": "k-tech"}


def row_kinds(T):
    """Colour the row rail by what the model has to estimate."""
    if T is None or "class" not in T.columns:
        return {}
    return {i: KIND.get(c, "k-return") for i, c in T["class"].items()}


LEGEND = ('<div class="legend">'
          '<span class="l-risk">risk parity</span>'
          '<span class="l-reg">regression</span>'
          '<span class="l-ml">machine learning</span>'
          '<span class="l-tech">technical and factor</span>'
          '<span class="l-bench">benchmark</span></div>')


# ------------------------------------------------------------------ phase 1

def phase1():
    r = PhaseReport(
        phase="Phase 1", title="The Idea",
        summary=("Extension from lessons learned in my macro portfolio "
                 "project. Asset universe and data collection."),
        status="complete", project=PROJECT)

    stats = get("fi_daily_stats")
    mskill = macro("predict_skill")

    r.metrics([
        ("11", "assets in the universe", None),
        ("10,944", "daily observations", None),
        ("1982-2026", "data available", None),
        ("1987-2026", "evaluated, after burn-in", None),
        ("+2.1%", "2y Treasury monthly out of sample R squared", "pass"),
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

    r.section("Predictive skill by instrument", (
        "The result from the parent project that pointed here. Out of sample R "
        "squared measures a forecast against a rolling mean: positive means the "
        "model beat simply predicting the average, and the scale is small by "
        "construction because monthly asset returns are mostly noise."))
    if mskill is not None:
        t = (mskill[["oos_r2"]] * 100).copy()
        t.columns = ["out of sample R squared, %"]
        if "cw_p" in mskill.columns:
            t["Clark-West p"] = mskill["cw_p"]
        t.index.name = "asset"
        t = t.sort_values("out of sample R squared, %", ascending=False)
        r.table(t.round(3), align_right=list(t.columns),
                heat=["out of sample R squared, %"], stars=["Clark-West p"],
                caption="Monthly forecasts from the parent project's univariate "
                        "combination across 152 signals, 1980 to 2015. "
                        "P-values are from the Clark-West test, used because "
                        "the rolling-mean benchmark is nested inside the "
                        "forecast: the larger model adds parameters whose true "
                        "value may be zero, which injects estimation noise and "
                        "biases a conventional test against it. Clark and West "
                        "(2007) correct for that bias.")
    r.prose(
        "<strong>The two year Treasury is the most forecastable asset at +2.1%, "
        "and the S&amp;P 500 is the least at zero.</strong> Every fixed income "
        "instrument except the 30 year is positive; the equity index is not.")
    r.prose(
        "Two percent forecastability of monthly forward returns sounds small, "
        "but given the inherent variance of asset returns it is a compelling "
        "result. The benchmark for this is Campbell and Thompson (2008), who "
        "showed that a monthly out of sample R squared of about "
        "<strong>0.5% is already economically meaningful</strong> for a "
        "mean-variance investor, and that an R squared as low as 0.25% can "
        "raise average monthly portfolio return by roughly a fifth in "
        "proportional terms. On that scale the two year Treasury's 2.1% is "
        "several times the threshold at which a forecast starts being worth "
        "acting on.")
    r.prose(
        "The same holds in the bond literature specifically. Cochrane and "
        "Piazzesi (2005) report in sample R squared up to 40% on annual excess "
        "bond returns from a forward rate factor, but the out of sample "
        "versions of that work land in low single digits once the coefficients "
        "have to be estimated in real time. Low single digit out of sample R "
        "squared is the normal range for a genuine return forecast, not a "
        "disappointing one.")

    r.section("The asset universe", (
        "Eleven assets at daily frequency. Data begins November 1982 and every "
        "model burns in for five years, so the first weight is formed in "
        "November 1987 and every comparison in this project starts there."))
    r.prose(
        "<strong>Everything runs on daily data.</strong> Both underlying "
        "sources are daily: the constant maturity Treasury holdings come from a "
        "bootstrapped zero curve built on daily Treasury quotes, and the funds "
        "report a daily net asset value. That gives 10,944 daily observations "
        "against 526 month-ends. The difference matters most for the covariance "
        "matrix, which is what the risk-based strategies are built from: an "
        "eleven asset matrix has 66 free parameters, and a five year burn-in "
        "supplies 60 monthly observations against roughly 1,260 daily ones.")
    r.table(pd.DataFrame([
        ("Treasuries", "4", "2, 5, 10 and 30 year constant maturity holdings, "
         "built from a bootstrapped zero curve on FRED constant maturity "
         "quotes, so they differ only in maturity"),
        ("Corporate credit", "4", "Short investment grade (VFSTX), "
         "intermediate (FBNDX), long (VWESX) and high yield (VWEHX)"),
        ("Securitized and municipal", "3", "GNMA agency mortgages (VFIIX), "
         "intermediate municipals (VWITX), high yield municipals (VWAHX)"),
    ], columns=["group", "n", "composition"]).set_index("group"))
    if stats is not None:
        cols = [c for c in ["group", "ann_return", "ann_vol", "sharpe",
                            "autocorr_1"] if c in stats.columns]
        t = stats[cols].copy()
        t.index.name = "asset"
        t.columns = ["group", "annualized return", "annualized volatility",
                     "Sharpe", "autocorrelation"][:len(cols)]
        r.table(t.round(4), align_right=[c for c in t.columns if c != "group"],
                heat=["autocorrelation"],
                caption="Daily asset statistics, November 1982 to August 2026, "
                        "annualized. Return is the arithmetic mean times 252. "
                        "Sharpe is mean excess return divided by the standard "
                        "deviation of excess returns. Autocorrelation is the "
                        "first-order autocorrelation of daily returns.")
    r.prose(
        "The autocorrelation column reflects how each market trades. "
        "Treasuries sit between 0.01 and 0.03, which is what a continuously "
        "quoted market looks like. High yield and the two municipal funds run "
        "between 0.24 and 0.25, because those markets trade thinly and are "
        "marked by matrix pricing rather than by transactions. That is a "
        "property of the asset class rather than of these particular funds, and "
        "its consequences are set out in the limitations in Phase 3.")
    r.prose(
        "<strong>The risk free rate is the 3 month Treasury bill.</strong> The "
        "1 month bill would be the more natural choice for a daily series, but "
        "the Federal Reserve only publishes it from July 2001 and this sample "
        "begins in November 1982. The 3 month bill covers the whole period, and "
        "the difference between the two averages a few basis points where they "
        "overlap.")

    r.section("Extension beyond the original project", (
        "What was already built, and what this project adds to it."))
    r.prose(
        "<strong>Carried over from the macro project.</strong> The fixed income "
        "machinery was built there and is reused here unchanged. Treasury par "
        "yields are bootstrapped into a zero curve. Constant maturity holdings "
        "are then "
        "priced off that single discount function, so a 2 year and a 30 year "
        "differ in maturity and nothing else rather than being two unrelated "
        "vendor series. Each holding genuinely ages and is rolled back to "
        "target, which is what a bond index does, and the resulting return is "
        "decomposed into <strong>carry</strong>, <strong>rolldown</strong>, "
        "<strong>duration</strong> and <strong>convexity</strong>. The "
        "walk-forward backtesting engine, the per-asset transaction costs and "
        "the sealed holdout come across intact, which is what makes results "
        "from the two projects directly comparable.")
    r.prose(
        "<strong>New here.</strong> Three things:")
    r.table(pd.DataFrame([
        ("A fixed income universe", "Eleven assets rather than the parent "
         "project's seven, and all of them bonds: four Treasury maturities, "
         "four corporate credit sleeves, agency mortgages and two municipal "
         "holdings. The credit and municipal sleeves are new instruments, not "
         "reweighted versions of what came before."),
        ("Bond-specific signals and factors", "Signals that are distinctly "
         "useful for forecasting fixed income and have no counterpart in "
         "equities or commodities: carry and rolldown by maturity, modified "
         "duration, forward rates at every curve node, the Cochrane-Piazzesi "
         "forward rate factor, and yield curve principal components as level, "
         "slope and curvature. The parent project's macro and credit signals "
         "are kept alongside them."),
        ("Return forecasting as well as risk-based approaches", "The parent "
         "project spent most of its effort forecasting returns and concluded "
         "that was the wrong place to spend it. This one tests forecasting "
         "again on a universe where it should work better, and also looks at "
         "portfolio construction via the covariance matrix."),
    ], columns=["addition", "what it means"]).set_index("addition"))

    r.next_up("Phase 2 - Strategies", [
        "Five model classes and how each forms its weights",
        "Machine learning model tuning",
        "Development results, and what goes forward",
    ])
    paginate(r, "phase1_idea")
    return r.render(OUT / "phase1_idea.html")


# ------------------------------------------------------------------ phase 2

def phase2():
    r = PhaseReport(
        phase="Phase 2", title="Strategies",
        summary=("Four families of model on one universe, one estimation "
                 "window and one benchmark. Everything on this page is measured "
                 "on the development sample only; the holdout is opened in "
                 "Phase 3."),
        status="complete", project=PROJECT)

    T = get("fi_dmodel_summary")
    ML = get("fi_dmodel_ml_chosen")
    MLS = get("fi_dmodel_ml_skill")
    UNI = get("fi_dmodel_regression_skill")
    RANK = get("fi_regressor_ranks")
    FAM = get("fi_regressor_families")
    TOP3 = get("fi_regressor_top3")
    RECUR = get("fi_regressor_recurring")
    FAMB = get("fi_regressor_families_both")
    UNI_M = get("fi_uni_regression_skill")
    MLS_M = get("fi_uni_ml_skill")
    REB = get("fi_rebal_summary")

    bd = None
    if T is not None and {"dev_sharpe", "dev_vs_agg"} <= set(T.columns):
        bd = float((T["dev_sharpe"] - T["dev_vs_agg"]).median())

    r.metrics([
        ("4", "model families", None),
        ("2", "ways of forming weights", None),
        (f"{bd:.3f}" if bd is not None else "0.807", "Agg Sharpe, development",
         None),
        ("60", "month burn-in, every model", None),
        ("development", "this page only", None),
    ])

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
         "cumulative moments rather than refitted each period."),
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
        t = (UNI * 100)[["dev_r2"]].copy()
        t.columns = ["daily"]
        if UNI_M is not None:
            t["monthly"] = (UNI_M * 100)["dev_r2"].reindex(t.index)
        t.index.name = "asset"
        r.table(t.round(3), align_right=list(t.columns), heat=list(t.columns),
                caption="Development sample only, out of sample R squared in "
                        "percent. The comparison is against a rolling mean "
                        "forecast: at each date, predict the average return so "
                        "far and see whether the model beats it. Zero means the "
                        "model added nothing; negative means it did worse than "
                        "assuming the future looks like the past.")
        r.prose(
            "<strong>Monthly returns are more forecastable than daily "
            "ones.</strong> The two year Treasury scores 1.08% monthly against "
            "0.08% daily, and the ordering holds across the universe. A single "
            "day of bond returns is mostly quoting and flow noise. A month "
            "accumulates enough carry, curve movement and spread change for a "
            "signal to attach to.")
    r.prose(
        "The municipal holdings are the exception and should be read with care. "
        "They score higher at daily than at monthly frequency, which reverses "
        "the pattern everywhere else. Municipal bonds are priced by matrix "
        "valuation rather than by trades, so their reported daily returns "
        "autocorrelate at around 0.25 and part of what a daily model predicts "
        "is a price change that has already occurred. Monthly aggregation "
        "removes most of that.")

    r.section("Which signals do the work", (
        "The combination averages every signal, so it is worth seeing what "
        "sits underneath it. Each signal was scored on its own, one "
        "expanding-window regression per signal per asset, on the development "
        "sample."))
    r.prose(
        "Ranking 256 signals by R squared would produce a leaderboard whether "
        "or not anything predicts, and reporting the winner for each of eleven "
        "assets gives 110 results to read. The more useful question is which "
        "signals recur. Taking each asset's ten best and keeping only those "
        "that appear for three or more assets reduces the field to ten.")
    if RECUR is not None:
        t = RECUR.copy()
        # A k-day momentum signal and a k-day reversal signal are the same
        # quantity with opposite sign, so they score identically. Keep one.
        t = t[~t.index.str.startswith("d_rev5_")]
        t.index = [signal_name(x) for x in t.index]
        t["best_on"] = [ASSET_NAME.get(a, a) for a in t["best_on"]]
        t["median_r2"] = t["median_r2"] * 100
        t["best_r2"] = t["best_r2"] * 100
        t.columns = ["family", "assets in top ten", "median R squared, %",
                     "best R squared, %", "best on"]
        t.index.name = "signal"
        r.table(t.round(3),
                align_right=["assets in top ten", "median R squared, %",
                             "best R squared, %"],
                heat=["median R squared, %"],
                caption="Signals appearing in the ten best for three or more "
                        "assets, daily data, development sample. Momentum and "
                        "reversal over the same horizon are one quantity with "
                        "opposite sign, so only one of each pair is shown.")

    r.prose(
        "<strong>Volatility and financial conditions account for six of the ten "
        "recurring signals</strong>, and the five day change in the VIX appears "
        "for seven of the eleven assets, more than any other. Nothing in the "
        "yield curve or carry families recurs at all. Whatever forecastability "
        "exists at daily frequency comes from the risk environment rather than "
        "from the term structure.")
    r.prose(
        "Signals defined on a single asset also predict others. The five day "
        "return on investment grade credit appears in the top ten for four "
        "assets, and realized volatility on short investment grade for three. "
        "In a universe where the first principal component explains most of the "
        "variance, a signal built on one holding is partly a signal about the "
        "common factor.")
    r.prose(
        "The magnitudes divide along the same line. The best signal reaches "
        "0.2% on the Treasuries and 1.5% on long credit. On high yield it "
        "reaches 7.5% and on intermediate municipals 6.3%, a difference "
        "attributable to the pricing delay rather than to forecastability.")
    if FAMB is not None:
        t = FAMB.copy()
        for c2 in t.columns:
            if c2 != "signals":
                t[c2] = t[c2] * 100
        t.columns = [c2.replace("mean_r2_", "mean, ").replace("best_r2_", "best, ")
                     .replace("signals", "signals in family") for c2 in t.columns]
        t.index.name = "family"
        r.table(t.round(3), align_right=list(t.columns),
                heat=[c2 for c2 in t.columns if c2 != "signals in family"],
                caption="By family, both frequencies, development sample. R "
                        "squared in percent. A blank means the family has no "
                        "counterpart in that library: short-term reversal is a "
                        "daily construction, and valuation and term premium are "
                        "monthly.")
    r.prose(
        "The monthly means are negative in every family while the monthly best "
        "values are the highest anywhere in the table, reaching 2.3% for macro "
        "signals and 2.3% for the term premium. Individual monthly signals are "
        "noisy, and most of them lose to a rolling mean. That is the case for "
        "combining them rather than picking one.")

    r.section("Machine learning, and how it was tuned", (
        "Four families from scikit-learn on the same signal panel: elastic "
        "net, ridge, random forest and gradient boosting. Each is run across a "
        "grid of hyperparameters, the configuration with the best development "
        "R squared is selected, and only that one is scored. The holdout is "
        "never consulted in the choice."))
    if MLS is not None:
        t = (MLS * 100)["dev_r2"].unstack()
        t.index.name = "family"
        stale = [c for c in ["hy", "muni", "muni_hy"] if c in t.columns]
        clean = [c for c in t.columns if c not in stale]
        t = t[clean + stale]
        r.table(t.round(2), align_right=list(t.columns), heat=list(t.columns),
                caption="Out of sample R squared by asset, percent, "
                        "development sample. The three right-hand columns are "
                        "the municipal and high yield funds, whose prices are "
                        "set by matrix valuation rather than by trades.")
        r.prose(
            "<strong>Out of sample and development are not in conflict.</strong> "
            "<em>Out of sample</em> describes how each forecast was formed: the "
            "model that predicts a given day was fitted only on data before it, "
            "so no observation is used to predict itself. <em>Development</em> "
            "describes the date range over which those forecasts are then "
            "scored, 1987 to 2015. The holdout is a further period, 2016 "
            "onward, which no model or model choice has seen at all. So a "
            "figure can be out of sample in the walk-forward sense and still "
            "sit inside the development window.")
    if ML is not None:
        picked = ", ".join(
            f"{i.replace('_', ' ')}: {row['params']}"
            for i, row in ML.iterrows()) if "params" in ML.columns else ""
        r.prose(
            f"<span class=\"note\">Hyperparameters selected on development: "
            f"{picked}.</span>")
    r.prose(
        "<strong>The per-asset view explains the aggregate.</strong> Averaged "
        "across all eleven assets the families look mildly positive, at +2.0% "
        "for the elastic net and +1.9% for the random forest. Splitting the "
        "universe reverses that: on the eight holdings that trade continuously "
        "every family is negative, and on high yield and the two municipal "
        "funds every family is strongly positive.")
    r.table(pd.DataFrame([
        ("Elastic net", "+2.03", "+7.67", "-0.09"),
        ("Random forest", "+1.90", "+7.83", "-0.33"),
        ("Ridge", "+1.60", "+8.39", "-0.95"),
        ("Gradient boosting", "+1.13", "+4.46", "-0.12"),
    ], columns=["family", "all 11 assets", "high yield and municipals",
                "the other 8 assets"]).set_index("family"),
        align_right=["all 11 assets", "high yield and municipals",
                     "the other 8 assets"],
        caption="Out of sample R squared, percent, daily data, development "
                "sample.")
    r.prose(
        "This also accounts for the daily figures exceeding the monthly ones "
        "for these models, which is the opposite of the regression result. "
        "Monthly aggregation averages the pricing delay away; a daily series "
        "retains it, and a flexible model locates it quickly. <strong>The daily "
        "advantage is the pricing delay rather than forecast skill.</strong>")
    r.prose(
        "The regression does not show this pattern. Averaging 256 univariate "
        "fits dilutes any single artefact rather than concentrating on it, "
        "which is the practical argument for the combination approach.")

    r.section("Two ways of turning a forecast into a portfolio", (
        "A return forecast can guide all of the allocation or only part of it. "
        "Both are tested."))
    r.prose(
        "<strong>Tilt.</strong> Start from equal weight and move away from it "
        "in proportion to the forecast, with a cap on how far any single "
        "position can go. The forecast is standardised across assets first, so "
        "what drives the tilt is which assets look good relative to the others "
        "rather than the level of the forecast:")
    r.formula(
        "w &nbsp;=&nbsp; clip( <span class='t t1'>w<sub>eq</sub></span> "
        "&nbsp;+&nbsp; <span class='t t2'>&Lambda;</span> &middot; "
        "<span class='t t3'>z</span> , &nbsp;bounds )",
        "bounded tilt, lambda = 0.5, cap 15 points per asset")
    r.prose(
        "With eleven assets equal weight is 9.1% each. A lambda of 0.5 means an "
        "asset one standard deviation above the cross-sectional average gets "
        "roughly 13.6% and one a standard deviation below gets 4.5%, before the "
        "15 point cap binds. So <em>Regression, tilt</em> in the results below "
        "is equal weight moved by the combined regression forecast, not a fixed "
        "blend of two portfolios.")
    r.prose(
        "<strong>Base.</strong> The signal alone sets the weights, long only "
        "and normalized to sum to one, with no base allocation underneath.")

    r.section("Risk parity, in detail", (
        "The fourth family, and the only one that never estimates an expected "
        "return. Both members work from the covariance matrix alone."))
    r.prose(
        "<strong>Equal risk contribution.</strong> Instead of putting equal "
        "money in each asset, put equal <em>risk</em> in each asset. A 30 year "
        "Treasury runs at roughly eight times the volatility of a two year, so "
        "it receives roughly one eighth the weight. Formally, solve for the "
        "weights at which every asset's contribution to portfolio variance is "
        "the same:")
    r.formula(
        "<span class='t t1'>w<sub>i</sub></span> &middot; "
        "<span class='t t3'>(&Sigma;w)<sub>i</sub></span> &nbsp;/&nbsp; "
        "&sigma;<sub>p</sub> &nbsp;=&nbsp; constant, for every i",
        "equal risk contribution")
    r.prose(
        "There is no closed form, so it is solved by fixed-point iteration: "
        "start at equal weight, compute each asset's risk contribution, shift "
        "weight away from the assets contributing too much, repeat until the "
        "contributions equalize.")
    r.prose(
        "<strong>Hierarchical risk parity</strong>, from Lopez de Prado (2016), "
        "fixes a specific weakness in that. Equal risk contribution uses the "
        "full covariance matrix, and when assets are as correlated as bonds "
        "are, that matrix is close to singular and the weights it produces "
        "become unstable. It also treats every asset as a peer, so four nearly "
        "identical Treasury maturities register as four separate bets rather "
        "than one bet held four ways.")
    r.table(pd.DataFrame([
        ("1. Cluster", "Convert the correlation matrix into a distance measure "
         "and build a tree by joining the closest assets first. Treasuries end "
         "up next to Treasuries, municipals next to municipals."),
        ("2. Quasi-diagonalize", "Reorder the matrix so similar assets sit "
         "adjacent. The result is close to block diagonal, which makes the "
         "structure usable without inverting anything."),
        ("3. Recursive bisection", "Split the tree in two, allocate between the "
         "halves in inverse proportion to their cluster variance, then repeat "
         "inside each half down to individual assets."),
    ], columns=["step", "what happens"]).set_index("step"))
    r.prose(
        "<strong>The property that matters is that it never inverts the "
        "covariance matrix.</strong> It uses the matrix only to measure "
        "distance between assets and to compute the variance of a cluster. "
        "Inversion is where estimation error gets amplified, and a near "
        "singular matrix amplifies it violently, so avoiding the inversion is "
        "what makes the method stable in exactly the situation this universe "
        "presents.")
    r.table(pd.DataFrame([
        ("Covariance estimator", "Ledoit-Wolf shrinkage toward a constant "
         "correlation target, which pulls the noisiest off-diagonal elements "
         "toward a structured estimate."),
        ("Window", "Expanding from the first day of data. Phase 3 shows this "
         "choice matters more than anything else tested."),
        ("Rebalancing", "Annual, chosen in Phase 3 on the development sample."),
        ("Expected returns", "None. The mean vector never enters either "
         "objective, which is why neither method has a forecast to decay."),
    ], columns=["choice", "what was done"]).set_index("choice"))

    r.section("Development results", (
        "Every model, the full development sample, net of per-asset "
        "transaction costs, measured against the Aggregate."))
    if T is not None:
        t = window(T, "dev", ["sharpe", "vol", "vs_agg", "p"],
                   ["Sharpe", "volatility", "vs the Agg", "p"])
        t.insert(0, "class", T["class"])
        bench_sharpe = float((T["dev_sharpe"] - T["dev_vs_agg"]).median())
        r.table(pd.DataFrame([
            ("Aggregate index", round(bench_sharpe, 4)),
        ], columns=["benchmark", "development Sharpe"]).set_index("benchmark"),
            align_right=["development Sharpe"],
            caption="Every edge below is measured against this.")
        r.table(t.round(4),
                align_right=[c for c in t.columns if c != "class"],
                stars=["p"], row_class=row_kinds(T),
                caption="Sorted by development Sharpe. Stars mark one-sided "
                        "bootstrap significance: *** below 0.01, ** below 0.05, "
                        "* below 0.10.")
        r._blocks.append(LEGEND)
    r.prose(
        "The colored rail on the left separates the two groups. Everything that "
        "needs a return forecast carries one color and everything built from "
        "the covariance matrix another, and the gap between those groups is "
        "wider than the gap between any two individual models.")
    r.prose(
        "Two patterns are worth naming. <strong>Every base variant scores well "
        "below its tilt</strong>: carry falls from 0.61 to 0.27, the regression "
        "from 0.67 to 0.50, momentum from 0.62 to 0.27. The volatility column "
        "explains it. A signal-weighted book concentrates into whatever the "
        "signal likes at that moment, which on a bond universe is usually the "
        "long end, so it runs at 7 to 9% volatility against 5% for the bounded "
        "version. And <strong>turnover separates the two groups as sharply as "
        "performance does</strong>: the risk-based methods trade 1.5 to 2% of "
        "the book a year, the forecast-driven ones 12 to 72%.")

    r.section("From Development to Validation", (
        "Two strategies are carried into the holdout."))
    r.prose(
        "<strong>Equal risk contribution and hierarchical risk parity, both on "
        "the Ledoit-Wolf covariance.</strong> Neither uses a return forecast.")
    r.prose(
        "Two variants were tested and neither is carried forward. Estimating "
        "the covariance within each macro regime moves hierarchical risk parity "
        "from 0.932 to 0.909 and equal risk contribution from 0.884 to 0.888. "
        "Splitting the sample into four states leaves each estimate with a "
        "quarter of the data, and on a matrix this correlated that costs more "
        "precision than the regime structure adds.")
    r.prose(
        "DCC-GARCH fares worse, taking hierarchical risk parity to 0.800 and "
        "equal risk contribution to 0.865. It lets correlations move over time, "
        "which is what Ledoit-Wolf shrinkage is designed to damp. On eleven "
        "bond funds that share one dominant factor, most of that movement is "
        "estimation noise.")
    r.prose(
        "Nothing from the return-forecasting side is carried. Out of sample R "
        "squared near one percent is real but too small to size a position on, "
        "and the base-weighted versions show what happens when it is allowed "
        "to try.")

    r.next_up("Phase 3 - Results", [
        "Validating strategies on the holdout data",
        "Trading and leverage costs",
        "Portfolio characteristics and conclusion",
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

    T = get("fi_dmodel_summary")
    REB = get("fi_rebal_summary")
    OVL = get("fi_overlay_summary")
    L = trim(get("fi_aligned_levered"))
    Bt = trim(get("fi_aligned_bootstrap"))
    C = get("fi_aligned_curves")
    DT = trim(get("fi_rp_duration_test"))
    TS = trim(get("fi_paper_turnover_sharpe"))
    TO = trim(get("fi_paper_turnover"))
    HW = get("fi_canonical_hrp_weights")
    HD = get("fi_canonical_hrp_duration")

    r.metrics([
        ("2,660", "holdout trading days", None),
        ("-0.081", "Agg Sharpe on the holdout", "fail"),
        ("+0.146", "hierarchical RP edge", "pass"),
        ("+0.175", "equal risk contribution edge", "pass"),
        ("0.060", "best holdout p-value", "pass"),
    ])

    r.section("The holdout", (
        "Every model from Phase 2 on data none of them was fitted to."))
    if T is not None:
        keep = [i for i in T.index if i.endswith(", annual")]
        t = window(T.loc[keep], "oos", ["sharpe", "vol", "vs_agg", "p"],
                   ["Sharpe", "volatility", "vs the Agg", "p"])
        t.insert(0, "dev Sharpe", T.loc[keep, "dev_sharpe"].round(3))
        r.table(t.round(4), align_right=list(t.columns), stars=["p"],
                caption="Holdout, 2016 to 2026, for the two strategies carried "
                        "forward from development.")
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

    r.section("Rebalancing frequency", (
        "Daily data separates estimating the covariance from trading on it. The "
        "matrix is estimated on every day available; the portfolio trades on "
        "whatever schedule costs justify."))
    if REB is not None:
        keep = [i for i in REB.index if i.startswith(("HRP, ", "ERC, "))
                and "rolling" not in i and "expanding" not in i]
        t = REB.loc[keep, [c for c in ["dev_sharpe", "oos_sharpe", "oos_vs_agg",
                                       "trades", "turnover"] if c in REB.columns]]
        t.columns = ["dev Sharpe", "holdout Sharpe", "holdout vs Agg",
                     "trades", "turnover"][:t.shape[1]]
        t.index.name = "strategy"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Expanding covariance window, all eleven assets. "
                        "Turnover is what was actually traded, one way, "
                        "annualised.")
    r.prose(
        "<strong>Less trading is better at every step, for both methods and in "
        "both windows.</strong> Annual rebalancing gives the highest Sharpe and "
        "the lowest turnover simultaneously, at 39 trades across thirty-eight "
        "years. Daily rebalancing turns over 15 to 17% of the book a year to "
        "produce a worse result. A bond covariance matrix estimated on an "
        "expanding window barely moves between rebalances, so frequent trading "
        "is mostly paying costs to correct drift that would have reverted.")
    r.prose(
        "One pattern runs through every row: <strong>hierarchical risk parity "
        "leads on development and equal risk contribution leads on the "
        "holdout.</strong> That is consistent across all eight combinations "
        "rather than a single cell, and it is what you would expect if the "
        "clustering step fits structure that does not fully generalise.")

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
        C = C.drop(columns=[c for c in DROP_ROWS if c in C.columns])
        roles = {c: ("benchmark" if c in ("1/N", "Agg index (VBMFX)")
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
                 "Growth of one dollar, every series held at the "
                 "Aggregate's volatility.")

    if OVL is not None:
        r.section("Leverage route", (
            "Scaling to the benchmark's risk means borrowing, and there are two "
            "ways to do it. Appendix A sets out both; this is what each is "
            "worth."))
        t = OVL[[c for c in ["dev_leverage", "dev_bp", "dev_sharpe",
                             "dev_vs_agg", "dev_p", "oos_sharpe", "oos_vs_agg",
                             "oos_p"] if c in OVL.columns]].copy()
        t.columns = ["scaling", "cost bp", "dev Sharpe", "dev vs Agg", "dev p",
                     "holdout Sharpe", "holdout vs Agg", "holdout p"][:t.shape[1]]
        t.index.name = "strategy"
        r.table(t.round(4), align_right=list(t.columns),
                stars=["dev p", "holdout p"],
                caption="Proportional scales every position, including the "
                        "municipal fund. Overlay holds municipals at their cash "
                        "weight and takes the borrowed exposure only through "
                        "instruments that have a derivative.")
        r.prose(
            "The overlay saves about sixteen basis points of borrowing cost, "
            "worth roughly +0.02 of Sharpe in development and +0.03 on the "
            "holdout. It also changes what the levered book holds: the extra "
            "exposure concentrates in Treasuries and credit rather than being "
            "spread across the strategy's own weights, so it is a slightly "
            "different portfolio and not purely a cheaper one. Every headline "
            "number in this project uses the proportional route, which is the "
            "more expensive of the two.")

    r.section("Significance", (
        "A stationary block bootstrap, set out in full in Appendix C: 5,000 "
        "paired resamples in blocks of random length, centred on the null of no "
        "edge, counting how often chance reaches the observed gap."))
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

    r.section("2021-2023 bond bear market performance", (
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
        d.columns = [c.replace("dm p", "p").replace("dm CI low", "CI low")
                     .replace("dm CI high", "CI high")
                     .replace("vs duration-matched 1/N", "vs duration matched")
                     for c in d.columns]
        r.table(d.round(4), align_right=list(d.columns), stars=["p"])
    r.prose(
        "<strong>The edge is unchanged.</strong> Hierarchical risk parity goes "
        "from +0.151 against plain equal weight to +0.147 against the duration "
        "matched version at p = 0.008, and equal risk contribution from +0.087 "
        "to +0.088 at p = 0.012. All intervals exclude zero.")

    r.section("Portfolio characteristics", (
        "A risk parity book is usually described and rarely shown. These are "
        "the realised weights, month by month."))
    if HW is not None and len(HW):
        W = HW[[c for c in HW.columns]].copy()
        W = W[list(W.mean().sort_values(ascending=False).index)]
        fig, ax = charts.new_axes(7.8, 3.9)
        ax.stackplot(W.index, [W[c].to_numpy() * 100 for c in W.columns],
                     labels=list(W.columns),
                     colors=[charts.SERIES[i % len(charts.SERIES)]
                             for i in range(len(W.columns))],
                     edgecolor="none")
        ax.set_ylim(0, 100)
        ax.set_xlim(W.index.min(), W.index.max())
        ax.set_ylabel("weight")
        charts.percent_axis(ax, decimals=0)
        # Legend outside the plot, stacked, ordered like the stack itself so a
        # reader can map label to band without counting colors.
        h, l = ax.get_legend_handles_labels()
        charts.legend(ax, handles=h[::-1], labels=l[::-1], loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), ncol=1, borderaxespad=0)
        r.figure(charts.to_svg(fig),
                 "Hierarchical risk parity weights over time, annual "
                 "rebalancing, largest average holding at the bottom.")
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
        if HD is not None:
            d = HD.squeeze()
            fig, ax = charts.new_axes(9.0, 2.6)
            ax.plot(d.index, d.to_numpy(), **charts.style_for("hero", 0))
            ax.set_ylabel("portfolio duration, years")
            charts.legend(ax, loc="upper right") if False else None
            r.figure(charts.to_svg(fig),
                     f"Hierarchical risk parity portfolio duration over time: "
                     f"mean {d.mean():.2f} years, ranging {d.min():.2f} to "
                     f"{d.max():.2f}. The Aggregate index runs around six.")
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
         "The universe is built from funds, which charge 20 to 80 basis points "
         "and carry manager decisions an index would not. Index or futures data "
         "would remove both."),
        ("Three holdings are marked with a lag",
         "High yield and the two municipal funds autocorrelate between 0.24 and "
         "0.25 at daily frequency, because those markets are matrix priced "
         "rather than traded. Lagged marks understate measured volatility, so a "
         "risk-based method will hold slightly more of those assets than it "
         "would on transaction prices. Correcting the variance for the "
         "autocorrelation moves the holdout edge from +0.146 to +0.108, so the "
         "effect is real but does not carry the result. The headline numbers "
         "apply no correction, which is the less favourable of the two."),
        ("Daily forecast skill on those three is not forecast skill",
         "Flexible models report 8 to 14% out of sample R squared on the stale "
         "holdings and close to zero on the Treasuries. That is the lag being "
         "detected, not the future. It is the reason the duration relationship "
         "in Appendix B is stated at monthly frequency, where the lag washes "
         "out, and the reason no forecast-driven strategy is carried into the "
         "holdout."),
        ("Eleven assets is a small universe",
         "Appendix B measures how much independent variation it contains, and "
         "the answer is less than the count suggests."),
        ("The holdout was opened once on the parent project",
         "It is clean of any model being fitted to it, but not of having been "
         "seen once before this project began."),
    ], columns=["limitation", "what it means"]).set_index("limitation"))
    paginate(r, "phase3_results")
    return r.render(OUT / "phase3_results.html")


# ---------------------------------------------------------------- appendix A

def appendix_universe():
    r = PhaseReport(
        phase="Appendix B", title="Market Dynamics",
        summary=("Decomposing the variance of the universe through principal "
                 "components, and measuring forecastability as a function of "
                 "duration and volatility. Neither result builds a portfolio. "
                 "Together they explain why the portfolio that works is built "
                 "from the covariance matrix."),
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

    r.section("Forecastability as a function of duration and volatility", (
        "Does a bond get harder to forecast as it gets riskier? Two ways of "
        "asking, on the same twelve observations."))
    r.prose(
        "<strong>The setup.</strong> Each of the twelve fixed income assets "
        "contributes one observation. Its <em>y</em> value is that asset's out "
        "of sample R squared, taken from the univariate combination forecast: "
        "one number per asset, measuring how much better the model predicts "
        "that asset's returns than a rolling mean does. Its <em>x</em> value is "
        "a measure of how much risk the asset carries, and two are used "
        "separately: modified duration, and annualised volatility.")
    r.prose(
        "That gives <strong>two bivariate relationships</strong>, not four: "
        "R squared against duration, and R squared against volatility. Each is "
        "then measured two ways, which is where the four numbers in the table "
        "come from.")
    r.table(pd.DataFrame([
        ("Spearman rho", "Correlation of the <em>ranks</em> rather than the "
         "values. Rank the twelve assets by R squared, rank them again by "
         "duration, and measure how closely the two orderings agree. This is "
         "the headline because the claim is about ordering, and because ranks "
         "are not distorted by one asset with an extreme duration."),
        ("Pearson r", "Correlation of the values themselves. Included as a "
         "check that the result is not an artefact of the rank transform."),
        ("OLS slope and t", "An ordinary least squares fit of R squared on the "
         "risk measure, across the twelve assets. The slope says how much "
         "predictive R squared is lost per year of duration; the t statistic "
         "says whether that slope is distinguishable from zero."),
        ("p-values", "One for each correlation, testing the null that the two "
         "orderings are unrelated."),
    ], columns=["statistic", "what it measures"]).set_index("statistic"))
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
        "Reading the table: <strong>Spearman gives -0.958 against duration and "
        "-0.818 against volatility</strong>, so the two orderings run almost "
        "exactly opposite. Pearson is -0.78 on both, so the result does not "
        "depend on the rank transform. The regression slope of -0.0016 means "
        "each additional year of duration costs about 16 basis points of "
        "predictive R squared, with a t statistic of -3.96 across twelve "
        "observations.")
    r.prose(
        "Twelve assets is a small sample and they are not independent draws "
        "from anything, so these p-values are descriptive rather than a clean "
        "hypothesis test. The reason to take the ordering seriously is that it "
        "also holds across model families that share no functional form, and "
        "that it has a mechanical explanation rather than only a statistical "
        "one.")
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
        phase="Appendix A", title="Implementation",
        summary=("Burn-in and estimation windows. Transaction costs and "
                 "leverage costs, both charged per asset. The constant-risk "
                 "convention every comparison is made under."),
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


    r.section("Transaction costs", (
        "Charged per asset on realised turnover, because a Treasury and a high "
        "yield municipal do not trade at the same spread."))
    r.table(pd.DataFrame([
        ("ust2y", "2bp", "On-the-run Treasury, the tightest market in fixed "
         "income"),
        ("ust5y", "3bp", "Treasury"),
        ("ust10y", "3bp", "Treasury"),
        ("ust30y", "4bp", "Treasury, wider at the long end"),
        ("mbs", "12bp", "Agency mortgages, TBA market"),
        ("ig_short", "15bp", "Short investment grade credit"),
        ("ig", "20bp", "Intermediate investment grade"),
        ("ig_long", "25bp", "Long investment grade, wider with duration"),
        ("muni", "25bp", "Intermediate municipals, thinly traded"),
        ("hy", "40bp", "High yield corporate"),
        ("muni_hy", "45bp", "High yield municipals, the widest in the "
         "universe"),
    ], columns=["asset", "round trip", "why"]).set_index("asset"),
        align_right=["round trip"])
    r.prose(
        "These are round-trip estimates and they are charged on every trade, "
        "so a strategy that rebalances daily pays them 252 times a year. That "
        "is why the rebalancing comparison in Phase 3 matters: annual "
        "rebalancing turns over about 2% of the book a year against 15% for "
        "daily, and the cost difference is larger than the difference in gross "
        "performance.")

    r.section("Leverage costs", (
        "Why leverage enters at all: a Sharpe ratio earned at 2.8% volatility "
        "and one earned at 4.2% are not comparable claims, so every strategy is "
        "scaled to the benchmark's own volatility before being ranked. Scaling "
        "up means borrowing, and borrowing costs money, so the only honest "
        "version of that comparison charges for it."))
    r.prose(
        "This matters more in fixed income than almost anywhere else. Frazzini "
        "and Pedersen's account of the low beta anomaly is that low volatility "
        "assets earn better risk-adjusted returns precisely <em>because</em> "
        "most investors cannot borrow cheaply: an investor who needs a return "
        "target and cannot lever must reach for volatile assets instead, "
        "bidding them up and leaving the low volatility ones cheap. A strategy "
        "that harvests that anomaly is therefore only available to someone who "
        "can finance it. <strong>The assumption throughout is an institutional "
        "book</strong> with access to repo, listed futures and cleared swaps, "
        "not a retail margin account.")
    r.prose(
        "What it costs then depends on what instrument carries the position, "
        "the same way transaction costs do.")
    r.table(pd.DataFrame([
        ("Treasuries, 2 to 30 year", "3bp", "Repo or the futures basis. "
         "<em>General collateral</em> repo is lending against any Treasury the "
         "borrower chooses to deliver, as opposed to a specific bond someone "
         "needs, so it is the cheapest secured borrowing that exists. SOFR is "
         "itself built from general collateral Treasury repo transactions, "
         "which makes financing a Treasury close to definitionally flat to the "
         "reference rate."),
        ("Agency mortgages", "15bp", "TBA dollar rolls and agency repo, which "
         "trade a few basis points wide of Treasury general collateral."),
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
        "barely differ across strategies.</strong> That is worth pausing on, "
        "because you would expect risk parity to finance far more cheaply than "
        "equal weight: it holds a third of the book in the two year Treasury, "
        "which borrows at 3bp. It does not, for two reasons. It tilts just as "
        "hard toward short investment grade, which costs 50bp. And it barely "
        "underweights the municipals at 110bp, because municipals are not the "
        "risky assets in this universe. They run at 4.0% and 5.5% volatility "
        "against the 30 year Treasury's 13.4%, so equalising risk contributions "
        "leaves them close to their equal weight.")
    if FROUTE is not None:
        t = FROUTE.drop(index=[i for i in DROP_ROWS if i in FROUTE.index],
                        errors="ignore").copy()
        t.columns = ["proportional, bp", "overlay, bp"][:t.shape[1]]
        t.index.name = "strategy"
        r.table(t.round(1), align_right=list(t.columns))
    r.prose(
        "Those blended rates assume <strong>proportional scaling</strong>: to "
        "run the book at 1.5 times, every position is bought at 1.5 times, "
        "including the municipal fund. That means taking a margin loan against "
        "an illiquid mutual fund, which no desk would actually do.")
    r.prose(
        "The realistic alternative is an <strong>overlay</strong>. Hold the "
        "cash book exactly as the strategy specifies, including the municipals "
        "at their unlevered weight, and obtain the <em>additional</em> exposure "
        "only through instruments that have a derivative: Treasury futures for "
        "the rates sleeve, total return swaps for credit. The municipal sleeve "
        "is never levered at all, so its 110bp never enters the marginal cost, "
        "and the borrowed portion is financed at the blend of the cheaper "
        "instruments instead. That works out around 28 basis points rather than "
        "40.")
    r.prose(
        "<strong>Every number in this project uses the proportional figure, the "
        "more expensive of the two</strong>, so the leverage drag reported here "
        "is an upper bound on what an implementation would pay.")

    r.section("What the assumption is worth")
    r.prose(
        "The unlevered comparison does not depend on it at all: hierarchical "
        "risk parity scores 0.659 against the Aggregate's 0.518 over the full "
        "sample holding no leverage, and every significance test runs on the "
        "unlevered series. The constant-risk comparison does depend on it, and "
        "it has a breakeven.")
    KEEP = ["Hierarchical RP", "Risk parity (ERC)", "HRP", "ERC"]
    if FBE is not None:
        keep = [c for c in ["leverage", "pays_bp", "breakeven_bp",
                            "headroom_bp"] if c in FBE.columns]
        t = FBE.reindex([i for i in KEEP if i in FBE.index])[keep].copy()
        t.columns = ["scaling", "pays, bp", "breakeven, bp",
                     "headroom, bp"][:len(keep)]
        t.index.name = "strategy"
        r.table(t.round(1), align_right=list(t.columns),
                caption="Against the Aggregate index.")
    if SENS is not None:
        t = SENS.reindex([i for i in KEEP + ["Agg index (VBMFX)"]
                          if i in SENS.index])
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

    r.section("References", (
        "Each of these is cited at the point it is used above."))
    r.prose(
        "Lopez de Prado, M. (2016). Building Diversified Portfolios that "
        "Outperform Out of Sample. <em>Journal of Portfolio Management</em>, "
        "42(4), 59-69. <span class=\"note\">Hierarchical risk parity, the "
        "clustering construction used throughout.</span>")
    r.prose(
        "Frazzini, A. and Pedersen, L. H. (2014). Betting Against Beta. "
        "<em>Journal of Financial Economics</em>, 111(1), 1-25. "
        "<span class=\"note\">The leverage-constraint argument in the leverage "
        "costs section.</span>")
    r.prose(
        "DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal Versus Naive "
        "Diversification: How Inefficient is the 1/N Portfolio Strategy? "
        "<em>Review of Financial Studies</em>, 22(5), 1915-1953. "
        "<span class=\"note\">Why the forecast tilts in Phase 2 are bounded "
        "rather than fed to an optimiser.</span>")
    r.prose(
        "Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. "
        "<em>Journal of the American Statistical Association</em>, 89(428), "
        "1303-1313. <span class=\"note\">The resampling scheme behind every "
        "p-value reported in this project.</span>")
    r.prose(
        "Campbell, J. Y. and Thompson, S. B. (2008). Predicting Excess Stock "
        "Returns Out of Sample: Can Anything Beat the Historical Average? "
        "<em>Review of Financial Studies</em>, 21(4), 1509-1531. "
        "<span class=\"note\">The benchmark for what counts as an "
        "economically meaningful out of sample R squared, cited in Phase 1."
        "</span>")
    r.prose(
        "Clark, T. E. and West, K. D. (2007). Approximately Normal Tests for "
        "Equal Predictive Accuracy in Nested Models. <em>Journal of "
        "Econometrics</em>, 138(1), 291-311. <span class=\"note\">The test "
        "behind the p-values on the forecast skill table in Phase 1.</span>")
    r.prose(
        "Cochrane, J. H. and Piazzesi, M. (2005). Bond Risk Premia. "
        "<em>American Economic Review</em>, 95(1), 138-160. "
        "<span class=\"note\">The forward rate factor, used as a signal and "
        "cited on the scale of bond return predictability.</span>")
    r.prose(
        "Ledoit, O. and Wolf, M. (2008). Robust Performance Hypothesis Testing "
        "with the Sharpe Ratio. <em>Journal of Empirical Finance</em>, 15(5), "
        "850-859. <span class=\"note\">The basis for the bootstrap in "
        "Appendix C, and the reference the 2025 study cited.</span>")
    r.prose(
        "Ledoit, O. and Wolf, M. (2004). Honey, I Shrunk the Sample Covariance "
        "Matrix. <em>Journal of Portfolio Management</em>, 30(4), 110-119. "
        "<span class=\"note\">The covariance estimator used by every "
        "risk-based strategy in this project.</span>")
    paginate(r, "appendix_method")
    return r.render(OUT / "appendix_method.html")


# ---------------------------------------------------------------- appendix C

def appendix_eval():
    r = PhaseReport(
        phase="Appendix C", title="Evaluating Results",
        summary="P-values, bootstrapping.",
        status="complete", project=PROJECT)

    r.section("How every p-value in this project is computed", (
        "Each table from here on carries a p-value against the benchmark. It is "
        "worth setting out how those are produced, because the textbook "
        "calculation does not apply to a Sharpe ratio and would overstate "
        "significance throughout."))
    r.prose(
        "<strong>The question being asked.</strong> A strategy beat the "
        "Aggregate by some margin. Could a strategy with no genuine edge have "
        "produced a gap that large, purely from the luck of which returns "
        "happened to land in this sample? The p-value is the fraction of the "
        "time the answer is yes.")
    r.prose(
        "<strong>Why not a t-test.</strong> The standard error of a Sharpe "
        "ratio assumes returns are independent and normally distributed. Bond "
        "returns are neither. They cluster: volatile stretches follow volatile "
        "stretches, and returns in adjacent periods are not independent of one "
        "another. Serial dependence means the sample contains fewer independent "
        "observations than it has rows, so a formula that counts rows produces "
        "an error bar that is too narrow and a p-value that is too small.")
    r.prose(
        "<strong>What is done instead.</strong> A stationary block bootstrap, "
        "following Politis and Romano. The procedure is mechanical:")
    r.table(pd.DataFrame([
        ("1", "Take excess returns", "Subtract the risk-free rate from both the "
         "strategy and the benchmark, on every date, and keep only dates where "
         "both exist."),
        ("2", "Resample in blocks", "Build a synthetic history the same length "
         "as the real one by drawing contiguous <em>blocks</em> of dates rather "
         "than individual days. Blocks preserve whatever autocorrelation and "
         "volatility clustering the data has; drawing single days would destroy "
         "it and hand back the too-narrow error bar."),
        ("3", "Randomise the block length", "Block lengths are drawn from a "
         "geometric distribution rather than fixed. That is the "
         "<em>stationary</em> part, and it prevents the result depending on an "
         "arbitrary choice of block size."),
        ("4", "Keep the pair together", "The same block of dates is drawn for "
         "the strategy and the benchmark. Both live through the same simulated "
         "market, so the common move cancels and what remains is the "
         "difference between them, which is the quantity in question."),
        ("5", "Repeat 5,000 times", "Each resample gives one Sharpe difference. "
         "Five thousand of them trace out the distribution of gaps this data "
         "could produce."),
        ("6", "Centre and count", "Shift that distribution to a mean of zero, "
         "which imposes the null of no true edge, then count how often a "
         "resample still reaches the observed gap. That fraction is the "
         "one-sided p-value."),
    ], columns=["", "step", "what it does"]).set_index(""))
    r.prose(
        "So <strong>p = 0.03 means that in 3% of five thousand simulated "
        "histories, a strategy with no real edge still beat the benchmark by as "
        "much as this one did.</strong> The 95% confidence intervals reported "
        "later come from the same 5,000 draws, read at the 2.5th and 97.5th "
        "percentiles.")
    r.prose(
        "Blocks average twelve periods. The seed is fixed, so the numbers "
        "reproduce exactly. Stars in every table mark *** below 0.01, ** below "
        "0.05 and * below 0.10.")
    r.prose(
        "One limit worth stating: this handles serial dependence, but it does "
        "not correct for having tried many strategies. Twenty candidates "
        "against a benchmark will produce a low p-value somewhere by chance, "
        "which is the reason the holdout exists and the reason candidates are "
        "chosen on development before it is opened.")

    r.section("How this extends the 2025 work", (
        "The original study bootstrapped a Sharpe difference too. Three things "
        "are done differently here, and each is worth a small amount."))
    r.prose(
        "The 2025 version drew 100,000 resamples of <em>individual</em> months, "
        "paired across both series, and reported the fraction of resampled "
        "differences at or below zero. It cited Ledoit and Wolf (2008), which "
        "is the right reference, though that paper argues for a block bootstrap "
        "rather than the independent draws it actually used.")
    r.table(pd.DataFrame([
        ("Blocks instead of single periods", "Individual draws destroy the "
         "autocorrelation and volatility clustering in the data, which makes "
         "the resampled distribution too narrow and the p-value too small."),
        ("Random block length", "Fixed blocks make the answer depend on the "
         "block size chosen. Drawing lengths from a geometric distribution "
         "removes that, which is what makes the bootstrap stationary."),
        ("Centring on the null", "The 2025 p-value was the raw fraction of "
         "resamples below zero, which is a confidence statement about the "
         "observed effect rather than a test against a null of no edge. "
         "Shifting the distribution to a mean of zero first imposes the null."),
    ], columns=["change", "why it matters"]).set_index("change"))
    r.prose(
        "Run side by side on the same data, the three changes move p-values by "
        "at most about 0.03, always in the conservative direction on the "
        "development sample. The 2025 conclusion was not wrong; it was measured "
        "with a method that would understate p on more autocorrelated data, "
        "which is exactly the situation in this universe.")
    paginate(r, "appendix_eval")
    return r.render(OUT / "appendix_eval.html")


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
        "Extension from lessons learned in my macro portfolio project. Asset "
        "universe and data collection.",
        "Five model classes, how each forms its weights, and how they score on "
        "the development sample.",
        "The held-out decade, the constant-risk comparison, the duration check, "
        "and portfolio characteristics.",
        "How much independent variation the universe contains, and how "
        "forecastability is distributed across it.",
        "Per-asset financing, the shared estimation window, and the "
        "constant-risk convention.",
        "How every p-value in this project is produced, and how that extends "
        "the validation in the 2025 study.",
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
<span class="eyebrow">Gus Guenther</span>
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
construction.</p>

<p><b>What worked.</b> Risk parity, which never needs an expected return at all.
I built and tested both classic risk parity and Marcos Lopez de Prado's
Hierarchical Risk Parity, benchmarked against the Bloomberg Aggregate with equal
weight as a second reference.</p>

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
    for fn in [phase1, phase2, phase3, appendix_method, appendix_universe,
               appendix_eval, index]:
        try:
            p = fn()
            print(f"  wrote {Path(p).name}")
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
