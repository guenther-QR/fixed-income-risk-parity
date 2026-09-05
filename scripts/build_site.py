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

# One line per page, shown on the summary card and as the page's own
# standfirst. Keyed by stem so the two cannot fall out of step.
PAGE_DESC = {
    "phase1_idea":
        "Extension from lessons learned in my macro portfolio project. Asset "
        "universe and data collection.",
    "phase2_strategies":
        "Regression, machine learning, technical, and risk-based strategy "
        "development.",
    "phase3_results":
        "Holdout decade results, constant risk comparisons and portfolio "
        "characteristics. Conclusions and next steps.",
    "appendix_method":
        "Rebalancing, leverage and transaction costs, and other portfolio "
        "construction mechanics.",
    "appendix_universe":
        "Independent variation, principal component analysis, and "
        "forecastability.",
    "appendix_eval":
        "P-value construction.",
}

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


STRATEGY_NAME = {
    "HRP + Rolling 60m overlay": "HRP + 60m rolling window overlay",
    "Rolling 60m, long only": "60m rolling window, long only",
    "Rolling 60m (1m target), long only":
        "60m rolling window (1m target), long only",
}


def result_table(r, df, caption, status=False):
    """Render a results frame the same way on every page.

    Beta is dropped, the benchmark row is called out, cells that clear five
    percent are tinted, and the alpha t statistic carries its own stars so a
    reader does not have to cross-reference two columns to see whether the
    excess return is real.
    """
    t = df.copy()
    t.index = [STRATEGY_NAME.get(i, i) for i in t.index]
    t = t.drop(columns=[c for c in ["beta"] if c in t.columns])
    for c in ["return", "vol", "turnover"]:
        if c in t.columns:
            t[c] = t[c] * 100
    for c in ["duration", "turnover"]:
        if c in t.columns:
            t[c] = t[c].round(1)
    if "t_alpha" in t.columns:
        n = (~t.index.str.contains("Agg index")).sum()
        t["t_alpha"] = [
            "" if abs(v) < 1e-9 else
            f"{v:.2f}" + ("***" if abs(v) > 2.58 else "**" if abs(v) > 1.96
                          else "*" if abs(v) > 1.645 else "")
            for v in t["t_alpha"]]
    t = t.rename(columns={
        "return": "return, %", "vol": "volatility, %", "sharpe": "Sharpe",
        "vs_agg": "vs the Agg", "p": "p", "alpha_pct_yr": "alpha, % a year",
        "t_alpha": "t", "duration": "duration", "turnover": "turnover, %"})
    t.index.name = "strategy"
    r.table(t.round(3), compact=True,
            align_right=[c for c in t.columns if c != "status"],
            stars=["p"], heat=["Sharpe"],
            row_class={i: ("bench-row" if "Agg index" in i else "")
                       for i in t.index},
            caption=caption)
    return r


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
        summary=PAGE_DESC["phase1_idea"],
        status="complete", project=PROJECT)

    stats = get("fi_daily_stats")
    mskill = macro("predict_skill")

    r.section("Macro Portfolio Project Summary", (
        "Last summer I worked on a portfolio construction project that used a "
        "rudimentary optimization framework and returns analysis to generate "
        "portfolios that tried to beat the 60/40. The Macro Portfolio Project "
        "Rebuild I recently completed added additional analysis including "
        "proper backtesting, a more sophisticated fixed income framework, and "
        "out of sample testing to evaluate portfolios built using regression "
        "and machine learning techniques."))
    r.prose(
        "Roughly 1,600 specifications were tested there: regime allocation "
        "across 1,296 design combinations, return regression on 182 signals, "
        "eight machine learning families, recession timing and cross sectional "
        "ranking, at monthly and daily frequency, on universes from 7 to 59 "
        "assets. None beat a 60/40 benchmark on both the development sample and "
        "the held-out decade.")
    r.prose(
        "<strong>One thing did stand out: fixed income looked more predictable "
        "than equity or commodities markets.</strong>")

    r.section("Predictive skill by instrument", (
        ""))
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
        "its consequences are flagged wherever a result leans on those "
        "holdings.")
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
        "target, and the resulting return is "
        "decomposed into <strong>carry</strong>, <strong>rolldown</strong>, "
        "<strong>duration</strong> and <strong>convexity</strong>. The "
        "walk-forward backtesting engine, the per-asset transaction costs and "
        "the sealed holdout methodology are also used here, with a new eye on "
        "per-asset leverage costs within implementation.")
    r.prose(
        "<strong>New here.</strong> Three things:")
    r.table(pd.DataFrame([
        ("A fixed income universe", "Eleven assets rather than the parent "
         "project's seven, and all of them bonds: four Treasury maturities, "
         "four corporate credit sleeves, agency mortgages and two municipal "
         "holdings."),
        ("Bond-specific signals and factors", "Signals that are distinctly "
         "useful for forecasting fixed income and have no counterpart in "
         "equities or commodities: rolldown by maturity, modified "
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
        "Development results",
    ])
    paginate(r, "phase1_idea")
    return r.render(OUT / "phase1_idea.html")


# ------------------------------------------------------------------ phase 2

def phase2():
    r = PhaseReport(
        phase="Phase 2", title="Strategies",
        summary=PAGE_DESC["phase2_strategies"],
        status="complete", project=PROJECT)

    DEV = get("fi_table_development")
    MLS = get("fi_dmodel_ml_skill")
    MLM = get("fi_ml_monthly_tuned")
    UNI = get("fi_dmodel_regression_skill")
    UNI_M = get("fi_uni_regression_skill")
    RECUR = get("fi_regressor_recurring")

    # ---------------------------------------------------------- regression
    r.section("Regression Methodology")
    r.prose(
        "For each asset and each signal independently, fit an expanding window "
        "univariate ordinary least squares regression of next month\'s excess "
        "return on the signal:")
    r.formula(
        "<span class=\'t t1\'>y<sub>t</sub></span> &nbsp;=&nbsp; "
        "&alpha; &nbsp;+&nbsp; &beta; &middot; "
        "<span class=\'t t3\'>x<sub>t-1</sub></span> &nbsp;+&nbsp; "
        "&epsilon;<sub>t</sub>",
        "one regression per signal, per asset")
    r.prose(
        "Then average the forecasts across signals into one number per asset "
        "per month. This is the Goyal-Welch combination, meant to average out "
        "statistical noise and provide cleaner forecasting.")

    r.section("Regression Development Results")
    if UNI is not None:
        t = (UNI * 100)[["dev_r2"]].copy()
        t.columns = ["daily"]
        if UNI_M is not None:
            t["monthly"] = (UNI_M * 100)["dev_r2"].reindex(t.index)
        t.index = [ASSET_NAME.get(a, a) for a in t.index]
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
        "The exception is the three holdings with the poorest day-to-day data "
        "quality. High yield and the two municipal funds are the only assets "
        "that score higher daily than monthly, and they are also the only "
        "assets priced by matrix valuation rather than by observed trades. "
        "Their reported daily returns autocorrelate at 0.24 to 0.25, against "
        "0.01 to 0.03 for the Treasuries.")
    r.prose(
        "<strong>Strategies that predict returns daily, especially ones which "
        "rely upon returns generated from these three assets, deserve extra "
        "scrutiny going forward.</strong>")

    r.section("Which signals do the work")
    r.prose(
        "With 256 signals, we look at which appear in the top ten best "
        "regressors for each individual asset to triangulate which signals are "
        "most robust.")
    if RECUR is not None:
        t = RECUR.copy()
        t = t[~t.index.str.startswith("d_rev5_")]
        t.index = [signal_name(x) for x in t.index]
        t["best_on"] = [ASSET_NAME.get(a, a) for a in t["best_on"]]
        t["median_r2"] = t["median_r2"] * 100
        t["best_r2"] = t["best_r2"] * 100
        t.columns = ["family", "assets in top ten", "median R squared, %",
                     "best R squared, %", "best on"]
        t.index.name = "signal"
        r.table(t.round(3), compact=True,
                align_right=["assets in top ten", "median R squared, %",
                             "best R squared, %"],
                heat=["median R squared, %"],
                caption="Signals appearing in the ten best for three or more "
                        "assets, daily data, development sample. Momentum and "
                        "reversal over the same horizon are one quantity with "
                        "opposite sign, so only one of each pair is shown.")

    # --------------------------------------------------- machine learning
    r.section("Machine Learning Methodology", (
        "Four families from scikit-learn on the same signal panel: elastic "
        "net, ridge, random forest and gradient boosting. Each is run across a "
        "grid of hyperparameters and the configuration with the best "
        "development R squared is selected."))
    if MLS is not None:
        t = (MLS * 100)["dev_r2"].unstack()
        t.index = [i.replace("_", " ").capitalize() for i in t.index]
        t.index.name = "family"
        stale = [c for c in ["hy", "muni", "muni_hy"] if c in t.columns]
        clean = [c for c in t.columns if c not in stale]
        t = t[clean + stale]
        t.columns = [ASSET_NAME.get(c2, c2) for c2 in t.columns]
        r.table(t.round(2), compact=True, align_right=list(t.columns),
                heat=list(t.columns),
                caption="Development sample, daily data, R squared in percent. "
                        "Each forecast is fitted only on data preceding the day "
                        "it predicts. The three right-hand columns are the "
                        "funds priced by matrix valuation rather than by "
                        "trades.")
    r.prose(
        "Looking at assets with problematic daily data, the municipals and high "
        "yield, whose difficulty arises from the illiquidity and marking of "
        "those assets as noted above, illuminates why the machine learning "
        "models appear to be decent predictors but likely cannot be trusted for "
        "actual medium and long term predictability.")
    r.prose(
        "Averaged across all eleven assets the families look mildly positive, "
        "at +2.0% for the elastic net and +1.9% for the random forest. "
        "Splitting the universe reverses that: on the eight holdings that trade "
        "continuously every family is negative, and on high yield and the two "
        "municipal funds every family is strongly positive.")
    r.table(pd.DataFrame([
        ("Elastic net", "alpha 0.1, L1 ratio 0.5", "+2.03", "+7.67", "-0.09"),
        ("Random forest", "150 trees, max depth 3", "+1.90", "+7.83", "-0.33"),
        ("Ridge", "alpha 1000", "+1.60", "+8.39", "-0.95"),
        ("Gradient boosting", "100 trees, depth 1, learning rate 0.01",
         "+1.13", "+4.46", "-0.12"),
    ], columns=["family", "specification", "all 11 assets",
                "high yield and municipals", "the other 8 assets"]
    ).set_index("family"), compact=True,
        align_right=["all 11 assets", "high yield and municipals",
                     "the other 8 assets"],
        caption="Development sample, daily data, R squared in percent. Each "
                "specification is the configuration with the best development "
                "score from its grid search.")
    r.prose(
        "With the unreliability of forecasting on a daily basis, we also apply "
        "machine learning to monthly forecasts and see that they deteriorate "
        "even after tuning.")
    if MLM is not None:
        t = (MLM[["all_11"]] * 100).copy()
        t.index = [i.replace("_", " ").capitalize() for i in t.index]
        t.index.name = "family"
        t.columns = ["monthly R squared, %"]
        r.table(t.round(2), align_right=list(t.columns),
                caption="Development sample, one month forward target, "
                        "averaged across all eleven assets. Each family is "
                        "tuned on its own grid for this horizon rather than "
                        "inheriting the daily settings. As such, we do not "
                        "have high hopes for machine learning models on this "
                        "project.")


    # -------------------------------------------------------- technical
    r.section("Technical Model Methodology")
    r.prose(
        "Another set of models was tested, those that rely purely on technical "
        "factors such as momentum: the change in price or returns over some "
        "past interval. These factor models are popular in the literature and "
        "we explore their use case in the fixed income universe here. We "
        "introduce three types of momentum models for our purposes.")
    r.prose(
        "<strong>Momentum 12-1</strong> ranks each holding by its return over "
        "the past twelve months, skipping the most recent one. "
        "<strong>Momentum (Sharpe)</strong> divides that trailing return by its "
        "own volatility, so an asset is ranked on how consistently it rose "
        "rather than how far. <strong>Rolling 60-month selection</strong> "
        "declines to fix a window at all: every month, each asset scores "
        "sixteen candidate windows on the trailing five years and trades "
        "whichever has predicted it best. This is essentially a regression "
        "based momentum strategy that is adaptive over time, and our own idea "
        "for the project.")
    r.prose(
        "A carry signal was intended as a fourth member of this family and is "
        "not reported. Carry on a bond is its yield less the financing rate, "
        "and seven of the eleven holdings are funds quoted as prices with no "
        "yield attached, so the signal can only be formed for the Treasury "
        "sleeve.")

    # ------------------------------------------------------- risk parity
    r.section("Risk Parity Methodology")
    r.prose(
        "The risk parity style of models, instead of focusing on predicting "
        "returns, looks at creating portfolios based on the realized covariance "
        "matrix among assets. This is easier to construct, and there is good "
        "evidence that our estimate of the covariance matrix actually improves "
        "with more data. We explore two types of risk parity models in this "
        "project.")
    r.prose(
        "<strong>Equal risk contribution.</strong> Instead of putting equal "
        "money in each asset, put equal <em>risk</em> in each asset. A 30 year "
        "Treasury runs at roughly eight times the volatility of a two year, so "
        "it receives roughly one eighth the weight. Formally, solve for the "
        "weights at which every asset\'s contribution to portfolio variance is "
        "the same:")
    r.formula(
        "<span class=\'t t1\'>w<sub>i</sub></span> &middot; "
        "<span class=\'t t3\'>(&Sigma;w)<sub>i</sub></span> &nbsp;/&nbsp; "
        "&sigma;<sub>p</sub> &nbsp;=&nbsp; constant, for every i",
        "equal risk contribution")
    r.prose(
        "There is no closed form, so it is solved by fixed-point iteration: "
        "start at equal weight, compute each asset\'s risk contribution, shift "
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
        "presents. Both methods take their covariance from a Ledoit-Wolf "
        "shrinkage estimator with a constant correlation target, estimated on "
        "an expanding window from the first day of data, and neither ever "
        "estimates an expected return, which is why neither has a forecast to "
        "decay.")

    # ------------------------------------------------ portfolio construction
    r.section("Portfolio Construction", (
        "A model produces a number per asset. Turning that number into weights "
        "is a separate step, and it is not the same step for every family."))
    r.prose(
        "<strong>Regression, machine learning and technical models.</strong> "
        "These produce a forecast or a score, not a portfolio. The score is "
        "standardised across assets on each date, so what drives the weights is "
        "which holdings look good against the others that day rather than the "
        "level of the forecast:")
    r.formula(
        "<span class=\'t t1\'>z<sub>i</sub></span> &nbsp;=&nbsp; "
        "( <span class=\'t t3\'>x<sub>i</sub></span> &minus; "
        "<span class=\'t t2\'>&mu;</span> ) &nbsp;/&nbsp; "
        "<span class=\'t t2\'>&sigma;</span>",
        "cross-sectional standardisation, across the eleven assets each day")
    r.prose(
        "Weights are then the positive part of that score, normalised to sum to "
        "one. An asset scoring below the daily average is not held; the rest "
        "are held in proportion to how far above it they sit. The book is long "
        "only and fully invested. The same rule is applied to all three "
        "families, so a comparison between them is a comparison of the signals "
        "rather than of their packaging.")
    r.prose(
        "<strong>Risk parity.</strong> These determine weights directly. "
        "Equal risk contribution solves for them by iteration and hierarchical "
        "risk parity produces them by recursive bisection down the cluster "
        "tree. No standardisation step is involved because there is no forecast "
        "to standardise.")
    r.prose(
        "<strong>Overlays.</strong> An overlay starts from the risk parity "
        "weights and moves each one by lambda times its standardised score "
        "times that same weight, floors anything negative at zero, then "
        "renormalises. The deviation is proportional to the base weight, so a "
        "large position moves further in absolute terms than a small one. It is "
        "not a fifty-fifty blend of two portfolios.")
    r.prose(
        "Rebalancing is annual for risk parity, the overlays and the rolling "
        "selection, and monthly for the two fixed-window momentum signals. "
        "Every figure on this page is net of per-asset transaction costs, which "
        "run from two basis points one way on the two year Treasury to "
        "forty-five on high yield municipals.")

    # ------------------------------------------------------------- results
    r.section("Development Results")
    if DEV is not None:
        result_table(
            r, DEV,
            "Development sample, monthly returns, net of per-asset costs. "
            "Shaded Sharpe cells beat the Aggregate at five percent or better. "
            "Stars on p and on the alpha t statistic: *** below 0.01, "
            "** below 0.05, * below 0.10.")
    r.prose(
        "<strong>Hierarchical risk parity and the two overlays built on it "
        "clear the index; nothing else does.</strong> Risk parity scores 0.952 "
        "against the Aggregate at 0.824, at p = 0.037. The rolling selection "
        "overlay reaches 1.056 at p = 0.004 and the momentum Sharpe overlay "
        "0.965 at p = 0.022. Equal risk contribution does not clear the bar at "
        "this frequency, at 0.886 and p = 0.190.")
    r.prose(
        "<strong>The overlays are harder to read than their p values "
        "suggest.</strong> Risk parity carrying the momentum Sharpe signal sits "
        "0.013 above risk parity alone. The two return series correlate at "
        "0.979 daily and the tracking error between them is about half a "
        "percent a year. Dividing a momentum signal by its own volatility ranks "
        "the low volatility assets highest, and risk parity already overweights "
        "those same assets, so the overlay pushes the portfolio toward where it "
        "already was. The 60 month rolling window overlay moves further and adds 0.10 "
        "of Sharpe, which is <strong>a real gap on development</strong>.")
    r.prose(
        "There is some evidence that momentum is worth something here when it "
        "is risk adjusted or when the window is chosen adaptively. The rolling "
        "60 month selection returns 7.58% a year against the index at 6.40%. "
        "It does not clear five percent on its own, at p = 0.301, so it is "
        "carried into the holdout as an open question rather than a finding.")
    r.prose(
        "<strong>The forecasting families do not work.</strong> All four "
        "machine learning models land below the index, from 0.688 to 0.754 "
        "against 0.824, none of them distinguishable from it. The combined "
        "regression is worse than the index and significantly so, at 0.515 and "
        "p = 0.0006. Momentum on the raw twelve month return, the standard "
        "definition from the factor literature, is the worst strategy on the "
        "page: 0.386 against 0.824 at p < 0.001.")
    r.prose(
        "<strong>Every model that estimates a return ends up holding "
        "duration.</strong> The Aggregate runs an effective duration of 4.2 "
        "years. The four machine learning books run 7.8 to 8.9 years, the "
        "combined regression 12.5, and raw momentum 8.4. A signal weighted "
        "portfolio concentrates into whatever has been rising, and in a "
        "universe of bonds that is the long end, so each of them arrives at a "
        "levered position on the level of rates without ever having been asked "
        "for one. Their betas to the index run from 1.21 to 2.03 while the "
        "risk-based methods sit between 0.59 and 0.87.")

    r.section("What we carry into the holdout")
    r.prose(
        "Three strategies cleared the Aggregate at five percent on "
        "development: hierarchical risk parity, and risk parity carrying each "
        "of the two overlays. Those go forward.")
    r.prose(
        "<strong>Equal risk contribution is carried as a comparison rather "
        "than as a candidate.</strong> It does not clear the bar on "
        "development, at p = 0.190. It is carried because the whole point of "
        "testing hierarchical risk parity is to know whether the clustering "
        "step earns anything over textbook risk parity, and that question only "
        "has an answer if both are measured at every stage. Dropping it after "
        "development would leave the comparison half finished.")
    r.prose(
        "We also carry the rolling 60 month selection and momentum on the "
        "Sharpe ratio in their standalone form, without the risk parity base "
        "underneath them. Neither cleared five percent. They are carried "
        "because we want to know whether these signals work on their own or "
        "only as a small adjustment to a portfolio that already works, and that "
        "question cannot be answered by leaving them out.")
    r.prose(
        "That makes six into the holdout. Three confirmatory, meaning they "
        "cleared the index at five percent on development: hierarchical risk "
        "parity, and risk parity carrying each of the two overlays. Three "
        "carried for other reasons: equal risk contribution as the comparison "
        "case, and the 60 month rolling window momentum strategy and momentum "
        "on the Sharpe ratio as open questions about whether either signal "
        "stands on its own.")
    r.prose(
        "Nothing from the regression or machine learning side is carried. Every "
        "one of them finishes below the index on development, and the R squared "
        "figures that made them look promising rest almost entirely on three "
        "holdings whose prices move late.")

    r.next_up("Phase 3 - Results", [
        "Validating strategies on the holdout data",
        "Trading and leverage costs",
        "Portfolio characteristics and conclusion",
    ])
    paginate(r, "phase2_strategies")
    return r.render(OUT / "phase2_strategies.html")


# ------------------------------------------------------------------ phase 3

def phase3():
    FULL = get("fi_table_fullsample")
    OOS_T = get("fi_table_holdout")
    r = PhaseReport(
        phase="Phase 3", title="Results",
        summary=PAGE_DESC["phase3_results"],
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

    r.section("The holdout", (
        "The six strategies carried out of Phase 2, on ten years of data none "
        "of them was fitted to."))
    if OOS_T is not None:
        result_table(
            r, OOS_T,
            "Holdout, January 2016 to August 2026, monthly returns, net of "
            "per-asset costs. Confirmatory strategies cleared the index at five "
            "percent on development; the rest were carried for comparison or as "
            "open questions.")
    r.prose(
        "<strong>Both risk parity methods clear the index, and neither overlay "
        "does.</strong> Equal risk contribution beats the Aggregate by 0.161 of "
        "Sharpe at p = 0.008, and hierarchical risk parity by 0.135 at "
        "p = 0.039. Their alphas are 0.63% and 0.41% a year, with t statistics "
        "of 2.64 and 1.83.")
    r.prose(
        "The index itself returned a Sharpe of <strong>-0.078</strong> over "
        "this decade, so beating it is not a demanding test on its own.")
    r.prose(
        "<strong>Both overlays failed.</strong> On development the rolling "
        "selection overlay added 0.10 of Sharpe over plain risk parity and "
        "cleared the index at p = 0.004. On the holdout it finishes 0.06 "
        "<em>below</em> plain risk parity and does not clear the index at all, "
        "at p = 0.240. The momentum Sharpe overlay does the same thing, ending "
        "0.09 below the portfolio it was supposed to improve. Whatever the "
        "overlays were adding on development, they do not add it here.")
    r.prose(
        "<strong>The two signals on their own do not work either.</strong> The "
        "rolling selection lands at -0.017 with an alpha of 0.26% and a t "
        "statistic of 0.26, which is nothing. Momentum on the Sharpe ratio "
        "returns -0.03% a year against the index at 1.74% and finishes 0.34 of "
        "Sharpe below it. It was one of the better technical strategies on "
        "development.")
    r.section("Comparing at constant risk", (
        "Risk parity beats the index on Sharpe while running less risk. Scaling "
        "it up to the index\'s own volatility tests whether the advantage "
        "survives once the risk is matched."))
    r.prose(
        "The Aggregate runs at 4.17% volatility and hierarchical risk parity at "
        "2.82%, so risk parity is levered 1.48 times to match, and financing is "
        "charged on the borrowed portion at the per-asset rates in Appendix A. "
        "Anything already running hotter than the index is scaled down, with "
        "the balance in cash.")
    if L is not None:
        keep = [c for c in ["full_leverage", "full_lev_cagr", "full_lev_vol",
                            "full_lev_sharpe", "full_lev_dd", "full_lev_vs_agg"]
                if c in L.columns]
        L = L.drop(index=[i for i in ["1/N"] if i in L.index])
        t = L[keep].copy()
        t.columns = ["scaling", "return", "volatility", "Sharpe",
                     "worst drawdown", "vs the Agg"][:len(keep)]
        t.index.name = "strategy"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Full sample, every series at the Aggregate's own 4.17% "
                        "volatility, financing charged.")
    if C is not None:
        # The Aggregate is the only benchmark; equal weight is not shown.
        C = C.drop(columns=[c for c in list(DROP_ROWS) + ["1/N"]
                            if c in C.columns])
        roles = {c: ("benchmark" if c == "Agg index (VBMFX)"
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

    r.section("2021-2023 bond bear market performance", (
        "The decade the strategies were tested on contained the worst bond "
        "market in forty years, which constrains what any result measured over "
        "it can establish."))
    r.table(pd.DataFrame([
        ("2021", "-1.75%", "-0.29%", "-0.32%"),
        ("2022", "-13.24%", "-7.59%", "-9.95%"),
        ("2023", "+5.62%", "+5.63%", "+6.06%"),
        ("2021-2023 cumulative", "-9.96%", "-2.67%", "-4.80%"),
        ("Worst drawdown", "-17.5%", "-10.3%", "-13.0%"),
    ], columns=["", "Agg index", "Hierarchical RP", "Risk parity"]).set_index(""),
        align_right=["Agg index", "Hierarchical RP", "Risk parity"])
    r.prose(
        "Both risk based methods lost much less through the rate shock: 7.6% in "
        "2022 against the index at 13.2%. They hold less duration than the "
        "index, which is the obvious explanation and the one worth testing.")

    r.section("Portfolio characteristics", (
        "What the portfolio actually holds, month by month."))
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

    if FULL is not None:
        r.section("Full Sample Results", (
            "What each strategy would have produced over the full thirty-nine "
            "years, with every setting fixed where development left it and "
            "nothing refitted."))
        result_table(
            r, FULL,
            "1987 to 2026, monthly returns, net of per-asset costs. The "
            "settings were chosen on the development sample and carried "
            "forward unchanged, so this is one continuous record rather than "
            "two studies joined together.")
        r.prose(
            "Twenty-eight of the thirty-nine years are the development sample, "
            "so these figures are dominated by the same data the strategies "
            "were selected on. They describe what holding these portfolios "
            "would have returned. The holdout is what tests whether the edge "
            "was real.")

    r.section("Conclusions")
    r.prose(
        "<strong>Risk parity beat the Aggregate in every window we tested.</strong> "
        "Hierarchical risk parity clears the index on development, on the "
        "holdout and on the full sample. Equal risk contribution clears it on "
        "the holdout and the full sample but not on development. Neither result "
        "depends on a return forecast at any point in the construction.")
    r.prose(
        "<strong>The margin is not paid for with risk.</strong> Both methods "
        "run below the index on volatility and below it on beta, between 0.60 "
        "and 0.83, and both hold less duration than the index does.")
    r.prose(
        "<strong>It survives costs and institutional financing.</strong> "
        "Turnover is 1.5% to 2.1% a year, and there is 90 to 252 basis points "
        "of headroom to the financing breakeven.")
    r.prose(
        "<strong>Return forecasting adds nothing on this universe.</strong> "
        "The combined regression and all four machine learning families finish "
        "below the index on development. The R squared figures that made them "
        "look promising rest on three holdings whose prices move late, and the "
        "gap disappears when the forecast horizon moves from a day to a month.")
    r.prose(
        "<strong>The technical overlays are not worth running.</strong> Both "
        "cleared the index on development and both finish below plain risk "
        "parity on the holdout. Momentum on its own is worse still: on the raw "
        "twelve month definition it is the worst strategy tested, and the "
        "risk-adjusted version loses to the index out of sample.")
    r.prose(
        "<strong>The 60 month rolling window momentum strategy is the one open "
        "question.</strong> It returns more than any other strategy on "
        "development and it does not clear five percent in any window. It is "
        "our own construction and it deserves a proper test on data this "
        "project has not touched.")

    r.section("Next steps")
    r.table(pd.DataFrame([
        ("Replace funds with indices or futures",
         "The universe is mutual funds, which charge 20 to 80 basis points and "
         "carry manager decisions. High yield\'s daily returns autocorrelate "
         "at 0.24 from stale pricing. Index or futures data removes both."),
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

    paginate(r, "phase3_results")
    return r.render(OUT / "phase3_results.html")


# ---------------------------------------------------------------- appendix A

def appendix_universe():
    r = PhaseReport(
        phase="Appendix B", title="Market Dynamics",
        summary=PAGE_DESC["appendix_universe"],
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
    REB = get("fi_rebal_summary")
    OVL = get("fi_overlay_summary")
    r = PhaseReport(
        phase="Appendix A", title="Implementation",
        summary=PAGE_DESC["appendix_method"],
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

    r.section("The Aggregate's effective duration", (
        "Quoted as 4.2 years throughout, and estimated rather than taken from "
        "a vendor."))
    r.prose(
        "The index is held as a fund, so its duration is not published in the "
        "return series. It is estimated by regressing the index's daily return "
        "on the daily change in the ten year Treasury yield and taking the "
        "negative of the slope, which is the definition of effective duration:")
    r.formula(
        "<span class='t t1'>r<sub>t</sub></span> &nbsp;=&nbsp; &alpha; "
        "&nbsp;&minus;&nbsp; <span class='t t2'>D</span> &middot; "
        "<span class='t t3'>&Delta;y<sub>t</sub></span> &nbsp;+&nbsp; "
        "&epsilon;<sub>t</sub>",
        "effective duration from the yield sensitivity")
    r.prose(
        "Over the full sample that gives <strong>4.16 years</strong>. The "
        "figure is used only for comparison against the strategies' own "
        "durations, which are computed directly from their weights and the "
        "modified duration of each holding, so nothing in the results depends "
        "on it.")

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
    Bt = trim(get("fi_aligned_bootstrap"))
    r = PhaseReport(
        phase="Appendix C", title="Evaluating Results",
        summary=PAGE_DESC["appendix_eval"],
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

    paginate(r, "appendix_eval")
    return r.render(OUT / "appendix_eval.html")


# ------------------------------------------------------------------- index

def index():
    # The same three tables the phase pages render, so the summary cannot drift
    # away from the results it is summarising.
    TB = {tag: get(f"fi_table_{name}") for tag, name in
          [("dev", "development"), ("oos", "holdout"), ("full", "fullsample")]}

    body = ""
    if all(v is not None for v in TB.values()):
        order = TB["full"].sort_values("sharpe", ascending=False).index
        for name in order:
            cells = []
            for tag in ["dev", "oos", "full"]:
                t = TB[tag]
                if name not in t.index:
                    cells.append('&mdash;<span class="d">not carried</span>')
                    continue
                sh = t.loc[name, "sharpe"]
                if name == "Agg index":
                    sub_ = "benchmark"
                else:
                    e, pv = t.loc[name, "vs_agg"], t.loc[name, "p"]
                    mk = ("***" if pv < .01 else "**" if pv < .05
                          else "*" if pv < .10 else "")
                    sub_ = f"{e:+.3f}{mk} &nbsp;p={pv:.3f}"
                sig = (name != "Agg index"
                       and pd.notna(t.loc[name, "p"])
                       and t.loc[name, "p"] < 0.05)
                td = ' class="sig-cell"' if sig else ""
                cells.append(f'<span{td}>{sh:.3f}'
                             f'<span class="d">{sub_}</span></span>')
            cls = ' class="me"' if name.startswith(("HRP", "ERC")) else ""
            body += (f"<tr{cls}><th>{name}</th>"
                     + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    cards = "".join(
        f'<a class="card" href="{stem}.html"><span class="num">{label}</span>'
        f'<span class="ttl">{label}: {title}</span>'
        f'<span class="dsc">{PAGE_DESC[stem]}</span></a>'
        for stem, label, title in PAGES)

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

<p><b>What was tested.</b> Univariate regression forecasts combined across 256
signals, four machine learning families tuned on the development sample, three
technical momentum models including an adaptive rolling window of my own, and
two risk-based methods built from the covariance matrix alone.</p>

<p><b>What worked.</b> Risk parity strategies, which work by utilizing
risk-based position weighting instead of return forecasting. I built and tested
both classic risk parity and Marcos Lopez de Prado's Hierarchical Risk Parity,
benchmarked against the Bloomberg Aggregate.</p>

</div>

<h2>Sharpe ratio, and edge against the Aggregate index</h2>
<div class="scroll">
<table>
<thead><tr><th>Strategy</th><th>Development<br>1987-2015</th><th>Holdout<br>2016-2026</th><th>Full sample<br>1987-2026</th></tr></thead>
<tbody>{body}</tbody>
</table>
</div>
<p class="note">Shaded cells beat the Aggregate at five percent or better.
Stars mark one-sided bootstrap significance: *** below 0.01, ** below 0.05,
* below 0.10.</p>

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
