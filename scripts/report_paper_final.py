"""The final paper. Two parts, plain language."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro.report import charts  # noqa: E402
from macro.report.builder import PhaseReport  # noqa: E402

P = ROOT / "data/processed"


def maybe(p):
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def main() -> int:
    T = maybe(P / "fi_paper_table.parquet")
    TS = maybe(P / "fi_paper_turnover_sharpe.parquet")
    TO = maybe(P / "fi_paper_turnover.parquet")
    DT = maybe(P / "fi_paper_duration.parquet")
    fipred = maybe(P / "fi_predictability_risk.parquet")
    nets = maybe(P / "fi_bench_final.parquet")

    r = PhaseReport(
        phase="Portfolio Research",
        title="Bonds Are Easier to Forecast Than Stocks",
        summary=("Part 1 rebuilds a macro allocation study I wrote in 2025, "
                 "fixes nine errors, and tests it properly. Nothing beat 60/40 "
                 "out of sample, and the reason is measurable. Part 2 takes the "
                 "one result that held up and builds a fixed income portfolio "
                 "around it."),
        status="complete")

    r.metrics([
        ("1.17 to 0.82", "Sharpe after fixing errors", None),
        ("~1,600", "specifications tested", None),
        ("2.88", "independent bets in the universe", "fail"),
        ("-0.96", "forecastability vs duration", "pass"),
        ("0.93 vs 0.74", "risk parity vs equal weight", "pass"),
    ])

    # ---------------------------------------------------------------- PART 1
    r.section("Part 1: Rebuilding the 2025 analysis")

    r.section("The errors", (
        "The original study reported a Sharpe ratio of 1.17. It had nine "
        "errors. These are the ones that mattered."))
    r.table(pd.DataFrame([
        ("Risk-free rate set to zero in every period",
         "Inflated every Sharpe ratio in the study"),
        ("Credit modeled as bonds that never default",
         "Overstated high yield by 3.4% a year"),
        ("Equity returns excluded dividends",
         "Pushed the optimizer toward bonds and gold"),
        ("Misaligned weight vector in the code",
         "The portfolio described was not the one computed"),
        ("Portfolio return as a weighted geometric mean",
         "Tables did not add up"),
        ("Off by one error dropping the last year of each block",
         "Every correlation and macro statistic"),
        ("No backtest", "Every result was in sample"),
    ], columns=["error", "effect"]).set_index("error"))
    r.prose("After fixing all of it, the Sharpe was 0.82.")

    r.section("What I built to test it", (
        "The setup mattered more than any of the models."))
    r.table(pd.DataFrame([
        ("Real risk-free curve",
         "Treasury par yields bootstrapped to a zero curve. Money market points "
         "at one and three months had to be added. Without them the "
         "interpolation snaps to the six month rate and the risk-free rate is "
         "off by 14.8 basis points. With them it is off by 1.9."),
        ("Bond returns from that curve",
         "Every Treasury is a constant maturity holding priced off the same "
         "discount function, split into carry, rolldown, duration and "
         "convexity. The 2 year and the 30 year differ only in maturity."),
        ("Sealed holdout",
         "2016 to 2026 throws an error if you read it. Unlocking it writes a "
         "timestamped log entry. This makes 'I only looked once' checkable "
         "rather than something I assert."),
        ("Specification log",
         "Every model is written to a file as it is fitted, before I know if it "
         "worked. Without this you can only correct for the models you "
         "remember running."),
        ("Leverage charged",
         "A 0.9 Sharpe at 3% volatility is not better than a 0.7 at 10% until "
         "you pay to borrow the difference. Every table shows the unlevered "
         "number, the leverage needed, and the levered number after financing."),
        ("Costs per asset",
         "2 basis points on a Treasury bill, 45 on high yield municipals. One "
         "blended rate would flatter whatever trades the illiquid legs hardest."),
        ("Publication lags",
         "CPI for March is not in the data on April 10th, because nobody had "
         "it yet."),
    ], columns=["control", "what it does"]).set_index("control"))

    r.section("What I tested, and what happened")
    r.table(pd.DataFrame([
        ("Regime allocation",
         "1,296 specifications covering regime definition, persistence, "
         "confirmation delay, probability weighting, shrinkage and optimizer",
         "85% looked good in sample. Multiple testing p-value of 1.000. Zero "
         "of the eight best were positive out of sample"),
        ("Return regression",
         "182 monthly and 179 daily signals, eight model families including "
         "elastic net, random forest and gradient boosting",
         "Three of eight positive once credit is excluded, and all tiny"),
        ("Recession timing",
         "NBER and GDP targets, with a 12 month delay on training labels "
         "because NBER dates recessions late",
         "Looked good in sample. Out of sample the classifier's AUC fell from "
         "0.82 to 0.21"),
        ("Cross sectional ranking",
         "Ranking assets against each other rather than forecasting each one",
         "Strong signal, t-statistic of 8.26, but it never became a portfolio"),
        ("Universe expansion", "7 assets to 59 assets",
         "Zero of 27 strategies positive out of sample"),
    ], columns=["approach", "what it was", "result"]).set_index("approach"))

    r.section("Why it failed, in one number")
    r.prose(
        "There is a standard result in active management that says your "
        "information ratio equals your skill multiplied by the square root of "
        "the number of independent bets you make.")
    r.prose(
        "My seven asset universe had <strong>2.88 independent bets</strong>. "
        "Gold, stocks, credit and four points on one yield curve are not seven "
        "different things. They are closer to three.")
    r.prose(
        "With a measured skill level of 0.076 and twelve decisions a year, the "
        "ceiling on the information ratio is about <strong>0.44</strong>. That "
        "is before costs, and it assumes you capture all of your skill "
        "perfectly. The signals were not the problem. The portfolio was too "
        "small to hold them.")
    r.prose(
        "One footnote worth including. My original 2025 portfolio, the "
        "20/35/30/10/5 mix, actually beat 60/40 on the sealed holdout. It "
        "ranked near the bottom on training data and near the top out of "
        "sample. It required no forecast, which is probably why it did not "
        "decay.")

    # ---------------------------------------------------------------- PART 2
    r.section("Part 2: Risk parity in fixed income")

    r.section("The finding that led here")
    r.prose(
        "Bond returns get harder to forecast as they get riskier. Across twelve "
        "fixed income assets, the rank correlation between out of sample R "
        "squared and duration is <strong>negative 0.958</strong>. Every bond "
        "with duration over five years has negative predictive R squared.")
    if fipred is not None:
        t = fipred[["oos_r2", "vol", "duration", "risk_share"]]
        t.index.name = "asset"
        r.table(t.round(4), align_right=list(t.columns))
    r.prose(
        "The reason is not complicated. A short bond's return is mostly carry, "
        "which you know when you buy it. A long bond's return is mostly the "
        "change in yields, which you do not. The same fact makes short bonds "
        "low risk.")
    r.prose(
        "Bonds are still the better place to look. Out of sample R squared "
        "reached 2.13% on the 2 year Treasury and 1.81% on high yield. On "
        "equities it was 0.05%.")

    r.section("The universe", (
        "Eleven assets, monthly, 1982 to 2026. Four Treasury maturities built "
        "from the bootstrapped curve, four credit funds, three securitized."))
    r.prose(
        "The 3 month bill is excluded. It is a cash proxy at 0.9% volatility "
        "and every risk minimizing method just piles into it. A portfolio that "
        "wants less risk should hold less, not relabel cash as an asset.")
    r.prose(
        "This universe has <strong>1.92 independent bets out of 11</strong>. "
        "The first principal component explains 70.4% of the variance. Fixed "
        "income is basically one factor, and adding more bonds does not fix "
        "that. I tried, going from 12 assets to 26, and independence went "
        "down rather than up.")

    r.section("The two methods")
    r.prose(
        "<strong>Risk parity, also called Equal Risk Contribution.</strong> "
        "Instead of putting equal money in each asset, you put equal risk in "
        "each asset. A 30 year Treasury is about eight times more volatile than "
        "a 2 year, so it gets about one eighth the weight. You solve for the "
        "weights where every asset contributes the same amount to portfolio "
        "variance.")
    r.prose(
        "<strong>Hierarchical Risk Parity, from Marcos Lopez de Prado "
        "(2016).</strong> This fixes a specific weakness in standard risk "
        "parity. Risk parity uses the full covariance matrix. When assets are "
        "highly correlated, which bonds are, that matrix is nearly singular and "
        "working with it produces unstable weights. It also treats every asset "
        "as a peer, so four nearly identical Treasury maturities look like four "
        "separate bets.")
    r.table(pd.DataFrame([
        ("1. Cluster",
         "Group assets into a tree based on how correlated they are. "
         "Treasuries end up next to Treasuries, municipals next to municipals."),
        ("2. Reorder",
         "Sort the covariance matrix so similar assets sit next to each other, "
         "which makes it close to diagonal."),
        ("3. Split top down",
         "Walk down the tree splitting capital between each pair of groups by "
         "their relative risk, then repeat inside each group."),
    ], columns=["step", "what happens"]).set_index("step"))
    r.prose(
        "The important part is that HRP <strong>never inverts the covariance "
        "matrix</strong>. It only uses it to measure distance and group "
        "variance. That is what makes it stable when everything is correlated, "
        "which is exactly our situation.")

    r.section("Results")
    if T is not None:
        d = T[["dev_cagr", "dev_vol", "dev_sharpe", "dev_vs_1N", "dev_dd"]].copy()
        d.columns = ["return", "volatility", "Sharpe", "vs equal weight",
                     "worst drawdown"]
        d.index.name = "strategy"
        r.table(d.round(4), align_right=list(d.columns),
                caption="Development period, 1982 to 2015.")
        o = T[["oos_cagr", "oos_vol", "oos_sharpe", "oos_vs_1N", "oos_dd"]].copy()
        o.columns = ["return", "volatility", "Sharpe", "vs equal weight",
                     "worst drawdown"]
        o.index.name = "strategy"
        r.table(o.round(4), align_right=list(o.columns),
                caption="Holdout period, 2016 to 2026. Nothing was fitted on this.")

    if nets is not None:
        fig, ax = charts.new_axes(9.0, 3.6)
        pick = ["Hierarchical RP", "Risk parity (ERC)", "1/N",
                "Agg index (VBMFX)", "2s10s barbell 50/50"]
        for i, c in enumerate([p for p in pick if p in nets.columns]):
            s = nets[c].dropna()
            ax.plot(s.index, (1 + s).cumprod(),
                    color=charts.SERIES[i % len(charts.SERIES)],
                    linewidth=2.0 if "Hierarchical" in c else 1.3, label=c)
        ax.axvline(pd.Timestamp("2016-01-01"), color=charts.MUTED,
                   linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("growth of 1, log scale")
        charts.legend(ax, loc="upper left")
        r.figure(charts.to_svg(fig),
                 "Dashed line marks the start of the holdout.")

    r.section("Is it just a bet on holding shorter bonds?", (
        "This was the obvious objection. Risk parity underweights volatile "
        "assets, volatility in bonds is duration, so it naturally holds less "
        "duration. If that is the whole story you could just buy shorter bonds "
        "and skip the math."))
    r.prose(
        "So I rebuilt the benchmark. Equal weight, scaled up or down to match "
        "each strategy's own duration, with the difference held in cash.")
    if DT is not None:
        d = DT.copy()
        d.index.name = "strategy"
        r.table(d.round(4), align_right=list(d.columns))
    r.prose(
        "The edge does not change. It is not a duration bet.")

    r.section("Does trading cost eat it?")
    if TS is not None and TO is not None:
        both = pd.concat([TS.add_suffix(" Sharpe"), TO.add_suffix(" turnover")],
                         axis=1)
        both.index.name = "strategy"
        r.table(both.round(4), align_right=list(both.columns),
                caption="Sharpe and annual turnover by rebalancing frequency.")
    r.prose(
        "No. Turnover is 5% to 14% a year. Risk parity actually trades less "
        "than equal weight, because bond correlations are stable while equal "
        "weight has to trade back against price drift every month. Annual "
        "rebalancing is slightly best, which says the covariance estimate is "
        "stable enough that monthly re-optimization is mostly noise.")
    r.prose(
        "Two other checks. 27 out of 27 combinations of covariance estimator, "
        "lookback window and rebalancing frequency were positive in sample. And "
        "levered to match equal weight's volatility at a 50 basis point "
        "financing spread, HRP delivers 0.87 and risk parity 0.85, against 0.74 "
        "for equal weight.")

    r.section("What I would and would not claim")
    r.checks([
        (True, "Risk parity beats equal weight on this universe",
         "0.93 and 0.88 against 0.74, both intervals exclude zero"),
        (True, "The edge is not explained by duration",
         "unchanged against a duration matched benchmark, p = 0.009 and 0.013"),
        (True, "It beats a 2s10s barbell and is ahead of the Agg index",
         "barbell 0.64, Vanguard Total Bond 0.73"),
        (True, "Costs and borrowing do not eat it",
         "turnover under 15% a year, still ahead levered"),
        (False, "It is proven out of sample",
         "2016 to 2026 is positive but small and not significant"),
        (False, "Any forecast is involved",
         "risk parity uses the covariance matrix only, no return prediction"),
    ])
    r.prose(
        "The honest summary is that the in sample result is solid and the out "
        "of sample result is directionally right but unproven. From 2016 to "
        "2026 equal weight returned 2.2% a year at a Sharpe of 0.03. There was "
        "very little for anything to separate on.")

    r.section("Limitations")
    r.table(pd.DataFrame([
        ("Mutual funds, not indices",
         "The bond funds charge 20 to 80 basis points and carry manager "
         "decisions an index would not. Their daily returns also autocorrelate, "
         "high yield at 0.29, because illiquid bonds get priced with a lag."),
        ("The Agg comparison starts in 1987",
         "That is when Vanguard Total Bond launched, so it runs on a shorter "
         "window than the other rows."),
        ("Out of sample is not significant",
         "Positive on all three benchmarks but with intervals spanning zero."),
        ("The holdout was opened once for Part 1",
         "Part 2's holdout is clean of fitting but not of me having seen that "
         "decade."),
    ], columns=["limitation", "what it means"]).set_index("limitation"))

    out = ROOT / "reports/final_paper.html"
    r.render(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
