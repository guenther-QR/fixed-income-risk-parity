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


def get(name):
    try:
        return pd.read_parquet(P / f"{name}.parquet")
    except Exception:
        return None


# ------------------------------------------------------------------ phase 1

def phase1():
    r = PhaseReport(
        phase="Phase 1", title="The Idea",
        summary=("This project starts where another one ended. A macro "
                 "allocation study tested about 1,600 specifications against a "
                 "sealed holdout and nothing beat 60/40. One result did hold up, "
                 "and it pointed at bonds."),
        status="complete")

    fipred = get("fi_predictability_risk")
    stats = get("fi_asset_stats")
    r.metrics([
        ("~1,600", "specifications that failed first", None),
        ("-0.96", "forecastability vs duration", "pass"),
        ("2.13%", "best bond out of sample R squared", "pass"),
        ("0.05%", "equity out of sample R squared", "fail"),
        ("11", "assets in this universe", None),
    ])

    r.section("Where this comes from", (
        "The prior project rebuilt a 2025 macro study, fixed nine accounting "
        "errors in it, and gave it the out of sample test it never had."))
    r.prose(
        "The corrected Sharpe was 0.82, down from a reported 1.17. Then about "
        "1,600 specifications were tested: regime allocation across 1,296 "
        "design combinations, return regression on 182 signals, eight machine "
        "learning families, recession timing, cross sectional ranking, at both "
        "monthly and daily frequency, on universes ranging from 7 to 59 assets. "
        "None beat a 60/40 benchmark out of sample.")
    r.prose(
        "The reason turned out to be measurable rather than mysterious. An "
        "information ratio is skill multiplied by the square root of the number "
        "of independent bets. That universe contained <strong>2.88 independent "
        "bets</strong>, which caps the achievable information ratio near 0.44 "
        "before costs. The signals were fine. The portfolio was too small to "
        "hold them.")

    r.section("The finding that survived", (
        "Buried in the failure was one relationship that held up everywhere it "
        "was tested."))
    r.prose(
        "<strong>Bond returns get harder to forecast as they get riskier.</strong> "
        "Across twelve fixed income assets, the rank correlation between out of "
        "sample R squared and modified duration is negative 0.958, with p below "
        "0.001. Against volatility it is negative 0.818. Every bond with "
        "duration above five years has negative predictive R squared.")
    if fipred is not None:
        cols = [c for c in ["oos_r2", "vol", "duration", "risk_share"]
                if c in fipred.columns]
        t = fipred[cols]
        t.index.name = "asset"
        r.table(t.round(4), align_right=cols)

        def f(ax):
            ax.scatter(fipred["duration"], fipred["oos_r2"] * 100, s=70,
                       color=charts.SERIES[0], zorder=3)
            for a, row in fipred.iterrows():
                ax.annotate(a, (row["duration"], row["oos_r2"] * 100),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=charts.MUTED)
            ax.axhline(0, color=charts.MUTED, linewidth=1.0, linestyle="--")
            ax.set_xlabel("modified duration, years")
            ax.set_ylabel("out of sample R squared, percent")
        fig, ax = charts.new_axes(8.0, 3.6)
        f(ax)
        r.figure(charts.to_svg(fig),
                 "Rank correlation negative 0.958. Every asset above five years "
                 "of duration sits below the line.")

    r.prose(
        "The reason is not complicated. A short bond's return is mostly carry, "
        "which you know when you buy it. A long bond's return is mostly the "
        "change in yields, which you do not. The same fact that makes short "
        "bonds predictable also makes them low risk. Predictability and risk "
        "turn out to be two readings of one quantity.")
    r.prose(
        "This was first noticed with equities in the universe, where the obvious "
        "objection was that equities were doing the work. Removing them made the "
        "relationship stronger, not weaker. It is a property of bonds.")

    r.section("Why bonds anyway", (
        "If forecastability runs inverse to risk, why not give up on "
        "forecasting? Because the absolute levels still favour bonds."))
    r.prose(
        "Out of sample R squared against the prevailing mean reached 2.13% on "
        "the two year Treasury and 1.81% on high yield. On equities it was "
        "0.05%. Nine of twelve fixed income assets are forecastable out of "
        "sample; two of seven multi asset ones are. Whatever is going to work, "
        "it is more likely to work here.")

    r.section("The universe", (
        "Eleven assets, monthly, 1982 to 2026."))
    r.table(pd.DataFrame([
        ("Treasuries", "2, 5, 10 and 30 year constant maturity holdings, built "
         "from a bootstrapped zero curve so they differ only in maturity"),
        ("Credit", "Short, intermediate and long investment grade, plus high "
         "yield corporate"),
        ("Securitized", "GNMA mortgages, intermediate municipals, high yield "
         "municipals"),
    ], columns=["group", "what is in it"]).set_index("group"))
    if stats is not None:
        cols = [c for c in ["group", "cagr", "vol", "sharpe", "corr_ust10y"]
                if c in stats.columns]
        t = stats[cols]
        t.index.name = "asset"
        r.table(t.round(4), align_right=[c for c in cols if c != "group"])
    r.prose(
        "The three month bill is excluded. It is a cash proxy at 0.9% "
        "volatility, and every risk minimizing method just piles into it. A "
        "portfolio that wants less risk should hold less, not relabel cash as an "
        "asset.")
    r.prose(
        "<strong>This universe has 1.92 independent bets out of 11.</strong> The "
        "first principal component explains 70.4% of the variance. Fixed income "
        "is basically one factor, and adding more bonds does not fix it. Going "
        "from 12 assets to 26 made independence go down rather than up, because "
        "the additions were redundant: interpolated curve points, a second "
        "mortgage fund correlated 0.98 with the first, duplicate managers in the "
        "same sector.")

    r.next_up("Phase 2 - Strategies", [
        "Regression and regime approaches, which do not work here either",
        "Risk parity, which does",
        "Lopez de Prado's hierarchical extension",
    ])
    return r.render(OUT / "phase1_idea.html")


# ------------------------------------------------------------------ phase 2

def phase2():
    r = PhaseReport(
        phase="Phase 2", title="Strategies",
        summary=("Three approaches on the same universe. Forecast driven tilts "
                 "and regime conditioning both fail, in the same way they failed "
                 "on the multi asset book. Risk parity works, and it works "
                 "without any forecast at all."),
        status="complete")

    fsum = get("fi_forecast_summary")
    dev = get("fi_paper_table")
    r.metrics([
        ("9 of 12", "assets forecastable", "pass"),
        ("-0.07", "best forecast tilt vs 1/N", "fail"),
        ("+0.14", "risk parity vs 1/N", "pass"),
        ("+0.19", "hierarchical risk parity vs 1/N", "pass"),
        ("0", "forecasts used by the winner", None),
    ])

    r.section("What does not work", (
        "Worth showing, because it is the same failure as the parent project and "
        "it is what makes the positive result believable."))
    r.prose(
        "Nine of twelve assets are forecastable out of sample here, a better hit "
        "rate than the multi asset universe achieved. Tilting the portfolio "
        "toward the assets with the highest forecast return still loses.")
    if fsum is not None:
        cols = [c for c in ["cagr", "vol", "sharpe", "vs_1N", "max_drawdown"]
                if c in fsum.columns]
        t = fsum[cols]
        t.index.name = "strategy"
        r.table(t.round(4), align_right=cols,
                caption="Development sample, net of costs.")
    r.prose(
        "Carry loses 0.07, momentum 0.12, the regression tilt 0.07. Carry is the "
        "interesting one: it is fixed income's native signal and needs no "
        "statistical estimate at all, and it still does not beat equal weight.")
    r.prose(
        "Regime conditioning fails the same way. On the parent project a 1,296 "
        "specification grid produced 85% positive results in sample, a multiple "
        "testing p value of 1.000, and zero of eight winners positive on the "
        "holdout. Conditioning even the <em>covariance</em> on the regime, which "
        "is a much narrower claim than conditioning the mean, made a risk parity "
        "book worse by 0.11 Sharpe.")
    r.prose(
        "The pattern across both projects is consistent. The assets that carry "
        "the risk are the ones nobody can forecast, so a forecast can only move "
        "the portfolio through positions too small to matter.")

    r.section("Risk parity", (
        "Also called Equal Risk Contribution. It uses no forecast of any kind."))
    r.prose(
        "Instead of putting equal money in each asset, you put equal risk in "
        "each asset. A 30 year Treasury is about eight times more volatile than "
        "a 2 year, so it gets roughly one eighth the weight. Formally you solve "
        "for the weights where every asset's risk contribution "
        "<code>w<sub>i</sub> (&Sigma;w)<sub>i</sub> / &sigma;<sub>p</sub></code> "
        "is equal.")
    r.prose(
        "The covariance matrix is re-estimated every month on all data available "
        "to that point, using Ledoit-Wolf shrinkage toward a constant "
        "correlation target. The expected return vector never enters the "
        "objective. That is the whole method.")
    r.prose(
        "What it harvests is the low beta anomaly documented by Frazzini and "
        "Pedersen (2014): low volatility assets have historically delivered "
        "better risk adjusted returns than high volatility ones. Their argument "
        "is that the anomaly exists <em>because</em> investors are leverage "
        "constrained, which is why any strategy harvesting it runs at low "
        "volatility and needs borrowing to compete on a ratio basis. That is "
        "also why the leverage accounting in Phase 3 is not a technicality.")

    r.section("Lopez de Prado's extension", (
        "Hierarchical Risk Parity, from the Journal of Portfolio Management, "
        "2016. It fixes a specific weakness in standard risk parity."))
    r.prose(
        "Risk parity works with the full covariance matrix. When assets are "
        "highly correlated, which bonds are, that matrix is nearly singular and "
        "the weights it produces become unstable. It also treats every asset as "
        "a peer, so four nearly identical Treasury maturities look like four "
        "separate bets rather than one bet held four ways.")
    r.table(pd.DataFrame([
        ("1. Cluster", "Convert the correlation matrix into a distance measure, "
         "then build a tree by joining the closest assets first. Treasuries end "
         "up next to Treasuries, municipals next to municipals."),
        ("2. Quasi-diagonalize", "Reorder the matrix so that similar assets sit "
         "next to each other. The result is close to block diagonal, which means "
         "the structure is visible without inverting anything."),
        ("3. Recursive bisection", "Split the tree in two, allocate between the "
         "halves in inverse proportion to their cluster variance, then repeat "
         "inside each half until you reach individual assets."),
    ], columns=["step", "what happens"]).set_index("step"))
    r.prose(
        "<strong>The key property is that HRP never inverts the covariance "
        "matrix.</strong> It uses it only to measure distance between assets and "
        "to compute the variance of a cluster. Matrix inversion is where "
        "estimation error gets amplified, and a near singular matrix amplifies "
        "it violently. Avoiding the inversion is what makes HRP stable in "
        "exactly the situation we are in.")
    r.prose(
        "The second property matters as much here. Because allocation happens "
        "between clusters before it happens between assets, adding a redundant "
        "bond does not dilute the rest of the portfolio. In a universe where the "
        "first principal component explains 70% of variance and several "
        "instruments are near duplicates, that is the difference between "
        "measuring diversification and assuming it.")

    if dev is not None:
        cols = [c for c in ["dev_cagr", "dev_vol", "dev_sharpe", "dev_vs_1N",
                            "dev_dd"] if c in dev.columns]
        t = dev[cols].copy()
        t.columns = ["return", "volatility", "Sharpe", "vs equal weight",
                     "worst drawdown"][:len(cols)]
        t.index.name = "strategy"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Development period, 1982 to 2015.")

    r.next_up("Phase 3 - Results and Holdout", [
        "Is the edge just a bet on shorter bonds?",
        "Turnover, costs and borrowing",
        "The sealed holdout, and what comes next",
    ])
    return r.render(OUT / "phase2_strategies.html")


# ------------------------------------------------------------------ phase 3

def phase3():
    r = PhaseReport(
        phase="Phase 3", title="Results, Holdout and Next Steps",
        summary=("The in sample result is solid and survives the obvious "
                 "objections. The out of sample result is positive on all three "
                 "benchmarks but not statistically significant, and this page "
                 "says so plainly."),
        status="complete")

    T = get("fi_paper_table")
    DT = get("fi_paper_duration")
    TS = get("fi_paper_turnover_sharpe")
    TO = get("fi_paper_turnover")
    nets = get("fi_bench_final")

    r.metrics([
        ("0.93", "hierarchical RP Sharpe", "pass"),
        ("0.74", "equal weight Sharpe", None),
        ("+0.146", "edge after duration matching", "pass"),
        ("0.009", "development p-value", "pass"),
        ("not sig.", "holdout result", "fail"),
    ])

    r.section("Results against three benchmarks")
    if T is not None:
        cols = [c for c in ["dev_cagr", "dev_vol", "dev_sharpe", "dev_vs_1N",
                            "dev_dd"] if c in T.columns]
        t = T[cols].copy()
        t.columns = ["return", "volatility", "Sharpe", "vs equal weight",
                     "worst drawdown"][:len(cols)]
        t.index.name = "strategy"
        r.table(t.round(4), align_right=list(t.columns),
                caption="Development period, 1982 to 2015.")
        cols = [c for c in ["oos_cagr", "oos_vol", "oos_sharpe", "oos_vs_1N",
                            "oos_dd"] if c in T.columns]
        o = T[cols].copy()
        o.columns = ["return", "volatility", "Sharpe", "vs equal weight",
                     "worst drawdown"][:len(cols)]
        o.index.name = "strategy"
        r.table(o.round(4), align_right=list(o.columns),
                caption="Holdout period, 2016 to 2026. Nothing was fitted on this.")

    if nets is not None:
        fig, ax = charts.new_axes(9.0, 3.6)
        for i, c in enumerate(nets.columns):
            s = nets[c].dropna()
            ax.plot(s.index, (1 + s).cumprod(),
                    color=charts.SERIES[i % len(charts.SERIES)],
                    linewidth=2.0 if "Hierarchical" in c else 1.2, label=c)
        ax.axvline(pd.Timestamp("2016-01-01"), color=charts.MUTED,
                   linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("growth of 1, log scale")
        charts.legend(ax, loc="upper left")
        r.figure(charts.to_svg(fig),
                 "Dashed line marks the start of the holdout.")

    r.section("Is it just a bet on shorter bonds?", (
        "This is the objection that should kill the result if anything does. "
        "Risk parity underweights volatile assets, volatility in bonds is "
        "duration, so it naturally holds less duration than equal weight."))
    r.prose(
        "If that is the whole story, an investor who wants less duration can "
        "simply hold less duration and skip the covariance matrix entirely. So "
        "the benchmark was rebuilt: equal weight, scaled up or down to match "
        "each strategy's own portfolio duration, with the difference held in "
        "cash at the risk free rate.")
    if DT is not None:
        d = DT.copy()
        d.index.name = "strategy"
        r.table(d.round(4), align_right=list(d.columns))
    r.prose(
        "<strong>The edge does not change.</strong> Hierarchical risk parity "
        "goes from +0.150 against plain equal weight to +0.146 against the "
        "duration matched version. All three intervals exclude zero. It is not a "
        "duration bet.")

    r.section("Costs, turnover and borrowing")
    if TS is not None and TO is not None:
        both = pd.concat([TS.add_suffix(" Sharpe"), TO.add_suffix(" turnover")],
                         axis=1)
        both.index.name = "strategy"
        r.table(both.round(4), align_right=list(both.columns),
                caption="By rebalancing frequency.")
    r.prose(
        "Turnover is 5% to 14% a year. Risk parity actually trades <em>less</em> "
        "than equal weight, because bond correlations are stable while equal "
        "weight has to trade back against price drift every month. Annual "
        "rebalancing is marginally best, which says the covariance estimate is "
        "stable enough that monthly re-optimization is mostly noise.")
    r.prose(
        "Two more checks. 27 out of 27 combinations of covariance estimator, "
        "lookback window and rebalancing frequency were positive in sample. And "
        "levered to match equal weight's volatility at a 50 basis point "
        "financing spread, hierarchical risk parity delivers 0.87 and risk "
        "parity 0.85, against 0.74 for equal weight. Neither costs nor borrowing "
        "eat the result.")

    r.section("What I would and would not claim")
    r.checks([
        (True, "Risk parity beats equal weight on this universe",
         "0.93 and 0.88 against 0.74, both bootstrap intervals exclude zero"),
        (True, "The edge is not explained by duration",
         "unchanged against a duration matched benchmark, p = 0.009 and 0.013"),
        (True, "It beats a 2s10s barbell and leads the Agg index",
         "barbell 0.64, Vanguard Total Bond 0.73"),
        (True, "Costs and borrowing do not eat it",
         "turnover under 15% a year, still ahead when levered"),
        (False, "It is proven out of sample",
         "2016 to 2026 is positive on all three benchmarks but not significant"),
        (False, "Any forecast is involved",
         "the covariance matrix only, no return prediction anywhere"),
    ])
    r.prose(
        "The honest summary is that the in sample result is solid and the out of "
        "sample result is directionally right but unproven. From 2016 to 2026 "
        "equal weight returned 2.2% a year at a Sharpe of 0.03. There was very "
        "little for anything to separate on.")

    r.section("Next steps")
    r.table(pd.DataFrame([
        ("Replace funds with indices or futures",
         "The universe is mutual funds, which charge 20 to 80 basis points and "
         "carry manager decisions. High yield's daily returns autocorrelate at "
         "0.29 from stale pricing. Index or futures data removes both problems "
         "and would make the daily work trustworthy."),
        ("Add genuinely different exposures",
         "Adding more US bonds reduced independence rather than raising it. TIPS, "
         "international sovereigns, emerging market debt and bank loans are "
         "different factors, not more of the same one."),
        ("Test the result on a second market",
         "If the duration and forecastability relationship is structural it "
         "should appear in gilts and bunds too. That would be much stronger "
         "evidence than another robustness check on the same data."),
        ("Cost aware optimisation",
         "On the parent project, trading a fraction of the way toward the target "
         "each period instead of fully rebalancing raised Sharpe from 0.06 to "
         "0.23 with turnover cut ninefold. Turnover is already low here, so the "
         "gain would be smaller, but the method is worth carrying over."),
    ], columns=["step", "why"]).set_index("step"))

    r.section("Limitations")
    r.table(pd.DataFrame([
        ("Mutual funds, not indices",
         "Fees of 20 to 80 basis points and manager idiosyncrasy. Daily returns "
         "autocorrelate, high yield at 0.29, because illiquid bonds are priced "
         "with a lag."),
        ("The Agg comparison starts in 1987",
         "That is when Vanguard Total Bond launched, so it runs on a shorter "
         "window than the other rows."),
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
    rows = [
        ("1", "The Idea", "phase1_idea.html",
         "Where this comes from: a macro project where 1,600 specifications "
         "failed, and the one finding that survived. Bond returns get harder to "
         "forecast as duration rises, at a rank correlation of negative 0.96."),
        ("2", "Strategies", "phase2_strategies.html",
         "Forecast tilts and regime conditioning fail here too. Risk parity "
         "works without any forecast, and Lopez de Prado's hierarchical version "
         "works better. How both methods actually operate."),
        ("3", "Results and Holdout", "phase3_results.html",
         "Against equal weight, a Treasury barbell and the Agg index. The "
         "duration matched test that should have killed it. Costs, turnover, "
         "borrowing, and the sealed holdout."),
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
h1{{font-size:2.1rem;margin:.5rem 0 .4rem;letter-spacing:-.02em}}
.sub{{color:var(--mut);max-width:40rem;margin:0 0 .6rem}}
.stats{{display:flex;flex-wrap:wrap;gap:1.6rem;padding:1.4rem 0;margin:1.6rem 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
.stat b{{display:block;font-family:"IBM Plex Mono",monospace;font-size:1.4rem;font-weight:500}}
.stat span{{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}}
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
<p class="sub">A bond only portfolio built on one finding: bond returns get
harder to forecast as they get riskier. Risk parity beats equal weight by 0.14
Sharpe in sample, and Lopez de Prado's hierarchical version by 0.19. The edge
survives duration matching, costs and borrowing.</p>
<div class="stats">
  <div class="stat"><b>-0.96</b><span>forecastability vs duration</span></div>
  <div class="stat"><b>0.93</b><span>HRP Sharpe</span></div>
  <div class="stat"><b>0.74</b><span>equal weight</span></div>
  <div class="stat"><b>0.009</b><span>p-value, duration matched</span></div>
</div>
{cards}
<footer>
The project this grew out of:
<a class="plain" href="https://guenther-QR.github.io/macro-portfolio-rebuild/">macro-portfolio-rebuild</a>.
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
