# Fixed Income Risk Parity

Gus Guenther, UCLA Anderson MFE

A bond-only portfolio that tries to beat its benchmark, built after a macro
allocation study showed how hard equity returns are to predict but also turned up
real evidence that fixed income returns are predictable.

**[Read the full writeup](https://guenther-qr.github.io/fixed-income-risk-parity/)**

This is the second half of a larger project. The first half rebuilt a
[2025 summer study](https://github.com/guenther-QR/All-Weather-Portfolio-Construction),
fixed nine errors in it, and tested about 1,600 specifications against a sealed
holdout. Nothing beat a 60/40 benchmark. That work is in
[macro-portfolio-rebuild](https://github.com/guenther-QR/macro-portfolio-rebuild).

---

## What this project found

I developed a range of predictive models here using regression forecasts, carry
and momentum signals, and machine learning. Those results were inconclusive, for
a reason that turned out to be structural rather than technical: in fixed income,
the assets you can forecast are the ones carrying almost none of the risk.

What did work is **risk parity**, which never needs an expected return at all. I
built and tested both classic risk parity and Marcos Lopez de Prado's
Hierarchical Risk Parity.

The benchmark is the **Bloomberg Aggregate**, proxied by Vanguard Total Bond
Market, because that is what a fixed income mandate is measured against. Equal
weight across the same eleven assets is carried as a second reference. The
Aggregate is never levered: it sets the risk budget and the strategies are scaled
to fit inside it.

| Strategy | Development<br>1987-2015 | Holdout<br>2016-2026 | Full sample<br>1987-2026 |
|---|---|---|---|
| **Hierarchical Risk Parity** | **0.933** (+0.121, p 0.036) | 0.036 (+0.115, p 0.042) | **0.659** (+0.153, p 0.002) |
| **Risk parity (ERC)** | **0.884** (+0.057, p 0.168) | 0.057 (+0.134, p 0.006) | **0.624** (+0.108, p 0.007) |
| Agg index (VBMFX) | 0.824 | -0.072 | 0.518 |
| Equal weight | 0.805 | 0.026 | 0.552 |
| 2s10s barbell, 50/50 | 0.655 | -0.152 | 0.441 |

Sharpe ratio, with the edge against the Aggregate and its one-sided p-value from
a stationary block bootstrap.

**The result clears the Aggregate in all three windows, holdout included.** That
deserves one caveat. The holdout decade contained the worst bond market in forty
years, the Aggregate returned a negative Sharpe across it, and equal weight beat
the index too (+0.103, p 0.003). So clearing the Aggregate out of sample is a
real result against the benchmark a mandate uses, and a weaker claim about skill
than it first looks. Against equal weight the holdout is +0.012 with an interval
spanning zero.

Against equal weight, for completeness:

| Strategy | Development | Holdout | Full sample |
|---|---|---|---|
| Hierarchical Risk Parity | +0.150 (p 0.009) | +0.012 (p 0.389) | +0.122 (p 0.010) |
| Risk parity (ERC) | +0.085 (p 0.015) | +0.030 (p 0.178) | +0.077 (p 0.006) |

## How these are measured

**One start date for everything.** Equal weight and the barbell need no
covariance estimate. Risk parity and hierarchical risk parity need one, so a 60-month estimation window puts their first tradeable month at
November 1987. Every series here, benchmarks included, starts there.

That window matters more than it looks. The five years before it hold the highest
absolute bond returns in the sample, but cash paid 7.45% over them, so an 11.63%
bond return was only 4.18% of excess return earned at 6.64% volatility in the
unstable rate environment after the Volcker disinflation.

| Equal weight over | Return | Cash rate | Excess | Vol | Sharpe |
|---|---|---|---|---|---|
| 1982-11 to 1987-10 | 11.63% | 7.45% | 4.18% | 6.64% | 0.575 |
| 1987-11 to 2015-12 | 6.87% | 3.22% | 3.66% | 4.41% | 0.805 |

A high-rate decade flatters nominal returns and punishes risk-adjusted ones,
worth holding onto when reading any bond result spanning the early 1980s.

**One volatility for everything.** A growth-of-1 chart puts the lowest volatility
line at the bottom, which inverts the Sharpe ranking whenever the low volatility
strategy is the better one. Hierarchical risk parity runs at 2.8% volatility
against the Aggregate's 4.17%. Every chart and headline comparison is scaled to
the Aggregate's own volatility with financing charged.

The Aggregate is never levered. Levering a benchmark to match a strategy inverts
the question: a mandate has a risk budget and the strategy has to fit inside it.
Equal weight runs hotter than the Aggregate at 4.74%, so under this convention it
is scaled *down* to 0.88x with the balance in cash rather than levered up.

## The financing assumption

Risk parity has to be levered to compete on returns, so what leverage costs is
not a detail. This assumes an institutional book with access to repo, listed
futures and cleared swaps, not a retail margin account.

Financing is charged **per asset**, the same way trading costs are, because what
it costs to lever a position depends on what instrument carries it.

| Holding | Over the risk free rate | Route |
|---|---|---|
| Treasuries, 2-30y | 3bp | Repo or the futures basis |
| Agency mortgages | 15bp | TBA dollar rolls, agency repo |
| Investment grade credit | 50bp | Total return swap plus agent fee |
| High yield | 65bp | Same structure, priced wider |
| Municipals | 110bp | Margin loan; no derivative exists |

Levering a portfolio by L borrows L-1 of NAV against the book as it stands, so
what a strategy pays is the weighted average of what its own holdings cost. That
comes to 39-42bp for every method here, a spread of under three basis points
across them. Risk parity tilts toward the 2-year Treasury at 3bp, but it tilts
just as hard toward short investment grade at 50bp and keeps most of the
municipals, so the cheap Treasury tilt does not buy cheap financing.

Scaling every position proportionally means margin-lending against the municipal
fund, which no desk would do. The alternative is an overlay: hold municipals at
cash weight and take the borrowed exposure only through instruments with a
derivative.

| Strategy | Proportional | Overlay |
|---|---|---|
| Risk parity (ERC) | 41.1bp | 28.3bp |
| Hierarchical Risk Parity | 39.3bp | 27.8bp |

Every number reported uses the proportional figure, the more expensive of the
two, so the financing drag here is an upper bound.

The unlevered comparison does not depend on any of this: HRP scores 0.659 against
the Aggregate's 0.518 over the full sample holding no leverage, and every
significance test runs on the unlevered series. The levered comparison does
depend on it, and it has a breakeven.

| Strategy | Leverage required | Pays | Breakeven | Headroom |
|---|---|---|---|---|
| Risk parity (ERC) | 1.15x | 41bp | 294bp | +252bp |
| Hierarchical Risk Parity | 1.48x | 39bp | 129bp | +90bp |

Financing would have to be seven times more expensive than assumed before classic
risk parity reverses. The more leverage a strategy needs, the more of its edge
belongs to whoever finances it, which is why HRP has the least headroom despite
the highest unlevered Sharpe. That restates what produces the result rather than
undermining it: Frazzini and Pedersen's account of the low beta anomaly is that
it persists because most investors cannot lever cheaply.

## The universe

Eleven assets, monthly, November 1987 to August 2026.

- Four Treasury maturities (2, 5, 10, 30 year), built from a bootstrapped zero
  curve so they differ only in maturity
- Four credit sleeves: short, intermediate and long investment grade, plus high
  yield
- Three securitized and municipal: GNMA mortgages, intermediate municipals, high
  yield municipals

The 3 month bill is excluded. It is a cash proxy at 0.9% volatility, and every
risk minimizing method just piles into it. A portfolio that wants less risk
should hold less, not relabel cash as an asset.

Eleven tickers is not eleven decisions. The first principal component of monthly
excess returns holds **70.4% of the variance** and the average pairwise
correlation is **0.66**, so fixed income is close to a single factor.

Summarising that spectrum with the participation ratio of the correlation matrix
eigenvalues, `N_eff = (sum L)^2 / sum(L^2)`, gives **1.92 effective independent
assets out of 11**. The eigenvalues sum to 11 and their squares to 62.94, so
`121 / 62.94 = 1.92`. Eleven uncorrelated assets would return 11.00; eleven
copies of one asset would return 1.00. This is a descriptive statistic about the
covariance structure, not a hypothesis test. The full eigenvalue spectrum is in
`reports/phase1_idea.html`.

The practical consequence is that adding more bonds does not add much. Going from
12 assets to 26 lowered this measure rather than raising it.

## Why forecasting does not work here

Rank twelve fixed income assets by how forecastable they are, then rank them
again by modified duration, and the two orderings run opposite. The three most
forecastable rank 12th, 11th and 9th by duration; the three least forecastable
rank 1st, 2nd and 3rd.

| Test | vs duration | vs volatility |
|---|---|---|
| Spearman rho | **-0.958** (p < 0.001) | -0.818 (p = 0.001) |
| Pearson r | -0.781 (p = 0.003) | -0.780 (p = 0.003) |
| OLS slope | -0.0016 per year, t = -3.96 | -0.248, t = -3.93 |
| OLS R-squared | 0.61 | 0.61 |

Twelve assets is a small sample and they are not independent draws, so the
p-values are descriptive rather than a clean test. The rank table and regression
output are in `reports/phase1_idea.html`.

The mechanism is not complicated. A short bond's return is mostly carry and
rolldown, both known when you buy it. A long bond's return is mostly the change
in yields, which is not. The same fact that makes short bonds predictable also
makes them low risk.

So a forecast-driven portfolio faces a choice with no good branch. Size positions
by conviction and the book barely moves, because conviction sits in assets with
no risk in them. Size them to matter and the portfolio's risk is dominated by
assets the forecast cannot call. Every forecast tilt tested here lands below
equal weight on development: carry 0.734, momentum 0.684, regression 0.736,
against equal weight's 0.805.

Bonds are still the better place to look than stocks. Out of sample R squared
reached 2.13% on the 2 year Treasury and 1.81% on high yield. On equities it was
0.05%.

## The two methods

**Risk parity, also called Equal Risk Contribution.** Instead of putting equal
money in each asset, you put equal risk in each asset. A 30 year Treasury is
about eight times more volatile than a 2 year, so it gets about one eighth the
weight. You solve for the weights where every asset contributes the same amount
to portfolio variance.

**Hierarchical Risk Parity, from Marcos Lopez de Prado (2016).** This fixes a
specific weakness in standard risk parity.

Risk parity uses the full covariance matrix. When assets are highly correlated,
which bonds are, that matrix is nearly singular and working with it produces
unstable weights. It also treats every asset as a peer, so four nearly identical
Treasury maturities look like four separate bets.

HRP does three things instead:

1. **Cluster.** Group the assets into a tree based on how correlated they are.
   Treasuries end up next to Treasuries, municipals next to municipals.
2. **Reorder.** Sort the covariance matrix so similar assets sit next to each
   other, which makes it close to diagonal.
3. **Split top down.** Walk down the tree splitting capital between each pair of
   groups by their relative risk, then repeat inside each group.

The important part is that HRP **never inverts the covariance matrix**. It only
uses it to measure distance and group variance. That is what makes it stable when
everything is correlated, which is exactly the situation this universe presents.

## Checks

**Is it just a bet on holding shorter bonds?** Risk parity underweights volatile
assets, volatility in bonds is duration, so it naturally holds less duration. If
that is the whole story you could just buy shorter bonds and skip the math.

So the benchmark was rebuilt: equal weight, scaled up or down month by month to
match each strategy's own duration, with the difference held in cash.

| Strategy | vs plain equal weight | vs duration matched | 95% interval | p |
|---|---|---|---|---|
| Hierarchical Risk Parity | +0.151 | **+0.147** | [+0.035, +0.263] | 0.008 |
| Risk parity (ERC) | +0.087 | **+0.088** | [+0.023, +0.160] | 0.012 |
| Group risk budget | +0.089 | +0.090 | [+0.031, +0.155] | 0.004 |

The edge does not change. It is not a duration bet.

**Does trading cost eat it?** No. Turnover is 5% to 14% a year. Risk parity
actually trades less than equal weight, because bond correlations are stable
while equal weight has to trade back against price drift every month. Annual
rebalancing is slightly best, which says the covariance estimate is stable enough
that monthly re-optimization is mostly noise.

**Is it robust to estimation choices?** 27 out of 27 combinations of covariance
estimator, lookback window and rebalancing frequency were positive on
development, and 8 of 27 on the holdout. The failures are informative: every
expanding-window specification is positive on both, and the losses concentrate in
the short 60 month lookback and the EWMA estimator, both of which throw away the
long history that makes the covariance estimate stable.

**Does borrowing cost eat it?** Not at an institutional spread. See the financing
section above for the breakevens.

## Interpreting the holdout period

The test decade contained the worst bond market in forty years, which constrains
how much any result measured over it can establish.

| | Equal weight | Agg proxy | Hierarchical RP | Risk parity |
|---|---|---|---|---|
| 2022 | -13.53% | -13.24% | -7.59% | -9.95% |
| 2021-2023 cumulative | -8.75% | -9.96% | -2.67% | -4.80% |
| Holdout worst drawdown | -17.3% | -17.5% | -10.3% | -13.0% |

The Aggregate returned a Sharpe of -0.072 over the holdout and the barbell
-0.152, while equal weight managed +0.026 on 2.2% a year. Both risk parity methods cleared the Aggregate significantly, but so did
equal weight.
When the index itself is negative, clearing it is a lower bar than it looks,
which is why the comparison against equal weight is reported alongside and why it
is the one that does not clear.

## The best of each class, on the holdout

Reporting the forecast work as a group failure is fair but not informative. The
useful question is what happened to the single model in each class that looked
best on development.

| Class | Best in class | Dev Sharpe | Dev vs Agg | Holdout Sharpe | Holdout vs Agg | p |
|---|---|---|---|---|---|---|
| return regression | Max Sharpe (forecast) | **0.954** | +0.146 | **-0.002** | +0.069 | 0.253 |
| risk only | Hierarchical Risk Parity | 0.912 | +0.104 | 0.036 | +0.107 | **0.042** |
| regime conditional | Regime covariance RP | 0.868 | +0.061 | 0.051 | +0.123 | **0.009** |
| signal tilt | Carry tilt | 0.708 | -0.099 | -0.005 | +0.066 | 0.123 |

Window is 1992-2026, shorter than elsewhere because two candidates start later.

**The best development model in the whole comparison was a forecast model.** A
long-only maximum Sharpe portfolio built on forecast means scored 0.954 on
development, better than anything risk parity managed. On the holdout its Sharpe
fell to -0.002. It still shows a positive edge over the Aggregate because the
Aggregate was negative over that decade, but it is not significant and the
strategy delivered no risk-adjusted return at all.

The regime result splits cleanly. Conditioning the **covariance** on the regime
is roughly free and survives out of sample. Conditioning the **mean** does not: a
regime conditional maximum Sharpe scores 0.847 on development and -0.130 on the
holdout. Second moments estimate; first moments do not.

Only the risk-only and regime-covariance methods clear the Aggregate
significantly out of sample. Every construction that needs a forecast of returns
lands at p > 0.10 or worse, including the two that scored best on development.

## Conclusions

Supported by the evidence: risk parity outperforms the Bloomberg Aggregate on a
fixed income universe, significantly and in all three windows; it also
outperforms equal weight in sample and over the full sample; the margin is not
explained by duration; and it survives trading costs and institutional financing
with 90 to 252bp of headroom to the breakeven spread.

Not established: that the out of sample result demonstrates skill rather than
diversification. It clears the Aggregate at p = 0.042, but equal weight cleared
the Aggregate too, and against equal weight the holdout is +0.012 with an
interval spanning zero.

## The three phases

| Phase | Report | Scripts |
|---|---|---|
| 1. The Idea | `reports/phase1_idea.html` | `phase1_build_universe.py`, `phase1_wide_universe.py`, `phase1_evidence.py` |
| 2. Strategies | `reports/phase2_strategies.html` | `phase2_allocators.py`, `phase2_forecast_tilts.py`, `phase2_risk_parity.py` |
| 3. Results and Holdout | `reports/phase3_results.html` | `phase3_predictability.py`, `phase3_aligned_results.py`, `phase3_shorting.py` |

`reports/index.html` summarizes all three and links to each one.

## Repo layout

```
index.html            redirects to reports/index.html for GitHub Pages
reports/              one HTML per phase, plus the index
scripts/              one or more scripts per phase, named to match
  build_site.py       regenerates every report from the saved data
src/macro/
  curves/             zero curve bootstrap and term structure models
  data/               FRED and Yahoo clients, vendor splicing, quality checks
  portfolio/          optimizers, covariance estimators, cross section tools
  backtest/           walk forward engine, costs, leverage, sealed holdout,
                      specification log
  stats/              block bootstrap, Hansen SPA, deflated Sharpe
  signals/            predictor library
  report/             HTML report builder
```

## Running it

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key_here

python scripts/phase1_build_universe.py
python scripts/phase1_evidence.py
python scripts/phase2_allocators.py
python scripts/phase2_risk_parity.py
python scripts/phase3_predictability.py
python scripts/phase3_aligned_results.py
python scripts/build_site.py
```

A free FRED API key takes about a minute to get from
https://fred.stlouisfed.org/docs/api/api_key.html

Data files are not committed. The build scripts pull from FRED and Yahoo and
write to a local `data/` directory, which `.gitignore` excludes. The reports are
committed so you can read the results without running anything.

## Limitations

- The bond universe is built from **mutual funds**, not indices. They charge 20
  to 80 basis points and carry manager decisions an index would not. Their daily
  returns also autocorrelate, high yield at 0.29, because illiquid bonds get
  priced with a lag. Any daily result involving credit is affected by this.
- The out of sample result is **positive but not statistically significant**.
- The effective breadth figure is a descriptive statistic about the covariance
  structure, not a test, and it depends on the sample window used to estimate the
  correlation matrix.
- The holdout period was opened once on the first half of this project. It is
  clean of any model being fitted to it, but not of having been seen once before
  this half began.

## Reference

Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of
Sample. *Journal of Portfolio Management*, 42(4), 59-69.
