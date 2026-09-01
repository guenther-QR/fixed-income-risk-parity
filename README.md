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
Hierarchical Risk Parity, and both beat every benchmark on the development
sample. They stay positive out of sample but are not significant there. Over the
full sample the edge is significant again.

| Strategy | Development<br>1987-2015 | Holdout<br>2016-2026 | Full sample<br>1987-2026 |
|---|---|---|---|
| **Hierarchical Risk Parity** | **0.933** (+0.150, p 0.009) | 0.036 (+0.012, p 0.389) | **0.659** (+0.122, p 0.010) |
| **Risk parity (ERC)** | **0.884** (+0.085, p 0.015) | 0.057 (+0.030, p 0.178) | **0.624** (+0.077, p 0.006) |
| Inverse volatility | 0.874 (+0.076, p 0.009) | 0.033 (+0.006, p 0.386) | 0.609 (+0.062, p 0.009) |
| Equal weight | 0.805 | 0.026 | 0.552 |
| Agg proxy (Vanguard Total Bond) | 0.824 (+0.028, p 0.286) | -0.072 (-0.103, p 0.001) | 0.518 (-0.031, p 0.219) |
| 2s10s barbell, 50/50 | 0.655 (-0.144, p 0.075) | -0.152 (-0.170, p 0.042) | 0.441 (-0.106, p 0.083) |

Sharpe ratio, with the edge against equal weight and its one-sided p-value from a
stationary block bootstrap. Every series starts November 1987, see below.

## Two things I had to correct

**Start dates.** Equal weight and the barbell need no covariance estimate, so
they originally ran from November 1982. Risk parity, hierarchical risk parity and
inverse volatility need one, so a 60-month estimation window pushed them to
November 1987. Comparing series measured over different windows is not a
comparison, whichever way the bias runs.

And the bias does not run the way you would guess. Those five extra years hold
the highest absolute bond returns in the sample, so the instinct is that dropping
them should make the benchmark look worse. It does the opposite.

| Equal weight over | Return | Cash rate | Excess | Vol | Sharpe |
|---|---|---|---|---|---|
| 1982-11 to 1987-10, dropped | 11.63% | 7.45% | 4.18% | 6.64% | 0.575 |
| 1987-11 to 2015-12, kept | 6.87% | 3.22% | 3.66% | 4.41% | 0.805 |
| 1982-11 to 2015-12, as first run | 7.58% | 3.85% | 3.72% | 4.82% | 0.744 |

Cash paid 7.45% over those five years. An 11.63% bond return is a large number
and a mediocre one at the same time: 4.18% of excess return, earned at 6.64%
volatility in the unstable rate environment after the Volcker disinflation. That
is a Sharpe of 0.575 against 0.805 for the period that follows. The extra window
was dragging equal weight's Sharpe **down**, not lifting it.

So aligning the start dates makes the benchmark **harder**, and every edge
measured against it shrinks by roughly six basis points of Sharpe.

| Strategy | Edge as first reported | Edge on a common start |
|---|---|---|
| Hierarchical Risk Parity | +0.190 | +0.129 |
| Risk parity (ERC) | +0.140 | +0.079 |
| Inverse volatility | +0.130 | +0.069 |
| Agg proxy | -0.012 | +0.019 |

Everything now starts November 1987, benchmarks included. The result survives but
it is a third smaller than it was, and the Agg proxy moves from behind equal
weight to slightly ahead of it. It is also a reminder that a high-rate decade
flatters nominal returns and punishes Sharpe ratios, worth holding onto when
reading any bond result spanning the early 1980s.

**Volatility.** A growth-of-1 chart puts the lowest-volatility line at the
bottom, which inverts the Sharpe ranking whenever the low-volatility strategy is
the better one. Hierarchical risk parity runs at 2.8% volatility against equal
weight's 4.7%. Every chart and headline comparison is now levered to a common
volatility target with a 50 basis point financing spread charged on the borrowed
portion.

| Strategy | Leverage | Return | Volatility | Sharpe |
|---|---|---|---|---|
| **Hierarchical Risk Parity** | 1.68x | 5.75% | 4.7% | **0.598** |
| **Risk parity (ERC)** | 1.31x | 5.77% | 4.7% | **0.595** |
| Inverse volatility | 1.30x | 5.70% | 4.7% | 0.581 |
| Equal weight | 1.00x | 5.57% | 4.7% | 0.552 |
| Agg proxy | 1.14x | 5.33% | 4.7% | 0.506 |
| 2s10s barbell | 1.05x | 5.00% | 4.7% | 0.436 |

Full sample, levered to equal weight's own volatility. Note that leverage costs
HRP real Sharpe, 0.659 unlevered down to 0.598, and it still leads.

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

**This universe has 1.92 independent bets out of 11.** The first principal
component explains 70.4% of the variance. Fixed income is basically one factor,
and adding more bonds does not fix that. I tried, going from 12 assets to 26, and
independence went down rather than up.

## Why forecasting does not work here

Across twelve fixed income assets, the rank correlation between out of sample R
squared and duration is **negative 0.958**. Every bond with duration over five
years has negative predictive R squared.

The reason is not complicated. A short bond's return is mostly carry and
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
everything is correlated, which is exactly our situation.

## Checks

**Is it just a bet on holding shorter bonds?** This was the obvious objection.
Risk parity underweights volatile assets, volatility in bonds is duration, so it
naturally holds less duration. If that is the whole story you could just buy
shorter bonds and skip the math.

So I rebuilt the benchmark: equal weight, scaled up or down month by month to
match each strategy's own duration, with the difference held in cash.

| Strategy | vs plain equal weight | vs duration matched | 95% interval | p |
|---|---|---|---|---|
| Hierarchical Risk Parity | +0.151 | **+0.147** | [+0.035, +0.263] | 0.008 |
| Risk parity (ERC) | +0.087 | **+0.088** | [+0.023, +0.160] | 0.012 |
| Inverse volatility | +0.077 | +0.079 | [+0.025, +0.138] | 0.006 |
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

**Does borrowing cost eat it?** No. See the leverage table above.

## Why the holdout is hard to read

The test decade contained the worst bond market in forty years.

| | Equal weight | Agg proxy | Hierarchical RP | Risk parity |
|---|---|---|---|---|
| 2022 | -13.53% | -13.24% | -7.59% | -9.95% |
| 2021-2023 cumulative | -8.75% | -9.96% | -2.67% | -4.80% |
| Holdout worst drawdown | -17.3% | -17.5% | -10.3% | -13.0% |

Equal weight returned 2.2% a year over the holdout at a Sharpe of 0.026. The Agg
proxy was negative on a risk-adjusted basis, and the barbell more so. When the
benchmark itself is at zero, there is very little for a strategy to separate on.
A positive but small margin is what a real edge looks like under those
conditions, and it is also what noise looks like, which is why I do not claim
significance.

## What I would and would not claim

I would say: risk parity beats equal weight on a fixed income universe, the
margin is significant on development and on the full sample, it is not explained
by duration, and it survives trading and borrowing costs.

I would not say it is proven out of sample. The 2016 to 2026 numbers are positive
against all three benchmarks but small, with intervals spanning zero.

## The three phases

| Phase | Report | Scripts |
|---|---|---|
| 1. The Idea | `reports/phase1_idea.html` | `phase1_build_universe.py`, `phase1_wide_universe.py` |
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

Worth stating up front rather than having someone find them.

- The bond universe is built from **mutual funds**, not indices. They charge 20
  to 80 basis points and carry manager decisions an index would not. Their daily
  returns also autocorrelate, high yield at 0.29, because illiquid bonds get
  priced with a lag. Any daily result involving credit is affected by this.
- The sample **starts in November 1987**. Five years of available data are
  discarded so every strategy and benchmark shares a start date. That is the
  right trade, but it costs sample size.
- The out of sample result is **positive but not statistically significant**.
- The holdout was opened once for the first half of this project, before this
  half existed. So this holdout is clean of fitting, but not of me having seen
  that decade.

## Reference

Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of
Sample. *Journal of Portfolio Management*, 42(4), 59-69.
