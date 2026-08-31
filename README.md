# Fixed Income Risk Parity

Gus Guenther, UCLA Anderson MFE

A bond only portfolio built on one finding: bond returns get harder to forecast
as they get riskier.

**[Read the full writeup](https://GITHUBNAME.github.io/fixed-income-risk-parity/reports/final_paper.html)**

This is the second half of a larger project. The first half rebuilt a macro
allocation study I wrote in 2025, fixed nine errors in it, and tested about 1,600
specifications against a sealed holdout. Nothing beat a 60/40 benchmark. That
work is in
[macro-portfolio-rebuild](https://github.com/GITHUBNAME/macro-portfolio-rebuild).

---

## The finding

Bond returns get harder to forecast as they get riskier.

Across twelve fixed income assets, the rank correlation between out of sample R
squared and duration is **negative 0.958**. Every bond with duration over five
years has negative predictive R squared.

The reason is not complicated. A short bond's return is mostly carry, which you
know when you buy it. A long bond's return is mostly the change in yields, which
you do not. The same fact that makes short bonds predictable also makes them low
risk.

Bonds are still the better place to look than stocks. Out of sample R squared
reached 2.13% on the 2 year Treasury and 1.81% on high yield. On equities it was
0.05%.

## The universe

Eleven assets, monthly, 1982 to 2026.

- Four Treasury maturities (2, 5, 10, 30 year), built from a bootstrapped zero
  curve so they differ only in maturity
- Four credit funds: short, intermediate and long investment grade, plus high
  yield
- Three securitized: GNMA mortgages, intermediate municipals, high yield
  municipals

The 3 month bill is excluded. It is a cash proxy at 0.9% volatility, and every
risk minimizing method just piles into it. A portfolio that wants less risk
should hold less, not relabel cash as an asset.

**This universe has 1.92 independent bets out of 11.** The first principal
component explains 70.4% of the variance. Fixed income is basically one factor,
and adding more bonds does not fix that. I tried, going from 12 assets to 26, and
independence went down rather than up.

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

## Results

Development period, 1982 to 2015:

| Strategy | Return | Volatility | Sharpe | vs equal weight | Worst drawdown |
|---|---|---|---|---|---|
| **Hierarchical Risk Parity** | 5.84% | 2.69% | **0.93** | +0.19 | -4.2% |
| **Risk parity (ERC)** | 6.36% | 3.42% | **0.88** | +0.14 | -5.3% |
| Inverse volatility | 6.33% | 3.43% | 0.87 | +0.13 | -4.7% |
| Equal weight | 7.58% | 4.82% | 0.74 | 0.00 | -7.2% |
| Vanguard Total Bond Market | 6.22% | 3.88% | 0.73 | -0.01 | -5.8% |
| 2s10s barbell, 50/50 | 7.12% | 4.92% | 0.64 | -0.10 | -5.3% |

Holdout period, 2016 to 2026, which nothing was fitted on:

| Strategy | Sharpe | vs equal weight |
|---|---|---|
| Risk parity (ERC) | 0.06 | +0.03 |
| Hierarchical Risk Parity | 0.04 | +0.01 |
| Equal weight | 0.03 | 0.00 |
| Vanguard Total Bond Market | -0.07 | -0.10 |
| 2s10s barbell | -0.15 | -0.18 |

## Checks

**Is it just a bet on holding shorter bonds?** This was the obvious objection.
Risk parity underweights volatile assets, volatility in bonds is duration, so it
naturally holds less duration. If that is the whole story you could just buy
shorter bonds and skip the math.

So I rebuilt the benchmark: equal weight, scaled up or down to match each
strategy's own duration, with the difference held in cash.

| Strategy | vs plain equal weight | vs duration matched | 95% interval | p |
|---|---|---|---|---|
| Hierarchical Risk Parity | +0.150 | **+0.146** | [+0.033, +0.263] | 0.009 |
| Risk parity (ERC) | +0.085 | **+0.087** | [+0.022, +0.160] | 0.013 |
| Inverse volatility | +0.076 | +0.078 | [+0.024, +0.137] | 0.007 |

The edge does not change. It is not a duration bet.

**Does trading cost eat it?** No. Turnover is 5% to 14% a year. Risk parity
actually trades less than equal weight, because bond correlations are stable
while equal weight has to trade back against price drift every month.

| Strategy | Monthly | Quarterly | Semiannual | Annual |
|---|---|---|---|---|
| Equal weight | 0.74 / 0.12 | 0.74 / 0.08 | 0.74 / 0.07 | 0.74 / 0.05 |
| Risk parity | 0.88 / 0.09 | 0.88 / 0.07 | 0.88 / 0.05 | 0.89 / 0.05 |
| Hierarchical RP | 0.93 / 0.14 | 0.93 / 0.11 | 0.93 / 0.08 | 0.94 / 0.07 |

Sharpe / annual turnover. Annual rebalancing is slightly best, which says the
covariance estimate is stable enough that monthly re-optimization is mostly
noise.

**Is it robust to estimation choices?** 27 out of 27 combinations of covariance
estimator, lookback window and rebalancing frequency were positive in sample.

**Does borrowing cost eat it?** No. Levered to match equal weight's volatility at
a 50 basis point financing spread: HRP 0.87, risk parity 0.85, equal weight 0.74.

## What I would and would not claim

I would say: risk parity beats equal weight on a fixed income universe, the
margin is statistically significant in sample, it is not explained by duration,
and it survives trading and borrowing costs.

I would not say it is proven out of sample. The 2016 to 2026 numbers are positive
against all three benchmarks but small and not significant. That decade was hard
for every bond portfolio. Equal weight returned 2.2% a year at a Sharpe of 0.03,
so there was very little for anything to separate on.

## Repo layout

```
src/macro/
  curves/       zero curve bootstrap, Nelson-Siegel-Svensson, Hull-White,
                G2++, affine term structure
  data/         FRED and Yahoo clients, vendor splicing, quality checks
  portfolio/    optimizers, covariance estimators, cross section tools
  backtest/     walk forward engine, costs, leverage, sealed holdout,
                specification log
  stats/        block bootstrap, Hansen SPA, deflated Sharpe
  signals/      predictor library
  report/       HTML report builder

scripts/
  01_build_universe.py       11 asset bond panel
  02_allocation_engine.py    risk based allocators vs equal weight
  03_forecast_allocation.py  forecast driven tilts
  04_holdout_inference.py    the forecastability vs duration finding
  06_riskparity_deep.py      duration matched test, HRP, robustness
  08_shorting.py             long short and market neutral variants
  report_paper_final.py      regenerates the writeup

reports/
  final_paper.html           the full writeup
```

## Running it

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key_here

python scripts/01_build_universe.py
python scripts/06_riskparity_deep.py
python scripts/report_paper_final.py
```

A free FRED API key takes about a minute to get from
https://fred.stlouisfed.org/docs/api/api_key.html

Data files are not committed. The build scripts pull from FRED and Yahoo and
write to a local `data/` directory, which `.gitignore` excludes. The generated
report is committed so you can read the results without running anything.

## Limitations

Worth stating up front rather than having someone find them.

- The bond universe is built from **mutual funds**, not indices. They charge 20
  to 80 basis points and carry manager decisions an index would not. Their daily
  returns also autocorrelate, high yield at 0.29, because illiquid bonds get
  priced with a lag. Any daily result involving credit is affected by this.
- The **Vanguard Total Bond comparison starts in 1987**, not 1982, because that
  is when the fund launched. It runs on a shorter window than the other rows.
- The out of sample result is **positive but not statistically significant**.
- The holdout was opened once for the first half of this project, before this
  half existed. So this holdout is clean of fitting, but not of me having seen
  that decade.

## Reference

Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of
Sample. *Journal of Portfolio Management*, 42(4), 59-69.
