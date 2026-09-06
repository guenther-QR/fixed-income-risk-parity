# Fixed Income Risk Parity

Gus Guenther, September 2026

**[Read the full writeup](https://guenther-qr.github.io/fixed-income-risk-parity/)**

This project builds on my
[macro portfolio rebuild](https://github.com/guenther-QR/macro-portfolio-rebuild),
which showed how difficult equity returns are to predict but also produced real
evidence that fixed income returns are predictable. The goal here was to take
that evidence seriously and build a bond-only portfolio that beats its
benchmark.

---

## The asset universe

Eleven fixed income assets: four constant maturity Treasury holdings at 2, 5, 10
and 30 years, four corporate credit sleeves running from short investment grade
through to high yield, and three securitized and municipal positions covering
agency mortgages, intermediate municipals and high yield municipals. It uses the
same walk-forward backtesting framework as the prior project, so results across
the two are directly comparable.

The benchmark is the Bloomberg Aggregate, proxied by Vanguard Total Bond Market.
Daily data from November 1982, development through 2015, holdout from 2016 to
August 2026.

## What was tested

Univariate regression forecasts combined across 256 signals, four machine
learning families tuned on the development sample, three technical momentum
models including an adaptive rolling window of my own, and two risk-based
methods built from the covariance matrix alone.

## What worked

Risk parity strategies, which work by utilizing risk-based position weighting
instead of return forecasting. I built and tested both classic risk parity and
Marcos Lopez de Prado's Hierarchical Risk Parity.

| Strategy | Development<br>1987-2015 | Holdout<br>2016-2026 | Full sample<br>1987-2026 |
|---|---|---|---|
| HRP + 60m rolling adaptive momentum overlay | 1.056 (+0.232\*\*\*) | −0.006 (+0.073) | 0.721 (+0.208\*\*\*) |
| **HRP, annual** | 0.952 (+0.128\*\*) | 0.057 (+0.135\*\*) | 0.676 (+0.163\*\*\*) |
| **ERC, annual** | 0.886 (+0.062) | 0.082 (+0.161\*\*\*) | 0.636 (+0.123\*\*\*) |
| 60m rolling adaptive momentum | 0.901 (+0.077) | −0.017 (+0.061) | 0.616 (+0.103) |
| HRP + Momentum (Sharpe) overlay | 0.965 (+0.141\*\*) | −0.030 (+0.048) | 0.579 (+0.067) |
| Agg index | 0.824 | −0.078 | 0.513 |
| Momentum (Sharpe) | 0.696 (−0.128) | −0.415 (−0.336\*) | 0.265 (−0.248\*\*) |

Sharpe ratio, with the edge against the Aggregate in brackets. Stars mark
one-sided bootstrap significance: \*\*\* below 0.01, \*\* below 0.05, \* below 0.10.

Sharpe ratios and significance are measured on monthly returns. Portfolios are
built, traded and costed daily, and the covariance matrix is estimated from
daily data, because sampling the same span more finely improves a variance
estimate. Performance is scored monthly, because three of the eleven holdings
are priced by matrix valuation and their daily returns autocorrelate around
0.25, which distorts a daily Sharpe ratio.

## Conclusions

**Risk parity beat the Aggregate in every window we tested.** Hierarchical risk
parity clears the index on development, on the holdout and on the full sample.
Equal risk contribution clears it on the holdout and the full sample but not on
development. Neither result depends on a return forecast at any point in the
construction.

**The margin is not paid for with risk.** Both methods run below the index on
volatility. On duration they sit either side of it: hierarchical risk parity at
4.1 years against the index at 4.2, and equal risk contribution at 5.1. Neither
is reaching for return by extending duration, and neither is winning by holding
less rate risk.

**It survives costs and institutional financing.** Turnover is 1.5% to 2.1% a
year, and there is 90 to 252 basis points of headroom to the financing
breakeven.

**Return forecasting adds nothing on this universe.** The combined regression
and all four machine learning families finish below the index on development.
The R squared figures that made them look promising rest on three holdings whose
prices move late, and the gap disappears when the forecast horizon moves from a
day to a month.

**The technical overlays are not worth running.** Both cleared the index on
development and both finish below plain risk parity on the holdout. Momentum on
its own is worse still: on the raw twelve month definition it is the worst
strategy tested, and the risk-adjusted version loses to the index out of sample.

**The 60m rolling adaptive momentum strategy is the one open question.** It
returns more than any other strategy on development and it does not clear five
percent in any window. It is our own construction and it deserves a proper test
on data this project has not touched.

## Why risk parity and not forecasting

Skill is concentrated in the assets that carry almost none of the risk. The most
forecastable holdings are the short ones, and they account for a small share of
the universe's variance, while the thirty year Treasury and long credit dominate
it and have negative out of sample R squared. A forecast-driven portfolio
therefore faces a choice with no good branch: size positions by conviction and
the book barely moves, or size them to matter and the risk ends up dominated by
the assets the forecast cannot call.

If expected returns cannot be used where the risk is but the risk structure
itself can be estimated, the portfolio should be built from the covariance
matrix alone.

## Next steps

- **Replace funds with indices or futures.** The universe is mutual funds, which
  charge 20 to 80 basis points and carry manager decisions. High yield's daily
  returns autocorrelate at 0.24 from stale pricing. Index or futures data
  removes both.
- **Add genuinely different exposures.** Adding more US bonds reduced measured
  independence rather than raising it. TIPS, international sovereigns, emerging
  market debt and bank loans are different factors, not more of the same one.
- **Test on a second market.** If the duration and forecastability relationship
  is structural it should appear in gilts and bunds.
- **Cost aware optimisation.** On the parent project, trading a fraction of the
  way toward the target each period rather than fully rebalancing raised Sharpe
  from 0.06 to 0.23 with turnover cut ninefold.

---

## Layout

```
scripts/     numbered in reading order, phase1 through phase3 then appendices
src/macro/   the shared library: allocators, backtest engine, signals, stats
reports/     the generated writeup, one page per phase
```

Run any script from the repository root. `build_site.py` regenerates the
writeup from the stored results.

---

For research purposes only. This is not investment advice.
