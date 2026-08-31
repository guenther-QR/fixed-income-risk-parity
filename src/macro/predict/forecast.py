"""Out-of-sample return forecasting.

The design follows Rapach, Strauss and Zhou (2010) rather than the obvious
approach, and the difference is the whole point.

The obvious approach is to regress returns on all 152 signals at once. With
roughly 380 out-of-sample months that is hopeless: the estimator fits noise, and
the resulting forecast is reliably worse than assuming the historical average.
Goyal and Welch (2008) showed that even *individually*, most published predictors
fail to beat the prevailing mean out of sample.

What survives is combination. Fit each signal on its own, in a univariate
regression, then average the resulting forecasts. Averaging cancels the
estimation error that any single regression carries while retaining whatever
common signal they share, and it consistently beats both the kitchen sink and the
best individual predictor chosen ex post.

Two further disciplines, both from the same literature:

    The prevailing mean is the benchmark. Not zero, not the risk-free rate: the
    expanding-window historical average. A forecast that cannot beat it has no
    value, and reporting against a weaker benchmark hides that.

    Campbell and Thompson (2008) constraints. Clip forecasts to economically
    sensible ranges and restrict coefficient signs to theory. These are free
    restrictions that reliably improve out-of-sample accuracy, because they
    remove exactly the estimates that estimation error produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prevailing_mean(y: pd.Series, min_obs: int = 60) -> pd.Series:
    """
    Expanding-window historical average - the benchmark every forecast must beat.

    The value at t uses returns strictly before t, so it is a genuine forecast
    rather than a description.
    """
    return y.shift(1).expanding(min_periods=min_obs).mean()


def univariate_forecasts(y: pd.Series, X: pd.DataFrame, min_obs: int = 60,
                         horizon: int = 1) -> pd.DataFrame:
    """
    One expanding-window univariate forecast per signal.

    **Convention: the output is indexed by the period it predicts.** `forecast[t]`
    is a prediction of `y[t]`, formed from `X[t]` - which the signal library has
    already lagged so it is knowable before t - and from coefficients estimated
    only on pairs whose outcomes were realised by `t - horizon`.

    That indexing is stated so plainly because getting it wrong is silent and
    catastrophic. An earlier version returned forecasts indexed by the period the
    coefficients were estimated *through*, so the fit for period t had already
    seen y[t] before being scored against it. On pure noise with a junk signal it
    produced an out-of-sample R-squared of +2.0%, which looks like skill and is
    arithmetic.

    Computed from running sums rather than by re-fitting: an expanding OLS slope
    has a closed form in cumulative moments, which turns 152 signals across 500
    months into a few vectorized passes.
    """
    out = {}

    for col in X.columns:
        pair = pd.concat([X[col].rename("x"), y.rename("y")], axis=1).dropna()
        if len(pair) < min_obs + horizon:
            continue

        xs, ys = pair["x"], pair["y"]
        n = np.arange(1, len(pair) + 1)
        sx, sy = xs.cumsum(), ys.cumsum()
        sxx, sxy = (xs * xs).cumsum(), (xs * ys).cumsum()

        var = sxx - sx ** 2 / n
        cov = sxy - sx * sy / n
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(np.abs(var) > 1e-12, cov / var, 0.0)
        alpha = sy / n - beta * sx / n

        coef = pd.DataFrame({"alpha": alpha, "beta": beta}, index=pair.index)
        coef[n < min_obs] = np.nan

        # Shift by the horizon so the coefficients applied at t were estimated
        # only from outcomes already observed. At horizon 1 this means the fit
        # through t-1; at horizon 12, the fit through t-12.
        coef = coef.shift(horizon).reindex(X.index).ffill()
        out[col] = coef["alpha"] + coef["beta"] * X[col]

    return pd.DataFrame(out, index=X.index)


def combine(forecasts: pd.DataFrame, method: str = "mean",
            trim: float = 0.10) -> pd.Series:
    """
    Pool individual forecasts into one.

    The simple mean is the workhorse and is hard to beat. The trimmed mean drops
    the most extreme forecasts at each date, which guards against a single
    misbehaving signal without needing to identify it in advance.
    """
    if method == "mean":
        return forecasts.mean(axis=1)
    if method == "median":
        return forecasts.median(axis=1)
    if method == "trimmed":
        lo = forecasts.quantile(trim, axis=1)
        hi = forecasts.quantile(1 - trim, axis=1)
        masked = forecasts.where(forecasts.ge(lo, axis=0)
                                 & forecasts.le(hi, axis=0))
        return masked.mean(axis=1)
    raise ValueError(f"unknown combination method: {method}")


def campbell_thompson(forecast: pd.Series, benchmark: pd.Series,
                      floor: float | None = 0.0,
                      max_deviation: float | None = None) -> pd.Series:
    """
    Campbell-Thompson (2008) restrictions on a forecast.

    `floor` clips the forecast from below - zero for an equity risk premium,
    since a rational investor would not forecast a negative one. `max_deviation`
    bounds how far the forecast may stray from the prevailing mean, which caps
    the damage a badly estimated coefficient can do.

    Both are restrictions imposed by economics rather than fitted from data, so
    they cost no degrees of freedom and cannot overfit.
    """
    out = forecast.copy()
    if max_deviation is not None:
        lo = benchmark - max_deviation
        hi = benchmark + max_deviation
        out = out.clip(lower=lo, upper=hi)
    if floor is not None:
        out = out.clip(lower=floor)
    return out


# ----------------------------------------------------------------- evaluation

def oos_r2(actual: pd.Series, forecast: pd.Series,
           benchmark: pd.Series) -> float:
    """
    Campbell-Thompson out-of-sample R-squared.

        R2_OS = 1 - MSE(forecast) / MSE(benchmark)

    Negative means the forecast is worse than assuming the historical average,
    which Goyal and Welch found to be the normal case. Values above about 0.5%
    monthly are economically meaningful despite looking tiny.
    """
    d = pd.concat([actual.rename("a"), forecast.rename("f"),
                   benchmark.rename("b")], axis=1).dropna()
    if len(d) < 12:
        return np.nan
    mse_f = ((d["a"] - d["f"]) ** 2).mean()
    mse_b = ((d["a"] - d["b"]) ** 2).mean()
    return float(1 - mse_f / mse_b) if mse_b > 0 else np.nan


def clark_west(actual: pd.Series, forecast: pd.Series,
               benchmark: pd.Series) -> tuple[float, float]:
    """
    Clark-West (2007) test for nested forecast comparison.

    Comparing a model against the prevailing mean is a *nested* comparison, and
    the standard Diebold-Mariano test is biased against the larger model there:
    even under the null, the extra estimated parameters add noise that inflates
    its MSE. Clark-West adjusts for exactly that, so the test asks whether the
    model helps in population rather than whether it helped in this sample.

    Returns (statistic, one-sided p-value).
    """
    d = pd.concat([actual.rename("a"), forecast.rename("f"),
                   benchmark.rename("b")], axis=1).dropna()
    if len(d) < 24:
        return np.nan, np.nan

    e_b = (d["a"] - d["b"]) ** 2
    e_f = (d["a"] - d["f"]) ** 2
    adj = (d["b"] - d["f"]) ** 2
    f_hat = e_b - e_f + adj

    n = len(f_hat)
    mean = f_hat.mean()

    # Newey-West standard error; forecasts overlap, so serial correlation is real.
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    resid = f_hat - mean
    gamma0 = float((resid ** 2).mean())
    var = gamma0
    for k in range(1, max(lag, 1) + 1):
        cov = float((resid.iloc[k:] * resid.iloc[:-k].to_numpy()).mean())
        var += 2 * (1 - k / (lag + 1)) * cov
    se = np.sqrt(max(var, 1e-18) / n)

    stat = mean / se if se > 0 else np.nan
    from scipy import stats as _st
    return float(stat), float(1 - _st.norm.cdf(stat))


def certainty_equivalent(returns: pd.Series, rf: pd.Series,
                         gamma: float = 5.0,
                         periods_per_year: int = 12) -> float:
    """
    Certainty-equivalent return for a mean-variance investor.

        CE = mu - (gamma / 2) * sigma^2

    The economic counterpart to statistical accuracy. A forecast can be
    statistically significant and worthless, or marginally significant and
    valuable; the certainty equivalent measures which, in units an investor
    recognises.
    """
    ex = (returns - rf.reindex(returns.index)).dropna()
    mu = ex.mean() * periods_per_year
    var = ex.var() * periods_per_year
    return float(mu - 0.5 * gamma * var)


def ce_gain(strategy: pd.Series, benchmark: pd.Series, rf: pd.Series,
            gamma: float = 5.0, periods_per_year: int = 12) -> float:
    """Annualized certainty-equivalent gain of a strategy over a benchmark."""
    common = strategy.index.intersection(benchmark.index)
    return (certainty_equivalent(strategy.loc[common], rf, gamma, periods_per_year)
            - certainty_equivalent(benchmark.loc[common], rf, gamma, periods_per_year))
