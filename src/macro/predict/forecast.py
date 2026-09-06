"""Out-of-sample return forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prevailing_mean(y: pd.Series, min_obs: int = 60) -> pd.Series:
    """
    Expanding-window historical average - the benchmark every forecast must
    beat.
    """
    return y.shift(1).expanding(min_periods=min_obs).mean()


def univariate_forecasts(y: pd.Series, X: pd.DataFrame, min_obs: int = 60,
                         horizon: int = 1) -> pd.DataFrame:
    """One expanding-window univariate forecast per signal."""
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
        # only from outcomes already observed.
        coef = coef.shift(horizon).reindex(X.index).ffill()
        out[col] = coef["alpha"] + coef["beta"] * X[col]

    return pd.DataFrame(out, index=X.index)


def combine(forecasts: pd.DataFrame, method: str = "mean",
            trim: float = 0.10) -> pd.Series:
    """Pool individual forecasts into one."""
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
    """Campbell-Thompson (2008) restrictions on a forecast."""
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
    """Campbell-Thompson out-of-sample R-squared."""
    d = pd.concat([actual.rename("a"), forecast.rename("f"),
                   benchmark.rename("b")], axis=1).dropna()
    if len(d) < 12:
        return np.nan
    mse_f = ((d["a"] - d["f"]) ** 2).mean()
    mse_b = ((d["a"] - d["b"]) ** 2).mean()
    return float(1 - mse_f / mse_b) if mse_b > 0 else np.nan


def clark_west(actual: pd.Series, forecast: pd.Series,
               benchmark: pd.Series) -> tuple[float, float]:
    """Clark-West (2007) test for nested forecast comparison."""
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

    # Newey-West standard error; forecasts overlap, so serial correlation is
    # real.
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
    """Certainty-equivalent return for a mean-variance investor."""
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
