"""Hull-White one-factor short-rate model: calibration and Monte Carlo.

The model is

    dr(t) = (theta(t) - a * r(t)) dt + sigma dW(t)

where a is the mean-reversion speed and sigma the short-rate volatility. theta(t)
is not free: it is chosen so the model reproduces today's observed term structure
exactly, which is what makes Hull-White an arbitrage-free extension of Vasicek
rather than a competing description of the curve.

The 2025 study asserted that "the bond bull market is dead" as commentary. This
module is what turns that into a quantity: calibrate to the current curve,
simulate forward, and report the distribution of bond returns the curve implies.

Calibration here is *historical* - a and sigma come from the realised behaviour of
the short rate. A market calibration would instead fit to cap or swaption implied
volatilities, which are not available free. That choice is a limitation and is
reported as one: historical volatility is a backward-looking estimate and will
understate the price of convexity in a market that charges a volatility premium.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HullWhiteParams:
    a: float          # mean-reversion speed
    sigma: float      # short-rate volatility, in decimal per sqrt(year)
    half_life: float  # ln(2)/a, in years - the interpretable form of a

    def __str__(self) -> str:
        return (f"a={self.a:.4f} (half-life {self.half_life:.2f}y), "
                f"sigma={self.sigma * 100:.3f}%")


def calibrate(short_rate: pd.Series, periods_per_year: float = 252.0) -> HullWhiteParams:
    """
    Estimate a and sigma from a historical short-rate series (in percent).

    Discretising the SDE over one step gives an AR(1):

        r_{t+1} - r_t = (theta - a r_t) dt + sigma sqrt(dt) e

    so regressing the change on the level identifies -a*dt as the slope, and the
    residual standard deviation identifies sigma*sqrt(dt).
    """
    r = short_rate.dropna().to_numpy(dtype=float) / 100.0
    if r.size < 100:
        raise ValueError("need at least 100 observations to calibrate")

    dt = 1.0 / periods_per_year
    dr = np.diff(r)
    lvl = r[:-1]

    slope, intercept = np.polyfit(lvl, dr, 1)
    a = -slope / dt
    resid = dr - (slope * lvl + intercept)
    sigma = resid.std(ddof=2) / np.sqrt(dt)

    a = float(max(a, 1e-4))
    return HullWhiteParams(a=a, sigma=float(sigma), half_life=float(np.log(2.0) / a))


def _forward_curve(zero_row: pd.Series):
    """Instantaneous forward f(0,t) and its derivative, from a zero curve."""
    t = zero_row.index.to_numpy(dtype=float)
    y = zero_row.to_numpy(dtype=float) / 100.0
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]

    # f(0,t) = y(t) + t * dy/dt for continuously compounded zeros.
    dydt = np.gradient(y, t)
    f = y + t * dydt

    def f_at(u):
        return np.interp(u, t, f)

    return f_at, t, f


def simulate(zero_row: pd.Series, params: HullWhiteParams, horizon: float = 5.0,
             n_paths: int = 10_000, steps_per_year: int = 12,
             seed: int | None = 42) -> dict:
    """
    Simulate short-rate paths and the zero-coupon bond prices they imply.

    Returns the simulated short rate, and P(t, t+tenor) along each path for a set
    of tenors, from which bond returns can be computed.
    """
    rng = np.random.default_rng(seed)
    a, sigma = params.a, params.sigma
    f_at, _, _ = _forward_curve(zero_row)

    n_steps = int(horizon * steps_per_year)
    dt = 1.0 / steps_per_year
    times = np.arange(n_steps + 1) * dt

    # alpha(t) = f(0,t) + sigma^2/(2a^2) * (1 - e^{-at})^2 is the deterministic
    # shift that pins the model to the observed curve.
    def alpha(t):
        return f_at(t) + (sigma ** 2) / (2 * a ** 2) * (1 - np.exp(-a * t)) ** 2

    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = f_at(0.0)

    decay = np.exp(-a * dt)
    shock_sd = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))

    for i in range(n_steps):
        t0, t1 = times[i], times[i + 1]
        drift = alpha(t1) - alpha(t0) * decay
        r[:, i + 1] = r[:, i] * decay + drift + shock_sd * rng.standard_normal(n_paths)

    return {"times": times, "short_rate": r, "params": params}


def zero_price(sim: dict, zero_row: pd.Series, step: int, tenor: float) -> np.ndarray:
    """
    P(t, t+tenor) along every path at simulation `step`, in closed form.

        P(t,T) = A(t,T) exp(-B(t,T) r(t)),  B = (1 - e^{-a(T-t)}) / a
    """
    params = sim["params"]
    a, sigma = params.a, params.sigma
    t = sim["times"][step]
    T = t + tenor
    r_t = sim["short_rate"][:, step]

    tt = zero_row.index.to_numpy(dtype=float)
    yy = zero_row.to_numpy(dtype=float) / 100.0
    ok = np.isfinite(yy)

    def p0(u: float) -> float:
        return float(np.exp(-np.interp(u, tt[ok], yy[ok]) * u))

    f_at, _, _ = _forward_curve(zero_row)
    B = (1 - np.exp(-a * tenor)) / a
    lnA = (np.log(p0(T) / p0(t)) + B * f_at(t)
           - (sigma ** 2) / (4 * a) * B ** 2 * (1 - np.exp(-2 * a * t)))
    return np.exp(lnA - B * r_t)


def bond_return_distribution(sim: dict, zero_row: pd.Series, tenor: float = 10.0,
                             horizon_years: float = 1.0,
                             steps_per_year: int = 12) -> np.ndarray:
    """
    Total return over `horizon_years` on a constant-tenor zero-coupon bond.

    Buy P(0, tenor) today; after the horizon the bond has `tenor` years left again
    (the position is rolled), so the return is the price relative plus the value
    of having held it through the period.
    """
    step = int(horizon_years * steps_per_year)
    tt = zero_row.index.to_numpy(dtype=float)
    yy = zero_row.to_numpy(dtype=float) / 100.0
    ok = np.isfinite(yy)

    p_start = float(np.exp(-np.interp(tenor, tt[ok], yy[ok]) * tenor))
    p_aged = zero_price(sim, zero_row, step, tenor - horizon_years)
    return p_aged / p_start - 1.0
