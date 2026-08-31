"""Nelson-Siegel-Svensson curve fitting.

The Svensson (1994) extension of Nelson-Siegel (1987) writes the continuously
compounded zero yield at maturity m as

    y(m) = b0
         + b1 * (1 - e^-x1) / x1
         + b2 * ((1 - e^-x1) / x1 - e^-x1)
         + b3 * ((1 - e^-x2) / x2 - e^-x2)

with x1 = m/t1 and x2 = m/t2. The four beta terms carry economic meaning: b0 is
the long-run level, b1 the short-end deviation from it (so b0 + b1 is the
instantaneous short rate), and b2, b3 two humps at horizons set by t1 and t2.

The betas enter linearly once t1 and t2 are fixed, so fitting is a small nonlinear
search over the two decay parameters with linear least squares nested inside. That
is both faster and far more reliable than optimizing all six jointly, which is
notoriously prone to local minima.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

PARAM_NAMES = ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]


def _basis(m: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    """Design matrix whose columns are the four Svensson factor loadings."""
    m = np.asarray(m, dtype=float)
    x1 = np.maximum(m / tau1, 1e-12)
    x2 = np.maximum(m / tau2, 1e-12)
    e1, e2 = np.exp(-x1), np.exp(-x2)
    slope = (1.0 - e1) / x1
    return np.column_stack([
        np.ones_like(m),          # level
        slope,                    # slope
        slope - e1,               # first hump
        (1.0 - e2) / x2 - e2,     # second hump
    ])


def svensson(m, beta0, beta1, beta2, beta3, tau1, tau2):
    """Zero yield(s) at maturity `m` for the given parameters."""
    scalar = np.isscalar(m)
    out = _basis(np.atleast_1d(m), tau1, tau2) @ np.array([beta0, beta1, beta2, beta3])
    return float(out[0]) if scalar else out


# Economic bounds on the betas, in percent. Without them the fit is badly
# identified: beta0 is the infinite-maturity asymptote, but the data stop at 30
# years, so wildly offsetting betas can fit the observed range while implying
# nonsense asymptotics. An unbounded fit on this data produced beta0 between -224
# and +33 and beta3 up to 577, with a perfectly good 3bp RMSE.
BETA_BOUNDS = (
    np.array([0.0, -20.0, -30.0, -30.0]),      # lower
    np.array([20.0, 20.0, 30.0, 30.0]),        # upper
)


def _solve_betas(m: np.ndarray, y: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """Bounded linear least squares for the betas, given the decay parameters."""
    return lsq_linear(_basis(m, t1, t2), y, bounds=BETA_BOUNDS, method="bvls").x


def fit_one(maturities: np.ndarray, yields: np.ndarray,
            tau_start: tuple[float, float] = (1.5, 12.0)) -> dict:
    """
    Fit Svensson to one day's curve.

    Returns the six parameters plus the fit's RMSE in the same units as `yields`.

    Two constraints make the parameters comparable across days, which matters
    because they are used as time series in later phases:

      The betas are bounded to economically sensible ranges (see BETA_BOUNDS).

      tau1 < tau2 is imposed. The two hump terms are otherwise exchangeable -
      swapping (beta2, tau1) with (beta3, tau2) leaves the curve unchanged - so
      without an ordering the parameters are identified only up to a permutation
      and jump between days for no economic reason.
    """
    m = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    ok = np.isfinite(m) & np.isfinite(y)
    m, y = m[ok], y[ok]
    if m.size < 6:
        raise ValueError("need at least six points to identify six parameters")

    def sse(taus: np.ndarray) -> float:
        t1, t2 = taus
        if not (0.1 <= t1 <= 10.0 and 1.0 <= t2 <= 30.0) or t2 - t1 < 0.5:
            return 1e12
        fitted = _basis(m, t1, t2) @ _solve_betas(m, y, t1, t2)
        return float(np.sum((y - fitted) ** 2))

    best = None
    # A few starts, because the tau surface is multimodal even in two dimensions.
    for start in [tau_start, (0.5, 5.0), (2.5, 20.0), (1.0, 8.0)]:
        res = minimize(sse, np.array(start), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 800})
        if best is None or res.fun < best.fun:
            best = res

    tau1, tau2 = float(best.x[0]), float(best.x[1])
    betas = _solve_betas(m, y, tau1, tau2)
    rmse = float(np.sqrt(best.fun / m.size))

    return dict(zip(PARAM_NAMES, [*betas, tau1, tau2])) | {"rmse": rmse}


def fit_panel(zero: pd.DataFrame, every: int = 1) -> pd.DataFrame:
    """Fit Svensson to each row of a zero-curve panel (columns = maturities)."""
    mats = zero.columns.to_numpy(dtype=float)
    rows, index = [], []
    for date, row in zero.iloc[::every].iterrows():
        try:
            rows.append(fit_one(mats, row.to_numpy(dtype=float)))
            index.append(date)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="date"))


def short_rate(params: pd.DataFrame) -> pd.Series:
    """The instantaneous short rate implied by a fit: the m -> 0 limit, b0 + b1."""
    return params["beta0"] + params["beta1"]
