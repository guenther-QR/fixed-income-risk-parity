"""G2++: the two-factor additive Gaussian short-rate model.

Hull-White moves the entire curve off a single shock, so it cannot generate a
twist - every simulated scenario is a parallel-ish shift. The curve PCA shows
slope and curvature account for over 10% of daily variation, so that is a real
gap in any scenario set built from one factor.

G2++ writes the short rate as the sum of two correlated mean-reverting factors
plus a deterministic shift that pins the model to today's curve:

    r(t) = x(t) + y(t) + phi(t)
    dx = -a x dt + sigma dW1
    dy = -b y dt + eta   dW2        with  corr(dW1, dW2) = rho

Because the two factors revert at different speeds, a shock to the fast factor
decays out of the long end before the short end, which is exactly a steepening.
G2++ is equivalent to two-factor Hull-White and admits closed-form bond prices.

One thing this does *not* change: G2++ is still a risk-neutral model, so the
expected return on every bond remains the short rate. It improves the realism of
simulated risk, not the forecasting of returns. Return forecasting is the job of
`atsm.py`, which estimates prices of risk under the physical measure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class G2Params:
    a: float       # fast factor mean reversion
    b: float       # slow factor mean reversion
    sigma: float   # fast factor volatility
    eta: float     # slow factor volatility
    rho: float     # correlation between the two shocks

    def __str__(self) -> str:
        return (f"a={self.a:.4f} (half-life {np.log(2)/self.a:5.2f}y), "
                f"b={self.b:.4f} (half-life {np.log(2)/self.b:5.2f}y), "
                f"sigma={self.sigma*100:.3f}%, eta={self.eta*100:.3f}%, "
                f"rho={self.rho:+.3f}")


def yield_loading(decay: float, tau: np.ndarray) -> np.ndarray:
    """
    How a unit shock to a factor with the given decay moves the yield at tau.

    B(tau)/tau where B(tau) = (1 - e^{-z tau})/z. The loading is 1 at the very
    short end and decays toward zero as maturity lengthens, so a fast-decaying
    factor moves short yields much more than long ones.
    """
    tau = np.asarray(tau, dtype=float)
    z = max(decay, 1e-8)
    return (1.0 - np.exp(-z * tau)) / (z * tau)


def model_covariance(p: G2Params, tau: np.ndarray, dt: float) -> np.ndarray:
    """Model-implied covariance of yield changes across maturities, per period."""
    la = yield_loading(p.a, tau)
    lb = yield_loading(p.b, tau)
    return dt * (
        p.sigma ** 2 * np.outer(la, la)
        + p.eta ** 2 * np.outer(lb, lb)
        + p.rho * p.sigma * p.eta * (np.outer(la, lb) + np.outer(lb, la))
    )


def calibrate(zero: pd.DataFrame, maturities=(0.25, 1.0, 2.0, 5.0, 10.0, 30.0),
              periods_per_year: float = 252.0) -> G2Params:
    """
    Fit the five parameters to the empirical covariance of daily yield changes.

    Matching the whole covariance matrix - not just the volatility at each
    maturity - is what identifies the correlation and the two decay speeds. A
    one-factor model can match the diagonal but cannot reproduce the way
    correlations fall off between short and long maturities.
    """
    cols = [min(zero.columns, key=lambda c: abs(float(c) - m)) for m in maturities]
    tau = np.array([float(c) for c in cols])
    changes = zero[cols].diff().dropna() / 100.0
    target = changes.cov().to_numpy()
    dt = 1.0 / periods_per_year

    iu = np.triu_indices(len(tau))

    def residual(theta: np.ndarray) -> np.ndarray:
        a, b, sigma, eta, rho = theta
        p = G2Params(abs(a), abs(b), abs(sigma), abs(eta), np.clip(rho, -0.999, 0.999))
        model = model_covariance(p, tau, dt)
        # Scale by 1e4 so the optimiser works on numbers near unity.
        return (model[iu] - target[iu]) * 1e4

    best = None
    for start in [(0.6, 0.05, 0.010, 0.008, -0.7),
                  (1.2, 0.02, 0.015, 0.006, -0.5),
                  (0.3, 0.10, 0.008, 0.010, -0.9)]:
        res = least_squares(residual, np.array(start), method="lm", max_nfev=8000)
        if best is None or res.cost < best.cost:
            best = res

    a, b, sigma, eta, rho = best.x
    a, b, sigma, eta = abs(a), abs(b), abs(sigma), abs(eta)
    rho = float(np.clip(rho, -0.999, 0.999))
    if a < b:                      # keep `a` the fast factor for interpretability
        a, b, sigma, eta = b, a, eta, sigma
    return G2Params(a=float(a), b=float(b), sigma=float(sigma),
                    eta=float(eta), rho=rho)


def variance_term(p: G2Params, t: float, T: float) -> float:
    """V(t,T) from Brigo & Mercurio, the integrated variance of the log bond price."""
    a, b, s, e, rho = p.a, p.b, p.sigma, p.eta, p.rho
    dT = T - t
    va = (s ** 2 / a ** 2) * (dT + (2 / a) * np.exp(-a * dT)
                              - (1 / (2 * a)) * np.exp(-2 * a * dT) - 3 / (2 * a))
    vb = (e ** 2 / b ** 2) * (dT + (2 / b) * np.exp(-b * dT)
                              - (1 / (2 * b)) * np.exp(-2 * b * dT) - 3 / (2 * b))
    vc = (2 * rho * s * e / (a * b)) * (
        dT + (np.exp(-a * dT) - 1) / a + (np.exp(-b * dT) - 1) / b
        - (np.exp(-(a + b) * dT) - 1) / (a + b))
    return float(va + vb + vc)


def simulate(zero_row: pd.Series, p: G2Params, horizon: float = 5.0,
             n_paths: int = 10_000, steps_per_year: int = 12,
             seed: int | None = 42) -> dict:
    """Simulate the two factors and the short rate they imply."""
    rng = np.random.default_rng(seed)
    n_steps = int(horizon * steps_per_year)
    dt = 1.0 / steps_per_year
    times = np.arange(n_steps + 1) * dt

    # Exact one-step moments of the OU pair.
    sd_x = p.sigma * np.sqrt((1 - np.exp(-2 * p.a * dt)) / (2 * p.a))
    sd_y = p.eta * np.sqrt((1 - np.exp(-2 * p.b * dt)) / (2 * p.b))
    chol = np.array([[1.0, 0.0], [p.rho, np.sqrt(max(1 - p.rho ** 2, 0.0))]])

    x = np.zeros((n_paths, n_steps + 1))
    y = np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):
        z = rng.standard_normal((n_paths, 2)) @ chol.T
        x[:, i + 1] = x[:, i] * np.exp(-p.a * dt) + sd_x * z[:, 0]
        y[:, i + 1] = y[:, i] * np.exp(-p.b * dt) + sd_y * z[:, 1]

    tt = zero_row.index.to_numpy(dtype=float)
    yy = zero_row.to_numpy(dtype=float) / 100.0
    ok = np.isfinite(yy)
    f0 = np.interp(times, tt[ok], yy[ok])            # phi(t) anchored to the curve
    short = x + y + f0[None, :]

    return {"times": times, "x": x, "y": y, "short_rate": short,
            "params": p, "zero_row": zero_row}


def simulated_yield(sim: dict, step: int, tenor: float) -> np.ndarray:
    """
    The zero yield at `tenor` on every simulated path, at simulation `step`.

    Uses the affine form: the yield is the forward curve today plus each factor
    times its loading, so different loadings at different tenors produce twists.
    """
    p = sim["params"]
    t = sim["times"][step]
    zr = sim["zero_row"]
    tt = zr.index.to_numpy(dtype=float)
    yy = zr.to_numpy(dtype=float) / 100.0
    ok = np.isfinite(yy)

    def zero_at(u: float) -> float:
        return float(np.interp(u, tt[ok], yy[ok]))

    # Forward-starting base yield implied by today's curve.
    T = t + tenor
    base = (zero_at(T) * T - zero_at(t) * t) / tenor if t > 0 else zero_at(tenor)

    la = float(yield_loading(p.a, np.array([tenor]))[0])
    lb = float(yield_loading(p.b, np.array([tenor]))[0])
    conv = 0.5 * variance_term(p, t, T) / tenor
    return base + la * sim["x"][:, step] + lb * sim["y"][:, step] - conv


def slope_distribution(sim: dict, step: int, short_tenor: float = 2.0,
                       long_tenor: float = 10.0) -> np.ndarray:
    """The 2s10s slope across paths - the quantity a one-factor model cannot move."""
    return (simulated_yield(sim, step, long_tenor)
            - simulated_yield(sim, step, short_tenor))
