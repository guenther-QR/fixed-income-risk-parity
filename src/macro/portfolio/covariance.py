"""Covariance estimators for the walk-forward optimizer.

The sample covariance matrix is a poor input to mean-variance optimization. With
N assets it estimates N(N+1)/2 parameters, its extreme eigenvalues are biased
outward, and the optimizer then concentrates weight precisely on the directions
where that error is largest. The 2025 study used it unshrunk on a 60-month window.

Three estimators, in increasing order of what they assume:

    sample        no structure imposed; the baseline
    ledoit_wolf   shrunk toward constant correlation, with the optimal intensity
                  chosen analytically rather than tuned
    dcc_garch     conditional covariance that varies through time, so the
                  estimate reflects today's volatility regime rather than the
                  window average

The Phase 3 rolling correlations are the argument for the third: the stock-bond
correlation changed sign twice over the sample, and any single window average
sits somewhere in between, describing neither regime.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def sample(returns: pd.DataFrame, periods_per_year: int = 12) -> np.ndarray:
    """Plain sample covariance, annualized."""
    return returns.cov().to_numpy() * periods_per_year


def ledoit_wolf(returns: pd.DataFrame, periods_per_year: int = 12) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage toward a constant-correlation target.

    The target keeps each asset's own variance but replaces every pairwise
    correlation with their average. Shrinkage intensity is the analytic optimum
    from Ledoit and Wolf (2004): it trades the sample matrix's low bias against
    the target's low variance, and needs no tuning parameter.
    """
    X = returns.to_numpy(dtype=float)
    t, n = X.shape
    if t < 2:
        raise ValueError("need at least two observations")

    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / t

    var = np.diag(S)
    sd = np.sqrt(var)
    outer_sd = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = S / outer_sd
    np.fill_diagonal(corr, 1.0)

    # Average off-diagonal correlation defines the target.
    off = ~np.eye(n, dtype=bool)
    r_bar = np.nanmean(corr[off]) if n > 1 else 0.0
    F = r_bar * outer_sd
    np.fill_diagonal(F, var)

    # pi: sum of asymptotic variances of the sample covariance entries.
    Y = Xc ** 2
    pi_mat = (Y.T @ Y) / t - S ** 2
    pi_hat = pi_mat.sum()

    # rho: covariance between the sample entries and the target's entries.
    term = ((Xc ** 3).T @ Xc) / t - var[:, None] * S
    np.fill_diagonal(term, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(outer_sd > 0, np.sqrt(np.outer(var, 1.0 / var)), 0.0)
    rho_hat = np.diag(pi_mat).sum() + r_bar * (ratio * term).sum()

    # gamma: squared distance between the sample matrix and the target.
    gamma_hat = float(np.sum((F - S) ** 2))

    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    delta = float(np.clip(kappa / t, 0.0, 1.0))

    return (delta * F + (1 - delta) * S) * periods_per_year


def ledoit_wolf_intensity(returns: pd.DataFrame) -> float:
    """The shrinkage intensity alone, for reporting how much structure was imposed."""
    X = returns.to_numpy(dtype=float)
    t = X.shape[0]
    lw = ledoit_wolf(returns, periods_per_year=1)
    s = sample(returns, periods_per_year=1)
    denom = np.sum((lw - s) ** 2)
    return float(np.sqrt(denom) / (np.sqrt(np.sum(s ** 2)) + 1e-12)) if t > 1 else 0.0


def dcc_garch(returns: pd.DataFrame, periods_per_year: int = 12,
              refit_univariate: bool = True) -> np.ndarray:
    """
    Dynamic conditional correlation covariance, evaluated at the last observation.

    Engle (2002). Univariate GARCH(1,1) supplies each asset's conditional
    volatility; the standardized residuals then drive a correlation process

        Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}
        R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2

    and the covariance is D_t R_t D_t. Returns the conditional covariance for the
    period *following* the sample, which is what a portfolio formed today needs.

    Falls back to Ledoit-Wolf if the GARCH fits or the DCC optimization fail;
    a backtest should degrade rather than stop.
    """
    try:
        from arch import arch_model
    except ImportError:
        return ledoit_wolf(returns, periods_per_year)

    X = returns.dropna().to_numpy(dtype=float)
    t, n = X.shape
    if t < 100:
        return ledoit_wolf(returns, periods_per_year)

    # --- univariate conditional volatilities ---------------------------------
    sigma = np.empty_like(X)
    forecast_sd = np.empty(n)
    scale = 100.0                      # arch prefers percent-scale data
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(n):
                am = arch_model(X[:, j] * scale, vol="GARCH", p=1, q=1,
                                mean="Constant", dist="normal")
                res = am.fit(disp="off", show_warning=False)
                sigma[:, j] = np.asarray(res.conditional_volatility) / scale
                f = res.forecast(horizon=1, reindex=False)
                forecast_sd[j] = np.sqrt(float(f.variance.values[-1, 0])) / scale
    except Exception:
        return ledoit_wolf(returns, periods_per_year)

    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        return ledoit_wolf(returns, periods_per_year)

    z = X / sigma
    Qbar = np.cov(z, rowvar=False, ddof=1)

    # --- DCC parameters by quasi-maximum likelihood ---------------------------
    def nll(theta: np.ndarray) -> float:
        a, b = theta
        if a <= 0 or b <= 0 or a + b >= 0.999:
            return 1e10
        Q = Qbar.copy()
        total = 0.0
        for i in range(t):
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            try:
                sign, logdet = np.linalg.slogdet(R)
                if sign <= 0:
                    return 1e10
                total += logdet + z[i] @ np.linalg.solve(R, z[i])
            except np.linalg.LinAlgError:
                return 1e10
            Q = (1 - a - b) * Qbar + a * np.outer(z[i], z[i]) + b * Q
        return 0.5 * total

    from scipy.optimize import minimize
    best = None
    for start in [(0.02, 0.95), (0.05, 0.90), (0.01, 0.97)]:
        try:
            r = minimize(nll, np.array(start), method="Nelder-Mead",
                         options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue
    if best is None or not np.isfinite(best.fun) or best.fun >= 1e9:
        return ledoit_wolf(returns, periods_per_year)

    a, b = float(best.x[0]), float(best.x[1])
    if not (0 < a < 1 and 0 < b < 1 and a + b < 0.999):
        return ledoit_wolf(returns, periods_per_year)

    # --- roll forward one step ------------------------------------------------
    Q = Qbar.copy()
    for i in range(t):
        Q = (1 - a - b) * Qbar + a * np.outer(z[i], z[i]) + b * Q
    d = np.sqrt(np.diag(Q))
    R = Q / np.outer(d, d)
    D = np.diag(forecast_sd)
    H = D @ R @ D

    H = 0.5 * (H + H.T)
    if np.linalg.eigvalsh(H).min() <= 0:
        return ledoit_wolf(returns, periods_per_year)
    return H * periods_per_year


ESTIMATORS = {"sample": sample, "ledoit_wolf": ledoit_wolf, "dcc_garch": dcc_garch}


class CachedDCC:
    """
    DCC-GARCH re-estimated on a cadence rather than at every rebalance.

    A full refit costs roughly 1.8 seconds - eight univariate GARCH fits plus a
    DCC likelihood optimization - which is about fifteen minutes over a 500-month
    walk-forward for a single strategy. Re-estimating a covariance model annually
    and carrying it between refits is both standard practice and what makes the
    comparison against the other estimators affordable.

    The cadence is a modelling choice with a cost: the estimate is stale by up to
    `refit_every` periods, so this understates how quickly DCC adapts to a
    volatility shock. `refit_every=1` recovers the exact estimator.
    """

    def __init__(self, refit_every: int = 12, periods_per_year: int = 12):
        self.refit_every = refit_every
        self.periods_per_year = periods_per_year
        self._cache: np.ndarray | None = None
        self._fitted_at: int = -10 ** 9
        self.refits = 0

    def __call__(self, returns: pd.DataFrame) -> np.ndarray:
        n = len(returns)
        if self._cache is None or n - self._fitted_at >= self.refit_every:
            self._cache = dcc_garch(returns, self.periods_per_year)
            self._fitted_at = n
            self.refits += 1
        return self._cache
