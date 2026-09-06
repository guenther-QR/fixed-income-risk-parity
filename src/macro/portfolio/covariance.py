"""Covariance estimators for the walk-forward optimizer."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def sample(returns: pd.DataFrame, periods_per_year: int = 12) -> np.ndarray:
    """Plain sample covariance, annualized."""
    return returns.cov().to_numpy() * periods_per_year


def ledoit_wolf(returns: pd.DataFrame, periods_per_year: int = 12) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a constant-correlation target."""
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
    Dynamic conditional correlation covariance, evaluated at the last
    observation.
    """
    try:
        from arch import arch_model
    except ImportError:
        return ledoit_wolf(returns, periods_per_year)

    X = returns.dropna().to_numpy(dtype=float)
    t, n = X.shape
    if t < 100:
        return ledoit_wolf(returns, periods_per_year)

    # --- univariate conditional volatilities
    # ---------------------------------
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

    # --- DCC parameters by quasi-maximum likelihood
    # ---------------------------
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

    # --- roll forward one step
    # ------------------------------------------------
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
    """DCC-GARCH re-estimated on a cadence rather than at every rebalance."""

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
