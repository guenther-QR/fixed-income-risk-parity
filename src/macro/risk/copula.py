"""
Copula models of joint asset behaviour, and what they say about the left
tail.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


@dataclass
class CopulaFit:
    kind: str                  # "gaussian" or "student_t"
    corr: np.ndarray           # dependence matrix
    df: float | None           # degrees of freedom (t only)
    loglik: float
    n_params: int
    assets: list[str]

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    def __str__(self) -> str:
        nu = f", nu={self.df:.1f}" if self.df is not None else ""
        return f"{self.kind}{nu}: loglik={self.loglik:.1f}, AIC={self.aic:.1f}"


# -------------------------------------------------------------------
# marginals

def garch_residuals(returns: pd.DataFrame, dist: str = "skewt") -> pd.DataFrame:
    """Standardized residuals from a univariate GARCH(1,1) per asset."""
    try:
        from arch import arch_model
    except ImportError:
        return returns.copy()

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in returns.columns:
            s = returns[col].dropna()
            try:
                res = arch_model(s * 100, vol="GARCH", p=1, q=1,
                                 mean="Constant", dist=dist).fit(disp="off",
                                                                 show_warning=False)
                z = np.asarray(res.resid) / np.asarray(res.conditional_volatility)
                out[col] = pd.Series(z, index=s.index)
            except Exception:
                out[col] = (s - s.mean()) / s.std()
    return pd.DataFrame(out).dropna()


def to_uniform(residuals: pd.DataFrame) -> pd.DataFrame:
    """Probability integral transform by empirical rank."""
    n = len(residuals)
    return residuals.rank(method="average") / (n + 1.0)


# ---------------------------------------------------------------------
# fitting

def _corr_from_uniform(u: np.ndarray, quantile_fn) -> np.ndarray:
    x = quantile_fn(u)
    c = np.corrcoef(x, rowvar=False)
    return _nearest_psd(c)


def _nearest_psd(c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, eps, None)
    out = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def fit_gaussian(u: pd.DataFrame) -> CopulaFit:
    """Gaussian copula. Tail dependence is zero by construction for rho < 1."""
    arr = u.to_numpy(dtype=float)
    R = _corr_from_uniform(arr, stats.norm.ppf)
    x = stats.norm.ppf(arr)

    sign, logdet = np.linalg.slogdet(R)
    Rinv = np.linalg.inv(R)
    quad = np.einsum("ij,jk,ik->i", x, Rinv - np.eye(len(R)), x)
    loglik = float(-0.5 * len(arr) * logdet - 0.5 * quad.sum())

    n = len(R)
    return CopulaFit("gaussian", R, None, loglik, n * (n - 1) // 2, list(u.columns))


def _t_loglik(u: np.ndarray, R: np.ndarray, nu: float) -> float:
    d = R.shape[0]
    x = stats.t.ppf(u, df=nu)
    sign, logdet = np.linalg.slogdet(R)
    Rinv = np.linalg.inv(R)
    quad = np.einsum("ij,jk,ik->i", x, Rinv, x)

    const = (gammaln((nu + d) / 2) + (d - 1) * gammaln(nu / 2)
             - d * gammaln((nu + 1) / 2))
    joint = const - 0.5 * logdet - (nu + d) / 2 * np.log1p(quad / nu)
    marginal = np.sum(-(nu + 1) / 2 * np.log1p(x ** 2 / nu), axis=1)
    return float(np.sum(joint - marginal))


def fit_student_t(u: pd.DataFrame, nu_bounds=(2.1, 100.0)) -> CopulaFit:
    """
    Student-t copula, with the degrees of freedom fitted by profile
    likelihood.
    """
    arr = u.to_numpy(dtype=float)

    def neg(nu: float) -> float:
        R = _corr_from_uniform(arr, lambda p: stats.t.ppf(p, df=nu))
        return -_t_loglik(arr, R, nu)

    res = minimize_scalar(neg, bounds=nu_bounds, method="bounded",
                          options={"xatol": 1e-3})
    nu = float(res.x)
    R = _corr_from_uniform(arr, lambda p: stats.t.ppf(p, df=nu))

    n = len(R)
    return CopulaFit("student_t", R, nu, float(-res.fun),
                     n * (n - 1) // 2 + 1, list(u.columns))


# ------------------------------------------------------------ tail dependence

def t_tail_dependence(rho: float, nu: float) -> float:
    """Coefficient of lower tail dependence implied by a bivariate t copula."""
    rho = float(np.clip(rho, -0.999, 0.999))
    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * stats.t.cdf(arg, df=nu + 1))


def model_tail_matrix(fit: CopulaFit) -> pd.DataFrame:
    """Pairwise model-implied lower tail dependence."""
    n = len(fit.assets)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i, j] = 1.0
            elif fit.df is None:
                out[i, j] = 0.0          # Gaussian
            else:
                out[i, j] = t_tail_dependence(fit.corr[i, j], fit.df)
    return pd.DataFrame(out, index=fit.assets, columns=fit.assets)


def empirical_tail_dependence(u: pd.DataFrame, q: float = 0.10,
                              excess: bool = False) -> pd.DataFrame:
    """Non-parametric lower tail dependence at threshold `q`."""
    cols = list(u.columns)
    arr = u.to_numpy(dtype=float)
    n = len(cols)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            both = np.mean((arr[:, i] < q) & (arr[:, j] < q))
            out[i, j] = out[j, i] = both / q
    frame = pd.DataFrame(out, index=cols, columns=cols)
    if excess:
        frame = frame - q
        np.fill_diagonal(frame.values, 1.0 - q)
    return frame


def gaussian_exceedance_benchmark(rho: float, q: float = 0.10,
                                  n_sim: int = 200_000,
                                  seed: int = 0) -> float:
    """Exceedance correlation a *Gaussian* pair with this rho would show."""
    rho = float(np.clip(rho, -0.999, 0.999))
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((n_sim, 2)) @ L.T
    a, b = z[:, 0], z[:, 1]
    mask = (a < np.quantile(a, q)) & (b < np.quantile(b, q))
    if mask.sum() < 50:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def exceedance_counts(returns: pd.DataFrame, q: float = 0.10) -> pd.DataFrame:
    """Number of periods where both assets breached their q-quantile."""
    cols = list(returns.columns)
    n = len(cols)
    out = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            a, b = returns[cols[i]], returns[cols[j]]
            out[i, j] = int(((a < a.quantile(q)) & (b < b.quantile(q))).sum())
    return pd.DataFrame(out, index=cols, columns=cols)


def exceedance_correlation(returns: pd.DataFrame, q: float = 0.10,
                           side: str = "lower", min_obs: int = 20) -> pd.DataFrame:
    """Correlation computed only over jointly extreme periods."""
    cols = list(returns.columns)
    n = len(cols)
    out = np.full((n, n), np.nan)
    np.fill_diagonal(out, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = returns[cols[i]], returns[cols[j]]
            if side == "lower":
                mask = (a < a.quantile(q)) & (b < b.quantile(q))
            else:
                mask = (a > a.quantile(1 - q)) & (b > b.quantile(1 - q))
            if mask.sum() >= min_obs:
                out[i, j] = out[j, i] = a[mask].corr(b[mask])
    return pd.DataFrame(out, index=cols, columns=cols)


# ---------------------------------------------------------------- simulation

def simulate(fit: CopulaFit, n_paths: int, n_periods: int,
             seed: int | None = 42) -> np.ndarray:
    """Draw uniforms from a fitted copula."""
    rng = np.random.default_rng(seed)
    d = len(fit.assets)
    L = np.linalg.cholesky(fit.corr)

    z = rng.standard_normal((n_paths * n_periods, d)) @ L.T
    if fit.df is None:
        u = stats.norm.cdf(z)
    else:
        w = rng.chisquare(fit.df, size=(n_paths * n_periods, 1)) / fit.df
        x = z / np.sqrt(w)
        u = stats.t.cdf(x, df=fit.df)
    return u.reshape(n_paths, n_periods, d)


def simulate_returns(fit: CopulaFit, marginals: pd.DataFrame, n_paths: int,
                     n_periods: int, seed: int | None = 42) -> np.ndarray:
    """
    Simulated returns, mapping copula uniforms through empirical marginals.
    """
    u = simulate(fit, n_paths, n_periods, seed)
    out = np.empty_like(u)
    for k, col in enumerate(fit.assets):
        sample = marginals[col].dropna().to_numpy()
        out[:, :, k] = np.quantile(sample, np.clip(u[:, :, k], 1e-6, 1 - 1e-6))
    return out
