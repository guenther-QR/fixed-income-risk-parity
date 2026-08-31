"""Affine term structure model estimated under the physical measure.

Hull-White and G2++ are risk-neutral: expected returns on every bond equal the
short rate by construction, which is why neither can forecast excess returns. The
missing ingredient is the price of risk. An affine model with market prices of
risk lambda(X) = lambda0 + lambda1 X separates two measures:

    under Q  bonds are priced, and every bond earns r
    under P  bonds are *held*, and the expected excess return is
             beta' (lambda0 + lambda1 X_t)

That second quantity is a genuine, time-varying forecast of bond excess returns,
and the gap between the fitted yield and the yield the model would produce with
lambda set to zero is the term premium.

Estimation follows Adrian, Crump and Moench (2013). Their contribution is that a
model which had required numerically fragile maximum likelihood can be estimated
by three ordinary least squares regressions:

    1. pricing factors from the cross-section of yields (PCA)
    2. a VAR(1) on those factors, whose residuals are the priced shocks
    3. excess returns on the shocks and on the lagged factors, from which
       lambda0 and lambda1 follow in closed form

The Federal Reserve Bank of New York publishes term premium estimates from this
model, which makes it one of the few term-structure models whose output can be
checked directly against an official series.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ACMResult:
    factors: pd.DataFrame          # pricing factors X_t
    loadings: pd.DataFrame         # PCA loadings, factors x maturities
    beta: np.ndarray               # exposure of each maturity's excess return to shocks
    lambda0: np.ndarray            # constant price of risk
    lambda1: np.ndarray            # factor-dependent price of risk
    mu: np.ndarray                 # VAR intercept
    Phi: np.ndarray                # VAR transition
    Sigma: np.ndarray              # shock covariance
    delta0: float                  # short rate intercept
    delta1: np.ndarray             # short rate loadings
    sigma2: float                  # excess-return residual variance
    maturities: np.ndarray         # in periods
    expected_excess: pd.DataFrame = field(default_factory=pd.DataFrame)
    fitted_yield: pd.DataFrame = field(default_factory=pd.DataFrame)
    riskneutral_yield: pd.DataFrame = field(default_factory=pd.DataFrame)
    term_premium: pd.DataFrame = field(default_factory=pd.DataFrame)


def _label(n_periods: float, periods_per_year: int) -> str:
    y = n_periods / periods_per_year
    return f"{y:g}y"


def _log_prices(zero: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    """Log zero-coupon bond prices from continuously compounded yields in percent."""
    tau = zero.columns.to_numpy(dtype=float)
    return pd.DataFrame(-(zero.to_numpy() / 100.0) * tau, index=zero.index,
                        columns=(tau * periods_per_year).round().astype(int))


def _interp_yield(zero: pd.DataFrame, maturity_years: float) -> np.ndarray:
    """The zero yield at an arbitrary maturity, for every date in the panel."""
    grid = zero.columns.to_numpy(dtype=float)
    vals = zero.to_numpy(dtype=float)
    return np.array([np.interp(maturity_years, grid, row) for row in vals])


def fit(zero: pd.DataFrame, n_factors: int = 5, periods_per_year: int = 12,
        maturities_years=None) -> ACMResult:
    """
    Estimate the model on a panel of zero yields.

    `zero` must be sampled at `periods_per_year` (monthly by default) with
    maturities in years as columns. The full curve is used to interpolate the
    one-period-shorter bond prices that excess returns require.

    The cross-section needs to be *dense*. lambda1 has n_factors^2 free
    parameters - 25 for a five-factor model - so estimating it from a handful of
    maturities leaves it badly identified. On six maturities the fitted
    risk-neutral dynamics came out explosive, max|eig(Phi - lambda1)| = 1.06,
    which compounds over a ten-year recursion and threw fitted yields off by
    hundreds of basis points. ACM use every maturity from one to ten years;
    the default here is a semiannual grid over the same span.
    """
    if maturities_years is None:
        maturities_years = tuple(np.arange(1.0, 10.5, 0.5))

    zero = zero.dropna(how="any")
    n_vec = np.array([round(m * periods_per_year) for m in maturities_years], dtype=float)
    tau_years = n_vec / periods_per_year

    # Yields at the cross-section maturities, interpolated off the full curve.
    Yc = np.column_stack([_interp_yield(zero, m) for m in tau_years])
    z = pd.DataFrame(Yc, index=zero.index, columns=tau_years)
    logP = pd.DataFrame(-(Yc / 100.0) * tau_years, index=zero.index,
                        columns=n_vec.astype(int))

    # --- Step 1: pricing factors --------------------------------------------
    Y = z.to_numpy(dtype=float)
    Y_c = Y - Y.mean(axis=0)
    _, _, vt = np.linalg.svd(Y_c, full_matrices=False)
    W = vt[:n_factors]
    X = Y_c @ W.T                                       # T x K

    # --- Step 2: VAR(1) on the factors --------------------------------------
    X0, X1 = X[:-1], X[1:]
    Z = np.column_stack([np.ones(len(X0)), X0])
    coef, *_ = np.linalg.lstsq(Z, X1, rcond=None)       # (K+1) x K
    mu, Phi = coef[0], coef[1:].T
    V = X1 - Z @ coef                                   # T-1 x K innovations
    Sigma = np.cov(V, rowvar=False, ddof=1)

    # --- Excess returns -----------------------------------------------------
    # rx^{(n)}_{t+1} = p^{(n-1)}_{t+1} - p^{(n)}_t - r_t, with the short rate
    # taken as the shortest available yield.
    # The short rate is the shortest point on the *full* curve, not the shortest
    # cross-section maturity, which is a year out.
    short = _interp_yield(zero, float(zero.columns.min())) / 100.0 / periods_per_year

    rx = []
    for j, n in enumerate(n_vec):
        p_t = logP.iloc[:-1, j].to_numpy()
        # Price next period of the same bond, now one period shorter. Interpolated
        # in yield space off the full curve: linear interpolation of log prices
        # across a sparse maturity set clamps at the short end and is badly biased.
        tau_next = (n - 1) / periods_per_year
        y_next = _interp_yield(zero, tau_next)[1:]
        p_next = -(y_next / 100.0) * tau_next
        rx.append(p_next - p_t - short[:-1])
    rx = np.array(rx)                                   # N x (T-1)

    # --- Step 3: excess returns on shocks and lagged factors ----------------
    R = np.column_stack([np.ones(V.shape[0]), V, X0])
    theta, *_ = np.linalg.lstsq(R, rx.T, rcond=None)    # (1+K+K) x N
    a = theta[0]                                        # N
    beta = theta[1:1 + n_factors]                       # K x N
    c = theta[1 + n_factors:]                           # K x N

    resid = rx.T - R @ theta
    sigma2 = float(np.mean(resid ** 2))

    # --- Prices of risk in closed form --------------------------------------
    BB = beta @ beta.T
    BB_inv = np.linalg.pinv(BB)

    # The convexity adjustment: 0.5 * (beta_n' Sigma beta_n + sigma^2) per maturity.
    convexity = 0.5 * (np.einsum("kn,kl,ln->n", beta, Sigma, beta) + sigma2)
    lambda0 = BB_inv @ beta @ (a + convexity)
    lambda1 = BB_inv @ beta @ c.T

    # --- Short rate equation ------------------------------------------------
    r_obs = short[:len(X)]
    Zr = np.column_stack([np.ones(len(X)), X])
    dcoef, *_ = np.linalg.lstsq(Zr, r_obs, rcond=None)
    delta0, delta1 = float(dcoef[0]), dcoef[1:]

    res = ACMResult(
        factors=pd.DataFrame(X, index=z.index,
                             columns=[f"X{i+1}" for i in range(n_factors)]),
        loadings=pd.DataFrame(W, index=[f"X{i+1}" for i in range(n_factors)],
                              columns=z.columns),
        beta=beta, lambda0=lambda0, lambda1=lambda1,
        mu=mu, Phi=Phi, Sigma=Sigma, delta0=delta0, delta1=delta1,
        sigma2=sigma2, maturities=n_vec,
    )

    # --- Expected excess returns under P ------------------------------------
    # E_t[rx_{t+1}] = beta' (lambda0 + lambda1 X_t)
    prices = lambda0[None, :] + res.factors.to_numpy() @ lambda1.T
    exp_rx = prices @ beta                              # T x N
    res.expected_excess = pd.DataFrame(
        exp_rx * periods_per_year, index=res.factors.index,
        columns=[_label(n, periods_per_year) for n in n_vec],
    )

    # --- Fitted and risk-neutral yields, and the term premium ---------------
    res.fitted_yield, res.riskneutral_yield = _recursions(res, periods_per_year)
    res.term_premium = res.fitted_yield - res.riskneutral_yield
    return res


def _recursions(res: ACMResult, periods_per_year: int
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Roll the affine recursions forward, once with prices of risk and once without.

        A_{n+1}  = A_n + B_n'(mu - lambda0*) + 0.5(B_n' Sigma B_n + sigma^2) - delta0
        B_{n+1}' = B_n'(Phi - lambda1*) - delta1'

    Setting lambda0* = lambda1* = 0 gives the yields that would prevail if
    investors demanded no compensation for interest-rate risk. The difference
    between the two is the term premium.
    """
    K = len(res.delta1)
    n_max = int(res.maturities.max())
    out = {}

    for label, l0, l1 in [("fitted", res.lambda0, res.lambda1),
                          ("rn", np.zeros(K), np.zeros((K, K)))]:
        A = np.zeros(n_max + 1)
        B = np.zeros((n_max + 1, K))
        for n in range(n_max):
            A[n + 1] = (A[n] + B[n] @ (res.mu - l0)
                        + 0.5 * (B[n] @ res.Sigma @ B[n] + res.sigma2) - res.delta0)
            B[n + 1] = B[n] @ (res.Phi - l1) - res.delta1

        X = res.factors.to_numpy()
        cols, data = [], []
        for n in res.maturities.astype(int):
            price = A[n] + X @ B[n]
            data.append(-price / n * periods_per_year * 100.0)   # annualised, percent
            cols.append(_label(n, periods_per_year))
        out[label] = pd.DataFrame(np.array(data).T, index=res.factors.index, columns=cols)

    return out["fitted"], out["rn"]
