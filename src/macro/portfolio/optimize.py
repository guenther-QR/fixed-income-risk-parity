"""Mean-variance portfolio construction.

This replaces the `PortfolioOptimizer` class from the 2025 study, which contained
three defects that this module is built to avoid.

**Arithmetic and geometric means are different objects.** The 2025 optimizer set
`mu` to the annualized *geometric* return of each asset and then computed the
portfolio's return as `w . mu`. That is not valid: the geometric mean of a
portfolio is not the weighted average of its assets' geometric means, because the
Jensen gap does not distribute across a sum. Mean-variance requires arithmetic
means; geometric return is a property of a realised path and is reported
separately here, computed by actually compounding the portfolio series.

**The risk-free rate is not zero.** Sharpe is defined on excess returns. With
`rfr = 0` the 2025 tables reported return-to-volatility ratios, which inflated
every figure and, because the distortion scales with volatility, also changed the
*ranking* between portfolios of different risk.

**Cash is a position, not an omission.** Allowing `sum(w) <= 1` lets the portfolio
de-risk, with the remainder in the risk-free asset. This only has bite under an
objective that trades return against risk, so `max_utility` and `target_vol` are
provided alongside the classic objectives; maximum-Sharpe is invariant to the
cash weight and minimum-variance would hold nothing else.

Every optimizer here is *in-sample* by construction. Nothing in this module is a
strategy: fitting weights on a window and reporting their performance on that same
window is an upper bound on what was achievable, not a result. Phase 4 supplies the
walk-forward machinery that turns these into something testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Moments:
    """Annualized inputs to mean-variance optimization."""
    mu: np.ndarray            # arithmetic excess returns
    sigma: np.ndarray         # covariance
    assets: list[str]
    periods_per_year: int
    geometric: np.ndarray     # reported, never optimized on

    @property
    def n(self) -> int:
        return len(self.assets)


def estimate_moments(returns: pd.DataFrame, rf: pd.Series | None = None,
                     periods_per_year: int = 12) -> Moments:
    """
    Annualized excess-return moments.

    `mu` is arithmetic because that is what mean-variance requires. The geometric
    mean is carried alongside for reporting, so the two can never be confused at
    the point of use.
    """
    r = returns.dropna(how="any")
    excess = r.sub(rf.reindex(r.index), axis=0).dropna(how="any") if rf is not None else r

    mu = excess.mean().to_numpy() * periods_per_year
    sigma = excess.cov().to_numpy() * periods_per_year
    geo = ((1 + r).prod() ** (periods_per_year / len(r)) - 1).to_numpy()

    return Moments(mu=mu, sigma=sigma, assets=list(r.columns),
                   periods_per_year=periods_per_year, geometric=geo)


def _constraints(n: int, allow_cash: bool):
    kind = "ineq" if allow_cash else "eq"
    fun = ((lambda w: 1.0 - np.sum(w)) if allow_cash
           else (lambda w: np.sum(w) - 1.0))
    return [{"type": kind, "fun": fun}]


def _solve(objective, n: int, bounds, allow_cash: bool, n_starts: int = 2):
    """SLSQP from several starts - the surface is not always convex once bounded."""
    best = None
    rng = np.random.default_rng(0)
    starts = [np.full(n, 1.0 / n)]
    starts += [rng.dirichlet(np.ones(n)) for _ in range(n_starts - 1)]

    for w0 in starts:
        res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                       constraints=_constraints(n, allow_cash),
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("optimization failed from every start")
    return best.x


def portfolio_stats(w: np.ndarray, m: Moments) -> dict:
    """Expected excess return, volatility and Sharpe for a weight vector."""
    ret = float(w @ m.mu)
    vol = float(np.sqrt(w @ m.sigma @ w))
    return {"exp_excess_return": ret, "vol": vol,
            "sharpe": ret / vol if vol > 0 else np.nan,
            "cash_weight": float(1.0 - w.sum())}


def min_variance(m: Moments, bounds=(0.0, 1.0), allow_cash: bool = False) -> np.ndarray:
    if allow_cash:
        raise ValueError(
            "minimum variance with cash allowed is degenerate - cash has no "
            "variance, so the solution is 100% cash. Use target_vol or max_utility."
        )
    return _solve(lambda w: w @ m.sigma @ w, m.n, [bounds] * m.n, False)


def max_sharpe(m: Moments, bounds=(0.0, 1.0)) -> np.ndarray:
    """
    Tangency portfolio.

    Cash is deliberately not permitted: by two-fund separation it would rescale
    the portfolio without changing its Sharpe, so the objective cannot pin it
    down and the result would be arbitrary.
    """
    def neg_sharpe(w):
        vol = np.sqrt(w @ m.sigma @ w)
        return -(w @ m.mu) / vol if vol > 1e-12 else 1e6
    return _solve(neg_sharpe, m.n, [bounds] * m.n, False)


def max_utility(m: Moments, risk_aversion: float = 3.0, bounds=(0.0, 1.0),
                allow_cash: bool = True) -> np.ndarray:
    """
    Maximize w'mu - (gamma/2) w'Sigma w.

    The objective under which cash is meaningful: higher risk aversion pushes
    weight out of risky assets and into the unallocated remainder.
    """
    def neg_utility(w):
        return -(w @ m.mu) + 0.5 * risk_aversion * (w @ m.sigma @ w)
    return _solve(neg_utility, m.n, [bounds] * m.n, allow_cash)


def target_vol(m: Moments, target: float = 0.08, bounds=(0.0, 1.0)) -> np.ndarray:
    """
    Maximize expected return subject to volatility not exceeding `target`.

    The other objective that gives cash a role: when no risky mix can be held
    within the target, the optimizer holds less of everything.
    """
    n = m.n
    cons = _constraints(n, allow_cash=True) + [
        {"type": "ineq", "fun": lambda w: target ** 2 - w @ m.sigma @ w}
    ]
    best, rng = None, np.random.default_rng(0)
    for w0 in [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(4)]:
        res = minimize(lambda w: -(w @ m.mu), w0, method="SLSQP",
                       bounds=[bounds] * n, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("target-volatility optimization failed")
    return best.x


def equal_weight(m: Moments) -> np.ndarray:
    """1/N. The benchmark DeMiguel et al. (2009) show is hard to beat."""
    return np.full(m.n, 1.0 / m.n)


def risk_parity(m: Moments, bounds=(1e-4, 1.0)) -> np.ndarray:
    """Weights whose risk contributions are equal."""
    def obj(w):
        port_vol = np.sqrt(w @ m.sigma @ w)
        contrib = w * (m.sigma @ w) / port_vol
        return float(np.sum((contrib - contrib.mean()) ** 2))
    w = _solve(obj, m.n, [bounds] * m.n, False)
    return w / w.sum()


def realized_stats(returns: pd.DataFrame, w: np.ndarray, rf: pd.Series | None = None,
                   periods_per_year: int = 12) -> dict:
    """
    Statistics from actually compounding the weighted return series.

    This is the honest counterpart to `portfolio_stats`, which reports the
    optimizer's own expectations. Geometric return appears only here, computed
    from the realised path rather than from a weighted average of asset
    geometric means.
    """
    r = returns.dropna(how="any")
    port = pd.Series(r.to_numpy() @ w, index=r.index)
    n = len(port)
    vol = port.std() * np.sqrt(periods_per_year)
    cum = (1 + port).cumprod()

    out = {
        "geometric_return": (1 + port).prod() ** (periods_per_year / n) - 1,
        "arithmetic_return": port.mean() * periods_per_year,
        "vol": vol,
        "max_drawdown": (cum / cum.cummax() - 1).min(),
        "months": n,
    }
    if rf is not None:
        ex = (port - rf.reindex(port.index)).dropna()
        out["sharpe"] = ex.mean() * periods_per_year / vol if vol > 0 else np.nan
    return out
