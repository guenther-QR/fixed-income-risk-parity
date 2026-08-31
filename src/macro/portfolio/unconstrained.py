"""Unconstrained allocation: the only limit is 2x gross leverage.

Every earlier construction in this project carried constraints that were never
requested - long-only, per-asset deviation caps, structural weight ceilings. Each
was defensible in isolation and each quietly bounded what the strategy could
express. Removed here. Weights may be negative, exposure may exceed one, and the
single binding rule is

    sum |w_i| <= 2.0

Two consequences that must be priced rather than assumed away.

**Leverage is not free.** Gross exposure above 1x is borrowed, and borrowing
costs the risk-free rate plus a spread. The financing charge is
`(gross - 1) * spread` applied every period, so a book running at 2x pays the
spread on a full unit of capital annually.

**Shorting is not free either.** A short position pays a borrow fee and, for
anything but the most liquid names, that fee is not small. Charged here at the
same per-asset transaction rate applied to the short leg.

Removing constraints does not make a strategy better; it removes the excuse that
constraints were hiding the signal. If an unconstrained version still cannot
beat 60/40 after paying for its leverage, the constraint was never the problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import covariance as cov, optimize as opt

MAX_GROSS = 2.0


def scale_to_gross(w: np.ndarray, max_gross: float = MAX_GROSS) -> np.ndarray:
    """
    Scale weights so gross exposure respects the limit.

    Scaling rather than clipping: clipping individual weights changes the
    portfolio's composition, while scaling preserves the relative bet and only
    reduces its size. That keeps the strategy's view intact and adjusts the
    conviction.
    """
    gross = np.abs(w).sum()
    if gross > max_gross and gross > 0:
        return w * (max_gross / gross)
    return w


def gross_exposure(w: np.ndarray) -> float:
    return float(np.abs(w).sum())


def net_exposure(w: np.ndarray) -> float:
    return float(w.sum())


def financing_cost(w: np.ndarray, spread_bp: float = 50.0,
                   periods_per_year: int = 12) -> float:
    """Cost of the borrowed portion, per period."""
    borrowed = max(gross_exposure(w) - 1.0, 0.0)
    return borrowed * (spread_bp / 1e4) / periods_per_year


def short_cost(w: np.ndarray, rates: np.ndarray,
               periods_per_year: int = 12) -> float:
    """Borrow fee on short positions, per period."""
    shorts = np.minimum(w, 0.0)
    return float(np.abs(shorts) @ rates / periods_per_year)


@dataclass
class UnconstrainedMV:
    """
    Mean-variance with no sign or size restriction beyond gross leverage.

    The analytic tangency solution `Sigma^-1 mu` is used rather than a numerical
    optimizer, because with no bounds the problem has a closed form. It is also
    notoriously unstable - which is the honest state of an unconstrained
    mean-variance portfolio, and precisely what the constraints elsewhere were
    concealing.
    """
    max_gross: float = MAX_GROSS
    shrink: bool = True

    def __call__(self, train: pd.DataFrame, rf: pd.Series | None = None) -> np.ndarray:
        ex = (train.sub(rf.reindex(train.index), axis=0).dropna()
              if rf is not None else train)
        m = opt.estimate_moments(train, rf)
        sigma = cov.ledoit_wolf(ex) if self.shrink else cov.sample(ex)
        try:
            w = np.linalg.solve(sigma, m.mu)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(sigma) @ m.mu
        return scale_to_gross(w, self.max_gross)


@dataclass
class UnconstrainedTilt:
    """
    Benchmark plus an unconstrained overlay, limited only by gross leverage.

    The overlay is proportional to the cross-sectionally standardised edge, with
    `strength` setting its scale. Nothing clips an individual asset; the whole
    book is scaled back only if gross exposure would exceed the limit.
    """
    benchmark: dict[str, float]
    assets: list[str]
    strength: float = 0.30
    max_gross: float = MAX_GROSS

    def _bench(self) -> np.ndarray:
        return np.array([self.benchmark.get(a, 0.0) for a in self.assets])

    def apply(self, edge: np.ndarray) -> np.ndarray:
        e = np.asarray(edge, dtype=float)
        sd = e.std()
        if not np.isfinite(sd) or sd < 1e-12:
            return self._bench()
        z = (e - e.mean()) / sd
        return scale_to_gross(self._bench() + self.strength * z, self.max_gross)


def summarize_exposure(weights: pd.DataFrame) -> pd.DataFrame:
    """Gross, net and short exposure through time - what the limit is binding on."""
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    short = weights.clip(upper=0).abs().sum(axis=1)
    return pd.DataFrame({
        "gross": gross, "net": net, "short": short,
        "at_limit": (gross >= MAX_GROSS - 1e-6).astype(int),
    })
