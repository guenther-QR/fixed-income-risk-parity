"""Regime-conditional allocation.

The Phase 6 return tilts fought for about 0.03 of Sharpe. Conditioning on the
macro regime instead has roughly ten times the headroom in sample, and for a
reason worth stating plainly: **predicting a four-class label with 86% monthly
persistence is a far easier problem than predicting a continuous return with 1%
R-squared.** Classification error is where the tractable signal lives.

A diagnostic on the full sample, before any of this was built:

    perfect-foresight regime switching     Sharpe 0.855
    lagged regime, no foresight at all     Sharpe 0.880
    single static max-Sharpe portfolio     Sharpe 0.561

The middle line is the important one. Using *last month's* regime does as well as
knowing this month's, because regimes persist. If that survives out of sample, the
transition-prediction machinery is unnecessary - observation beats forecasting -
and that is a cleaner result than a marginal forecasting win would have been.

Two allocation rules are provided. `HardSwitch` holds the portfolio fitted for
whichever regime is currently observed. `Blend` holds a probability-weighted
mixture, which trades less and degrades more gracefully when the regime label is
uncertain, as it always is near a turning point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import covariance as cov, optimize as opt

REGIMES = ["Goldilocks", "Reflation", "Stagflation", "Deflation"]


@dataclass
class RegimeWeights:
    """Per-regime weights, fitted on a training window."""
    weights: dict[str, np.ndarray]
    assets: list[str]
    counts: dict[str, int] = field(default_factory=dict)
    fallback: np.ndarray | None = None

    def get(self, regime: str | float) -> np.ndarray:
        """
        Weights for a regime, falling back when the label is missing or the
        regime was too rare in training to fit.

        The fallback is the unconditional portfolio, not equal weight: a regime
        seen twice in the training window should contribute nothing, and
        defaulting to the best unconditional answer is more honest than
        defaulting to an arbitrary one.
        """
        if not isinstance(regime, str) or regime not in self.weights:
            return self.fallback
        return self.weights[regime]


def fit_regime_weights(returns: pd.DataFrame, regimes: pd.Series,
                       rf: pd.Series, objective: str = "risk_parity",
                       min_months: int = 24) -> RegimeWeights:
    """
    Fit one portfolio per regime, on the supplied window only.

    Regimes with fewer than `min_months` observations get no portfolio of their
    own. With four regimes and an expanding window that starts at 180 months,
    the rarer states are thin early on, and fitting a seven-asset covariance
    matrix on twenty observations produces weights that are noise wearing a
    label.
    """
    d = pd.concat([returns, regimes.rename("regime")], axis=1).dropna()
    R = d[returns.columns]
    g = d["regime"]
    RF = rf.reindex(d.index)

    def solve(sub: pd.DataFrame, subrf: pd.Series) -> np.ndarray | None:
        if len(sub) < min_months:
            return None
        m = opt.estimate_moments(sub, subrf)
        try:
            m.sigma = cov.ledoit_wolf(sub.sub(subrf.reindex(sub.index), axis=0).dropna())
            if objective == "risk_parity":
                return opt.risk_parity(m)
            if objective == "max_sharpe":
                return opt.max_sharpe(m)
            if objective == "min_variance":
                return opt.min_variance(m)
            raise ValueError(objective)
        except Exception:
            return None

    fallback = solve(R, RF)
    if fallback is None:
        fallback = np.full(R.shape[1], 1.0 / R.shape[1])

    weights, counts = {}, {}
    for state in REGIMES:
        mask = g == state
        counts[state] = int(mask.sum())
        w = solve(R[mask], RF[mask])
        if w is not None:
            weights[state] = w

    return RegimeWeights(weights=weights, assets=list(R.columns),
                         counts=counts, fallback=fallback)


@dataclass
class HardSwitch:
    """Hold the portfolio for whichever regime is currently observed."""
    regimes: pd.Series
    objective: str = "risk_parity"
    min_months: int = 24

    def __call__(self, train: pd.DataFrame, rf: pd.Series | None = None) -> np.ndarray:
        reg_train = self.regimes.reindex(train.index)
        fitted = fit_regime_weights(train, reg_train, rf, self.objective,
                                    self.min_months)

        # The regime label used for the decision is the last one observed inside
        # the training window - the engine has already excluded anything later,
        # so this is the most recent knowable state.
        current = reg_train.dropna()
        state = current.iloc[-1] if len(current) else np.nan
        return fitted.get(state)


@dataclass
class Blend:
    """
    Hold a probability-weighted mixture of the regime portfolios.

    Probabilities come from the empirical transition matrix applied to the
    currently observed state, so this is a one-step-ahead expectation rather than
    a forecast: it uses persistence, not prediction. With transition
    probabilities around 0.86 on the diagonal, the blend sits close to the
    hard-switch portfolio but moves between them gradually, which cuts turnover
    at regime boundaries where the label is least reliable.
    """
    regimes: pd.Series
    objective: str = "risk_parity"
    min_months: int = 24
    smoothing: float = 1.0          # 1.0 = one transition step, 0.0 = hard switch

    def __call__(self, train: pd.DataFrame, rf: pd.Series | None = None) -> np.ndarray:
        reg_train = self.regimes.reindex(train.index)
        fitted = fit_regime_weights(train, reg_train, rf, self.objective,
                                    self.min_months)

        current = reg_train.dropna()
        if len(current) == 0:
            return fitted.fallback
        state = current.iloc[-1]

        probs = _transition_row(current, state, self.smoothing)
        w = np.zeros(len(fitted.assets))
        total = 0.0
        for regime, p in probs.items():
            wr = fitted.get(regime)
            if wr is not None and p > 0:
                w += p * wr
                total += p
        return w / total if total > 0 else fitted.fallback


def _transition_row(history: pd.Series, state: str,
                    smoothing: float) -> dict[str, float]:
    """
    One row of the empirical transition matrix, estimated on the training window.

    Laplace-smoothed, so a transition never observed in training is assigned a
    small probability rather than zero. Without it an unseen transition makes the
    strategy structurally unable to move to that regime.
    """
    s = history.dropna()
    if len(s) < 2:
        return {state: 1.0}

    pairs = pd.DataFrame({"from": s.iloc[:-1].to_numpy(), "to": s.iloc[1:].to_numpy()})
    row = pairs[pairs["from"] == state]["to"].value_counts()
    counts = {r: float(row.get(r, 0.0)) + 0.5 for r in REGIMES}
    total = sum(counts.values())
    probs = {r: c / total for r, c in counts.items()}

    if smoothing < 1.0:
        hard = {r: (1.0 if r == state else 0.0) for r in REGIMES}
        probs = {r: smoothing * probs[r] + (1 - smoothing) * hard[r] for r in REGIMES}
    return probs


def regime_summary(fitted: RegimeWeights) -> pd.DataFrame:
    """Fitted weights per regime, for inspection."""
    rows = {}
    for state in REGIMES:
        w = fitted.weights.get(state)
        rows[state] = (dict(zip(fitted.assets, w)) if w is not None
                       else {a: np.nan for a in fitted.assets})
        rows[state]["n_months"] = fitted.counts.get(state, 0)
    return pd.DataFrame(rows).T
