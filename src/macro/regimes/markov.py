"""Markov-switching regime models.

The quadrant classifier in `classify.py` applies a rule chosen in advance. This
module instead *estimates* regimes: a Markov-switching model infers both the
number of periods spent in each state and the transition probabilities between
them, from the data alone.

The two answer different questions and it is worth keeping both. The quadrant
scheme is interpretable and maps onto an economic story a reader can follow. The
Markov model is agnostic about what a regime means but honest about when one
began, and it supplies smoothed *probabilities* rather than hard labels, which
matters at turning points where a rule flips back and forth.

Estimated with statsmodels' MarkovRegression by expectation-maximisation. The
label a state receives is arbitrary - the estimator has no notion of which state
is "the bad one" - so states are ordered by fitted volatility before being
returned, and the highest-volatility state is the crisis state by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_switching(series: pd.Series, k_regimes: int = 2,
                  switching_variance: bool = True,
                  trend: str = "c") -> dict:
    """
    Fit a Markov-switching model to one series.

    With `switching_variance` the model allows both mean and variance to differ
    across states, which is what lets it separate calm from turbulent periods
    rather than only high-return from low-return ones.
    """
    s = series.dropna()
    model = sm.tsa.MarkovRegression(
        s.to_numpy(dtype=float), k_regimes=k_regimes,
        trend=trend, switching_variance=switching_variance,
    )
    # EM on a switching model is fragile in two directions. Too few restarts and
    # it collapses onto a boundary solution, one state absorbing everything with
    # its variance driven to zero. Too many and some random start eventually
    # produces a singular design matrix, which surfaces as an SVD convergence
    # failure deep inside statsmodels. Neither is a property of the data, so the
    # fit is retried across a few restart counts and the first success is taken.
    res = None
    errors = []
    for reps, em in [(50, 100), (25, 100), (10, 150), (0, 200)]:
        try:
            res = model.fit(em_iter=em, search_reps=reps, disp=False)
            break
        except (np.linalg.LinAlgError, ValueError) as exc:
            errors.append(f"search_reps={reps}: {type(exc).__name__}")
    if res is None:
        raise RuntimeError("Markov fit failed at every setting: " + "; ".join(errors))

    # Parameters must be located by name. statsmodels orders them transition
    # probabilities first, then the regime-specific constants and variances, so
    # slicing params[:k] returns transition probabilities rather than means.
    names = list(res.model.param_names)
    means = np.array([res.params[names.index(f"const[{i}]")] for i in range(k_regimes)])
    variances = np.array([res.params[names.index(f"sigma2[{i}]")]
                          for i in range(k_regimes)], dtype=float)
    _check_degenerate(variances)

    order = np.argsort(variances)          # calmest state first

    smoothed = pd.DataFrame(
        np.asarray(res.smoothed_marginal_probabilities)[:, order],
        index=s.index, columns=[f"state{i}" for i in range(k_regimes)],
    )
    state = smoothed.to_numpy().argmax(axis=1)

    return {
        "result": res,
        "smoothed": smoothed,
        "state": pd.Series(state, index=s.index, name="state"),
        "means": means[order],
        "variances": variances[order],
        "transition": _transition(res, order, k_regimes),
        "expected_duration": _durations(res, order, k_regimes),
        "llf": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


def _check_degenerate(variances: np.ndarray, floor: float = 1e-10) -> None:
    """Reject a fit that has collapsed one state's variance to zero."""
    if np.any(variances <= floor):
        raise RuntimeError(
            f"Markov fit degenerate: a state variance collapsed to {variances.min():.2e}. "
            "Increase search_reps, or the series may not support this many regimes."
        )


def _transition(res, order: np.ndarray, k: int) -> pd.DataFrame:
    P = np.asarray(res.regime_transition)[:, :, 0]
    P = P[np.ix_(order, order)]
    names = [f"state{i}" for i in range(k)]
    return pd.DataFrame(P.T, index=names, columns=names)


def _durations(res, order: np.ndarray, k: int) -> pd.Series:
    P = np.asarray(res.regime_transition)[:, :, 0][np.ix_(order, order)]
    stay = np.clip(np.diag(P), 1e-9, 1 - 1e-9)
    return pd.Series(1.0 / (1.0 - stay), index=[f"state{i}" for i in range(k)],
                     name="expected_months")


def crisis_probability(fit: dict) -> pd.Series:
    """Smoothed probability of being in the highest-volatility state."""
    return fit["smoothed"].iloc[:, -1].rename("crisis_prob")


def compare_to_recessions(state_prob: pd.Series, usrec: pd.Series) -> dict:
    """
    How well an estimated state lines up with NBER recession dates.

    The model never sees the NBER dates, so agreement is genuine external
    validation rather than a fitted result.
    """
    j = pd.concat([state_prob.rename("p"), usrec.rename("rec")], axis=1).dropna()
    if j.empty:
        return {}
    in_rec = j.loc[j["rec"] == 1, "p"]
    out_rec = j.loc[j["rec"] == 0, "p"]
    return {
        "mean_prob_in_recession": float(in_rec.mean()),
        "mean_prob_outside": float(out_rec.mean()),
        "correlation": float(j["p"].corr(j["rec"])),
        "months_recession": int((j["rec"] == 1).sum()),
    }
