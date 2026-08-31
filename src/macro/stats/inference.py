"""Inference for backtests: what survives once the search itself is priced in.

Sixty-one specifications are in the log. That number is the problem this module
exists to address. The best of sixty-one strategies looks good whether or not any
of them has an edge, and a t-statistic computed as though the winner were the
only thing ever tried is not evidence - it is arithmetic performed on the
outcome of a search while ignoring the search.

Three corrections, in increasing strictness.

**Block bootstrap.** Monthly returns are not independent - volatility clusters,
drawdowns persist - so the textbook standard error on a Sharpe ratio is too
small. The stationary bootstrap of Politis and Romano resamples blocks of
geometrically distributed length, which preserves short-range dependence while
keeping the resampled series stationary. Used here for confidence intervals on a
Sharpe *difference*, since every claim in this project is a claim about beating
60/40 rather than about a level.

**Hansen's SPA.** Given many strategies and one benchmark, the null is that
*none* of them beats it. White's Reality Check tests this by bootstrapping the
maximum, which controls the family-wise error but is badly conservative when the
set contains obviously poor models - and this set is mostly poor models, since
they were logged as they failed. Hansen's refinement recenters those out of the
null distribution instead of letting them drag it down, which restores power
without giving up the size guarantee. Both are reported.

**Deflated Sharpe.** Bailey and Lopez de Prado's adjustment asks a slightly
different question: given N independent trials, what Sharpe would the best one
show under a null of no skill? Non-normality enters directly through skew and
kurtosis, which matters here because levered and drawdown-prone strategies are
exactly the ones whose returns are least normal.

None of this rescues a strategy that lost. It establishes whether the one that
won did so for a reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

EULER = 0.5772156649015329


def stationary_bootstrap_indices(n: int, mean_block: float, rng) -> np.ndarray:
    """
    One resampled index path under the Politis-Romano stationary bootstrap.

    Each step either continues the current block or starts a new one at a random
    position, with restart probability `1/mean_block`. Geometric block lengths
    are what make the resampled series stationary; fixed-length blocks are not,
    which biases variance estimates.
    """
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def _sharpe(x: np.ndarray, ppy: int) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ppy)) if sd > 1e-12 else 0.0


def sharpe_difference(strategy: pd.Series, benchmark: pd.Series,
                      rf: pd.Series | None = None, n_boot: int = 5000,
                      mean_block: float = 12.0, ppy: int = 12,
                      seed: int = 20260830) -> dict:
    """
    Bootstrap the Sharpe difference against a benchmark.

    Paired throughout: each resample draws the *same* block of dates from both
    series, so the common market move is differenced away rather than
    contributing noise to the comparison.
    """
    d = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if rf is not None:
        r = rf.reindex(d.index).fillna(0.0)
        d = d.sub(r, axis=0)
    S, B = d["s"].to_numpy(), d["b"].to_numpy()
    n = len(d)
    obs = _sharpe(S, ppy) - _sharpe(B, ppy)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        i = stationary_bootstrap_indices(n, mean_block, rng)
        draws[b] = _sharpe(S[i], ppy) - _sharpe(B[i], ppy)

    centred = draws - draws.mean()
    return {
        "sharpe_strategy": _sharpe(S, ppy),
        "sharpe_benchmark": _sharpe(B, ppy),
        "difference": obs,
        "se": float(draws.std(ddof=1)),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        # One-sided p under the null of no difference: how often does a
        # mean-zero resample reach the observed gap?
        "p_one_sided": float((centred >= abs(obs)).mean()),
        "n_months": n,
    }


def reality_check(losses: pd.DataFrame, n_boot: int = 5000,
                  mean_block: float = 12.0, seed: int = 20260830) -> dict:
    """
    White's Reality Check and Hansen's SPA over a family of strategies.

    `losses` holds one column per strategy of its per-period outperformance
    against the benchmark. The null is that no column has positive expectation.

    The two statistics differ only in what they recenter. White compares the
    observed maximum against the bootstrap distribution of the maximum with every
    model recentered on its own mean, which lets a badly losing model inflate the
    critical value. Hansen drops models whose evidence is far enough below zero
    that they cannot plausibly be part of the null - the `sqrt(2 log log n)`
    threshold - and recenters only the rest.
    """
    D = losses.dropna()
    n, k = D.shape
    if n < 24 or k < 1:
        return {}

    dbar = D.mean().to_numpy()
    rng = np.random.default_rng(seed)

    boot = np.empty((n_boot, k))
    for b in range(n_boot):
        i = stationary_bootstrap_indices(n, mean_block, rng)
        boot[b] = D.to_numpy()[i].mean(axis=0)

    omega = boot.std(axis=0, ddof=1) * np.sqrt(n)
    omega = np.where(omega > 1e-12, omega, 1e-12)

    t_obs = np.sqrt(n) * dbar / omega
    T_obs = max(0.0, t_obs.max())

    # White: recenter every model on its own mean.
    white = np.sqrt(n) * (boot - dbar) / omega
    p_white = float((np.maximum(white.max(axis=1), 0.0) >= T_obs).mean())

    # Hansen: models far below zero are excluded from the null.
    thresh = -np.sqrt(2.0 * np.log(max(np.log(n), 1.0001)))
    keep = t_obs >= thresh
    g = np.where(keep, dbar, 0.0)
    hansen = np.sqrt(n) * (boot - g) / omega
    p_hansen = float((np.maximum(hansen.max(axis=1), 0.0) >= T_obs).mean())

    best = D.columns[int(np.argmax(t_obs))]
    return {
        "n_strategies": k, "n_months": n,
        "best": best, "best_t": float(t_obs.max()), "spa_statistic": T_obs,
        "p_reality_check": p_white, "p_spa": p_hansen,
        "n_in_null": int(keep.sum()),
    }


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    variance_of_trials: float | None = None,
                    ppy: int = 12) -> dict:
    """
    Deflated Sharpe ratio: the probability the observed Sharpe reflects skill,
    given that it is the best of `n_trials`.

    The benchmark is not zero. It is the Sharpe the *maximum* of `n_trials`
    independent, genuinely skill-free strategies would be expected to show, which
    grows with the number of trials. Skew and kurtosis enter the standard error
    because a negatively skewed, fat-tailed return stream carries more estimation
    uncertainty in its Sharpe than a normal one of the same volatility.
    """
    x = returns.dropna().to_numpy()
    n = len(x)
    if n < 24:
        return {}

    sr = x.mean() / x.std(ddof=1)                     # per period, not annualised
    g3 = float(st.skew(x))
    g4 = float(st.kurtosis(x, fisher=False))

    v = variance_of_trials if variance_of_trials is not None else (1.0 / n)
    e = np.exp(1.0)
    sr0 = np.sqrt(v) * ((1 - EULER) * st.norm.ppf(1 - 1.0 / n_trials)
                        + EULER * st.norm.ppf(1 - 1.0 / (n_trials * e)))

    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(n - 1) / denom

    return {
        "sharpe_annual": sr * np.sqrt(ppy),
        "expected_max_under_null": float(sr0 * np.sqrt(ppy)),
        "skew": g3, "kurtosis": g4, "n_trials": n_trials,
        "deflated_sharpe_prob": float(st.norm.cdf(z)),
        "min_track_record_months": float(
            1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
            * (st.norm.ppf(0.95) / max(sr - sr0, 1e-6)) ** 2)
        if sr > sr0 else np.nan,
    }
