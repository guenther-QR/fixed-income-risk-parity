"""
Inference for backtests: what survives once the search itself is priced in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

EULER = 0.5772156649015329


def stationary_bootstrap_indices(n: int, mean_block: float, rng) -> np.ndarray:
    """
    One resampled index path under the Politis-Romano stationary bootstrap.
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
    """Bootstrap the Sharpe difference against a benchmark."""
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


def difference_t_test(strategy: pd.Series, benchmark: pd.Series,
                      rf: pd.Series | None = None, ppy: int = 12) -> dict:
    """Paired t-test on the difference in excess returns."""
    d = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if rf is not None:
        r = rf.reindex(d.index).fillna(0.0)
        d = d.sub(r, axis=0)
    diff = (d["s"] - d["b"]).to_numpy()
    n = len(diff)
    if n < 3:
        return {}
    sd = diff.std(ddof=1)
    if sd <= 0:
        return {}
    se = sd / np.sqrt(n)
    t = float(diff.mean() / se)
    p = float(st.t.sf(t, df=n - 1))          # one-sided: strategy is better
    crit = float(st.t.ppf(0.975, df=n - 1))
    return {
        "mean_difference": float(diff.mean() * ppy),
        "se": float(se * ppy),
        "t_stat": t,
        "p_one_sided": p,
        "ci_lo": float((diff.mean() - crit * se) * ppy),
        "ci_hi": float((diff.mean() + crit * se) * ppy),
        "n_obs": n,
    }


def reality_check(losses: pd.DataFrame, n_boot: int = 5000,
                  mean_block: float = 12.0, seed: int = 20260830) -> dict:
    """White's Reality Check and Hansen's SPA over a family of strategies."""
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
    Deflated Sharpe ratio: the probability the observed Sharpe reflects
    skill, given that it is the best of `n_trials`.
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
