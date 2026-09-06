"""
Regime-conditional expected returns, shrunk toward the unconditional mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import covariance as cov, optimize as opt
from .episodic import completed_months, episode_table, n_completed_episodes

REGIMES = ["Goldilocks", "Reflation", "Stagflation", "Deflation"]


def shrunk_regime_mean(returns: pd.DataFrame, regimes: pd.Series,
                       state: str, k: float = 60.0,
                       use_completed_only: bool = True,
                       asof: pd.Timestamp | None = None) -> tuple[np.ndarray, int]:
    """
    Expected excess returns for `state`, shrunk toward the unconditional
    mean.
    """
    mu_all = returns.mean().to_numpy() * 12.0

    if use_completed_only and asof is not None:
        eps = episode_table(regimes)
        months = [m for m in completed_months(eps, state, asof) if m in returns.index]
        sub = returns.loc[months] if months else returns.iloc[:0]
    else:
        sub = returns[regimes.reindex(returns.index) == state]

    n = len(sub)
    if n < 6:
        return mu_all, n

    mu_regime = sub.mean().to_numpy() * 12.0
    w = n / (n + k)
    return w * mu_regime + (1 - w) * mu_all, n


@dataclass
class RegimeMeanTilt:
    """
    Tilt away from a benchmark using shrunk regime-conditional expected
    returns.
    """
    regimes: pd.Series
    assets: list[str]
    benchmark: dict[str, float]
    k_shrink: float = 60.0
    max_dev_per_asset: float = 0.15
    max_total_dev: float = 0.40
    shade_rate: float = 0.5
    min_episodes: int = 1
    use_covariance: bool = False        # scale the tilt by inverse volatility
    log: list = field(default_factory=list)

    def _bench(self) -> np.ndarray:
        return np.array([self.benchmark.get(a, 0.0) for a in self.assets])

    def __call__(self, train: pd.DataFrame, rf: pd.Series | None = None) -> np.ndarray:
        bench = self._bench()
        reg = self.regimes.reindex(train.index).dropna()
        if reg.empty:
            return bench

        asof = train.index[-1]
        state = reg.iloc[-1]
        eps = episode_table(reg)
        if n_completed_episodes(eps, state, asof) < self.min_episodes:
            self.log.append({"date": asof, "regime": state, "action": "benchmark",
                             "n_obs": 0, "alpha": 0.0})
            return bench

        excess = (train.sub(rf.reindex(train.index), axis=0).dropna()
                  if rf is not None else train)
        mu, n_obs = shrunk_regime_mean(excess, reg, state, self.k_shrink,
                                       use_completed_only=True, asof=asof)
        mu_all = excess.mean().to_numpy() * 12.0

        # Deviation is driven by how far the regime moves each asset's
        # expected return relative to its unconditional value, standardised
        # across assets.
        edge = mu - mu_all
        if self.use_covariance:
            vol = excess.std().to_numpy() * np.sqrt(12)
            edge = edge / np.where(vol > 1e-9, vol, 1.0)

        sd = edge.std()
        if sd < 1e-12:
            return bench
        z = (edge - edge.mean()) / sd

        dev = np.clip(z * self.max_dev_per_asset,
                      -self.max_dev_per_asset, self.max_dev_per_asset)

        k_run = _run_length(reg, state)
        alpha = (1.0 - (1.0 - self.shade_rate) ** k_run
                 if self.shade_rate < 1.0 else 1.0)
        w = bench + alpha * dev
        w = np.maximum(w, 0.0)

        excess_dev = np.abs(w - bench).sum()
        if excess_dev > self.max_total_dev and excess_dev > 0:
            w = bench + (w - bench) * (self.max_total_dev / excess_dev)
            w = np.maximum(w, 0.0)

        total = w.sum()
        if total > 1.0:
            w = w / total

        self.log.append({"date": asof, "regime": state, "action": "tilt",
                         "n_obs": n_obs, "alpha": alpha,
                         "deviation": float(np.abs(w - bench).sum())})
        return w

    def activity(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)


def _run_length(regimes: pd.Series, state: str) -> int:
    k = 0
    for v in reversed(regimes.dropna().to_numpy()):
        if v == state:
            k += 1
        else:
            break
    return k


def regime_edge_table(returns: pd.DataFrame, regimes: pd.Series,
                      rf: pd.Series, k: float = 60.0) -> pd.DataFrame:
    """
    What each regime says about expected returns, before and after
    shrinkage.
    """
    excess = returns.sub(rf.reindex(returns.index), axis=0).dropna()
    mu_all = excess.mean() * 12.0
    rows = {}
    for state in REGIMES:
        sub = excess[regimes.reindex(excess.index) == state]
        if len(sub) < 6:
            continue
        raw = sub.mean() * 12.0
        w = len(sub) / (len(sub) + k)
        rows[f"{state} (raw)"] = raw
        rows[f"{state} (shrunk)"] = w * raw + (1 - w) * mu_all
    rows["Unconditional"] = mu_all
    return pd.DataFrame(rows).T
