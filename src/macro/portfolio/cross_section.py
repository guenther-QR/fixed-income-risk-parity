"""Cross-sectional portfolio construction: rank assets against each other."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_rank(signal: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Rank assets against each other at each date, scaled to [0, 1]."""
    r = signal.rank(axis=1, ascending=ascending, na_option="keep")
    n = signal.notna().sum(axis=1)
    return r.sub(1).div((n - 1).replace(0, np.nan), axis=0)


def cross_sectional_z(signal: pd.DataFrame) -> pd.DataFrame:
    """Demean and standardise across assets at each date."""
    mu = signal.mean(axis=1)
    sd = signal.std(axis=1).replace(0, np.nan)
    return signal.sub(mu, axis=0).div(sd, axis=0)


def long_short(signal: pd.DataFrame, n_long: int = 2, n_short: int = 2,
               gross: float = 1.0) -> pd.DataFrame:
    """
    Dollar-neutral: equal-weight the best `n_long`, short the worst
    `n_short`.
    """
    ranks = signal.rank(axis=1, ascending=False, na_option="keep")
    n = signal.notna().sum(axis=1)
    W = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    longs = ranks.le(n_long) & ranks.notna()
    shorts = ranks.gt(n.values[:, None] - n_short) & ranks.notna()

    W = W.mask(longs, gross / 2.0 / max(n_long, 1))
    W = W.mask(shorts, -gross / 2.0 / max(n_short, 1))
    W[n < (n_long + n_short)] = 0.0
    return W


def rank_weight(signal: pd.DataFrame, power: float = 1.0) -> pd.DataFrame:
    """Long-only weights proportional to cross-sectional rank."""
    r = cross_sectional_rank(signal)
    w = (r + 1e-6) ** power
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def tilt(signal: pd.DataFrame, benchmark: dict[str, float], assets: list[str],
         strength: float = 0.15, long_only: bool = True) -> pd.DataFrame:
    """Benchmark plus a rank-proportional overlay."""
    bv = pd.Series({a: benchmark.get(a, 0.0) for a in assets})
    z = cross_sectional_z(signal[assets])
    dev = z.mul(strength).fillna(0.0)
    dev = dev.sub(dev.mean(axis=1), axis=0)          # enforce zero net tilt
    W = dev.add(bv, axis=1)
    if long_only:
        W = W.clip(lower=0.0)
        W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0)
    return W.fillna(0.0)


def composite(signals: dict[str, pd.DataFrame], weights: dict[str, float] | None = None
              ) -> pd.DataFrame:
    """Average several rankings into one."""
    w = weights or {k: 1.0 for k in signals}
    total = sum(w.values())
    out = None
    for k, s in signals.items():
        r = cross_sectional_rank(s) * (w.get(k, 0.0) / total)
        out = r if out is None else out.add(r, fill_value=0.0)
    return out


def information_coefficient(signal: pd.DataFrame, forward: pd.DataFrame) -> pd.Series:
    """
    Per-date Spearman correlation between the ranking and realised returns.
    """
    common = signal.index.intersection(forward.index)
    s, f = signal.loc[common], forward.loc[common]
    out = {}
    for d in common:
        a, b = s.loc[d], f.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 3:
            out[d] = a[ok].rank().corr(b[ok].rank())
    return pd.Series(out).sort_index()


def ic_summary(ic: pd.Series, periods_per_year: int = 12) -> dict:
    ic = ic.dropna()
    if ic.empty:
        return {}
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    t = mean / sd * np.sqrt(len(ic)) if sd > 0 else np.nan
    return {
        "ic_mean": mean, "ic_std": sd, "ic_t_stat": float(t),
        "ic_ir": float(mean / sd) if sd > 0 else np.nan,
        "hit_rate": float((ic > 0).mean()), "n_periods": int(len(ic)),
        # Grinold: IR = IC * sqrt(breadth).
        "implied_ir": float(mean * np.sqrt(periods_per_year)) if np.isfinite(mean) else np.nan,
    }


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1)
