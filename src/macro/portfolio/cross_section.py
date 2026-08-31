"""Cross-sectional portfolio construction: rank assets against each other.

Every strategy in this project so far has been time-series - "is gold going up?"
asked separately for each asset, then assembled. That framing has a structural
weakness the Phase 12 breadth calculation exposed: the assets share one dominant
factor, so seven time-series forecasts are close to one forecast repeated seven
times. When the common factor moves against you, every position is wrong at once.

Cross-sectional framing asks a different question - "is gold going up *more than*
Treasuries?" - and a dollar-neutral implementation of it is immune to the common
factor by construction. The Fundamental Law counts bets, and a long-short book
built on relative ranks generates bets that are far more independent than the
levels they came from. This is where most documented factor alpha actually lives,
and it is the one construction the project never tried.

Three implementations, because they answer different questions:

    long_short      dollar-neutral: long the top ranks, short the bottom, net
                    zero. Pure relative view, no market exposure, and the honest
                    test of whether the ranking carries information.
    tilt            benchmark plus a rank-proportional overlay. Keeps the market
                    exposure that actually earns the risk premium and uses the
                    ranking only to shade. This is what a real mandate looks like.
    rank_weight     weights proportional to rank, long-only. The mildest version,
                    and the one most robust to a bad signal.

The ranking itself is deliberately crude. With seven to twelve assets there is no
room for a clever weighting scheme, and cross-sectional demeaning already removes
most of the estimation error that destroyed the time-series approaches - a rank
does not care whether the forecast was 2% or 20%, only that it was the largest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_rank(signal: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """
    Rank assets against each other at each date, scaled to [0, 1].

    Ranking rather than using the raw signal is the point of the exercise. A
    forecast of +2.1% means nothing on its own - the question is whether it is
    the largest available, and a rank answers that without inheriting the
    forecast's calibration error.
    """
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
    Dollar-neutral: equal-weight the best `n_long`, short the worst `n_short`.

    Net exposure is zero by construction, so the common factor cannot move the
    book. `gross` sets the size of each side, and the result must be read as an
    overlay to be funded, not a standalone portfolio - it earns no risk premium.
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
    """
    Benchmark plus a rank-proportional overlay.

    The overlay sums to zero across assets, so the benchmark's market exposure is
    preserved exactly and the ranking only redistributes within it. That
    separation matters for attribution: any performance difference is the
    ranking's, not a side effect of having taken more or less market risk.
    """
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
    """
    Average several rankings into one.

    Combining at the *rank* level rather than the signal level is deliberate: it
    makes the composite invariant to the scale of each input, so a signal
    measured in basis points does not dominate one measured in standard
    deviations purely by units.
    """
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

    This is the IC in the Fundamental Law, measured directly rather than inferred
    from an R-squared. Its mean is the skill and its standard deviation gives the
    t-statistic on that skill - `mean / std * sqrt(n)` - which is the cleanest
    available test of whether a ranking carries information.
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
        # Grinold: IR = IC * sqrt(breadth). Reported so the theoretical ceiling
        # sits next to the realised result rather than in a separate document.
        "implied_ir": float(mean * np.sqrt(periods_per_year)) if np.isfinite(mean) else np.nan,
    }


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1)
