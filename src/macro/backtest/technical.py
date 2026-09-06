"""Shared machinery for signal-driven portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PPY = 252
MONTH = 21
SKIP = 5
MIN_IC_OBS = 252
WINDOWS = [21, 42, 63, 126, 189, 252, 378, 504]

# Modified duration per holding, used to report what a portfolio is actually
# exposed to rather than only what it returned.
DUR = {
    # constant maturity Treasuries, computed from the curve
    "ust2y": 1.93, "ust5y": 4.45, "ust10y": 7.79, "ust30y": 14.73,
    # funds, average duration as published by the fund company
    "ig_short": 2.7,    # VFSTX, Vanguard Short-Term Investment-Grade
    "ig": 5.93,         # FBNDX, Fidelity Investment Grade Bond
    "ig_long": 11.8,    # VWESX, Vanguard Long-Term Investment-Grade
    "hy": 2.9,          # VWEHX, Vanguard High-Yield Corporate
    "mbs": 6.3,         # VFIIX, Vanguard GNMA
    "muni": 5.8,        # VWITX, Vanguard Intermediate-Term Tax-Exempt
    "muni_hy": 8.2,     # VWAHX, Vanguard High-Yield Tax-Exempt
}


# ------------------------------------------------------------------ signals

def xs_z(F: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score, so a signal says who looks good today."""
    return F.sub(F.mean(axis=1), axis=0).div(
        F.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def momentum_candidates(r: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every momentum window an adaptive strategy may choose from."""
    out = {}
    for w in WINDOWS:
        out[f"mom{w}"] = r.rolling(w).mean().shift(SKIP) * w
        out[f"momsharpe{w}"] = (r.rolling(w).mean()
                                / r.rolling(w).std().replace(0, np.nan)).shift(SKIP)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation on two aligned arrays with the NaNs already removed."""
    if len(x) < MIN_IC_OBS:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def rolling_signal(r, cand, Zc, assets, idx, burn_in, window, horizon=1):
    """Per-asset momentum window, re-chosen monthly on a trailing window."""
    keys = list(cand)
    if horizon == 1:
        fwd = {a: r[a].shift(-1).to_numpy() for a in assets}
    else:
        f = r.rolling(horizon).sum().shift(-horizon)
        fwd = {a: f[a].to_numpy() for a in assets}
    sig = {k: {a: cand[k][a].to_numpy() for a in assets} for k in keys}
    rows, dates = [], []
    for i in range(len(idx)):
        if i % MONTH:
            continue
        g = burn_in + i
        lo, hi = max(0, g - window), g - horizon
        row = {}
        for a in assets:
            yv = fwd[a][lo:hi]
            best, best_ic = keys[0], -np.inf
            for k in keys:
                xv = sig[k][a][lo:hi]
                m = np.isfinite(xv) & np.isfinite(yv)
                if m.sum() < MIN_IC_OBS:
                    continue
                ic = spearman(xv[m], yv[m])
                if np.isfinite(ic) and ic > best_ic:
                    best, best_ic = k, ic
            row[a] = best
        rows.append(row)
        dates.append(idx[i])
    CH = pd.DataFrame(rows, index=dates).reindex(idx).ffill()
    return pd.DataFrame({a: [Zc[CH.loc[t, a]].at[t, a] for t in idx]
                         for a in assets}, index=idx)


# ------------------------------------------------------------------ weights

def long_only(Z: pd.DataFrame, n: int) -> pd.DataFrame:
    """Positive part of the score, normalised. Long only and fully invested."""
    W = Z.clip(lower=0.0)
    return W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / n)


def long_short(Z: pd.DataFrame) -> pd.DataFrame:
    """Dollar neutral: z is already demeaned, so rescale to one per side."""
    pos = Z.clip(lower=0.0).sum(axis=1).replace(0, np.nan)
    neg = (-Z.clip(upper=0.0)).sum(axis=1).replace(0, np.nan)
    return Z.div(pd.concat([pos, neg], axis=1).max(axis=1), axis=0).fillna(0.0)


def rank_weights(S: pd.DataFrame) -> pd.DataFrame:
    """Asness, Moskowitz and Pedersen equation (1)."""
    R = S.rank(axis=1, na_option="keep")
    dev = R.sub(R.mean(axis=1), axis=0)
    pos = dev.clip(lower=0.0).sum(axis=1).replace(0, np.nan)
    return dev.div(pos, axis=0).fillna(0.0)


def overlay(base: pd.DataFrame, Z: pd.DataFrame, lam: float,
            cap: float | None = None) -> pd.DataFrame:
    """Move each base weight by lambda times its score times that weight."""
    W = (base + lam * Z.reindex(base.index) * base).clip(lower=0.0)
    if cap is not None:
        W = W.clip(upper=base + cap)
    return W.div(W.sum(axis=1), axis=0)


# ---------------------------------------------------------------- backtests

def run_long(W, r, rates, every):
    """Hold a long-only book, rebalancing every `every` days, net of costs."""
    idx = W.index
    R = r.reindex(idx).to_numpy()
    Wt, n = W.to_numpy(), W.shape[1]
    nets, held, traded = [], None, 0.0
    for i in range(len(idx)):
        if i % every == 0 or held is None:
            t = Wt[i]
            t = np.ones(n) / n if not np.isfinite(t).all() or t.sum() <= 0 \
                else t / t.sum()
            pre = np.zeros(n) if held is None else held
            traded += 0.0 if pre.sum() == 0 else float(np.abs(t - pre).sum()) / 2
            cost = float(np.abs(t - pre) @ rates)
            held = t
        else:
            cost = 0.0
        nets.append(float(R[i] @ held) - cost)
        g = held * (1 + R[i])
        held = g / g.sum() if g.sum() > 0 else held
    return pd.Series(nets, index=idx), traded / (len(idx) / PPY)


def run_ls(W, r, rates, every):
    """Hold a spread, resetting to target every `every` days, net of costs."""
    idx = W.index
    R = r.reindex(idx).to_numpy()
    Wt = W.to_numpy()
    nets, held, traded = [], np.zeros(W.shape[1]), 0.0
    for i in range(len(idx)):
        cost = 0.0
        if i % every == 0:
            t = Wt[i]
            if np.isfinite(t).all():
                cost = float(np.abs(t - held) @ rates)
                traded += float(np.abs(t - held).sum()) / 2.0
                held = t
        nets.append(float(R[i] @ held) - cost)
    return pd.Series(nets, index=idx), traded / (len(idx) / PPY)


# ---------------------------------------------------------------- inference

def nw_ols(y, Xm, lags=None):
    """OLS with Newey-West standard errors."""
    y = np.asarray(y, dtype=float)
    Xm = np.asarray(Xm, dtype=float)
    if lags is None:
        lags = max(1, int(4.0 * (len(y) / 100.0) ** (2.0 / 9.0)))
    lags = min(lags, max(1, len(y) // 4))
    X = np.column_stack([np.ones(len(y)), Xm])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 1e-18))
    return b, b / se
