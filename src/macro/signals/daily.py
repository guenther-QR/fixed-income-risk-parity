"""Daily signal library."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.fred import get_series

# Calendar days between the period a statistic describes and its public
# release.
PUBLICATION_LAG = {
    "CPIAUCSL": 45, "CPILFESL": 45, "PCEPI": 60, "INDPRO": 45,
    "PAYEMS": 35, "UNRATE": 35, "HOUST": 50, "PERMIT": 50,
    "UMCSENT": 20, "M2SL": 45, "AWHMAN": 35, "TCU": 45,
    "ICSA": 5, "NFCI": 8, "ANFCI": 8,
}

# Series whose daily returns are contaminated by stale marks. Momentum on these
# is not tradeable at daily frequency.
STALE_ASSETS = ["hy", "ig"]

MOM_WINDOWS = [5, 21, 63, 126, 252]
REV_WINDOWS = [1, 5]
MA_PAIRS = [(5, 21), (10, 50), (21, 126), (50, 200)]


def _lagged(sid: str, index: pd.DatetimeIndex) -> pd.Series:
    """A FRED series as a daily step function, with its publication lag applied."""
    s = get_series(sid)
    lag = PUBLICATION_LAG.get(sid, 30)
    s.index = s.index + pd.Timedelta(days=lag)
    return s.reindex(index, method="ffill")


def curve_signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Level, slope and curvature from daily constant-maturity yields."""
    out = pd.DataFrame(index=index)
    y = {}
    for sid, name in [("DGS3MO", "y3m"), ("DGS2", "y2"), ("DGS5", "y5"),
                      ("DGS10", "y10"), ("DGS30", "y30")]:
        try:
            y[name] = get_series(sid).reindex(index).ffill() / 100.0
        except Exception:
            continue
    if "y10" in y and "y2" in y:
        out["d_slope_2s10s"] = y["y10"] - y["y2"]
    if "y10" in y and "y3m" in y:
        out["d_slope_10y3m"] = y["y10"] - y["y3m"]
    if "y30" in y and "y10" in y:
        out["d_slope_10s30s"] = y["y30"] - y["y10"]
    if "y10" in y:
        out["d_level"] = y["y10"]
    if all(k in y for k in ["y2", "y5", "y10"]):
        out["d_curvature"] = 2 * y["y5"] - y["y2"] - y["y10"]

    for c in list(out.columns):
        out[f"{c}_chg5"] = out[c].diff(5)
        out[f"{c}_chg21"] = out[c].diff(21)
        roll = out[c].rolling(252, min_periods=120)
        out[f"{c}_z"] = (out[c] - roll.mean()) / roll.std()
    return out


def credit_signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily Aaa and Baa yields; the spread is the tradeable credit signal."""
    out = pd.DataFrame(index=index)
    try:
        aaa = get_series("DAAA").reindex(index).ffill() / 100.0
        baa = get_series("DBAA").reindex(index).ffill() / 100.0
    except Exception:
        return out
    out["d_baa_aaa"] = baa - aaa
    out["d_baa_10y"] = baa - get_series("DGS10").reindex(index).ffill() / 100.0
    for c in list(out.columns):
        out[f"{c}_chg5"] = out[c].diff(5)
        out[f"{c}_chg21"] = out[c].diff(21)
        roll = out[c].rolling(252, min_periods=120)
        out[f"{c}_z"] = (out[c] - roll.mean()) / roll.std()
    return out


def volatility_signals(index: pd.DatetimeIndex, returns: pd.DataFrame) -> pd.DataFrame:
    """Implied and realised volatility, and the gap between them."""
    out = pd.DataFrame(index=index)
    try:
        vix = get_series("VIXCLS").reindex(index).ffill()
        out["d_vix"] = vix
        out["d_vix_chg5"] = vix.diff(5)
        roll = vix.rolling(252, min_periods=120)
        out["d_vix_z"] = (vix - roll.mean()) / roll.std()
        rv = returns["sp500"].rolling(21, min_periods=15).std() * np.sqrt(252) * 100
        out["d_vrp"] = vix - rv
        out["d_vrp_z"] = (out["d_vrp"] - out["d_vrp"].rolling(252, min_periods=120).mean()
                          ) / out["d_vrp"].rolling(252, min_periods=120).std()
    except Exception:
        pass
    try:
        vix3m = get_series("VXVCLS").reindex(index).ffill()
        out["d_vix_term"] = vix3m / vix.replace(0, np.nan)
    except Exception:
        pass

    for a in returns.columns:
        for w in [21, 63]:
            rv = returns[a].rolling(w, min_periods=w // 2).std() * np.sqrt(252)
            out[f"d_rvol{w}_{a}"] = rv
            roll = rv.rolling(252, min_periods=120)
            out[f"d_rvol{w}_{a}_z"] = (rv - roll.mean()) / roll.std()
    return out


def momentum_signals(returns: pd.DataFrame) -> pd.DataFrame:
    """Trend at daily horizons, skipping the most recent five days."""
    out = pd.DataFrame(index=returns.index)
    for a in returns.columns:
        cum = (1 + returns[a]).cumprod()
        for w in MOM_WINDOWS:
            out[f"d_mom{w}_{a}"] = cum.pct_change(w)
            if w > 21:
                out[f"d_mom{w}s5_{a}"] = cum.shift(5).pct_change(w - 5)
        for s, l in MA_PAIRS:
            short = cum.rolling(s, min_periods=s).mean()
            long = cum.rolling(l, min_periods=l).mean()
            out[f"d_ma{s}_{l}_{a}"] = (short / long - 1.0)
    return out


def reversal_signals(returns: pd.DataFrame) -> pd.DataFrame:
    """Very short horizon returns, expected to carry a negative sign."""
    out = pd.DataFrame(index=returns.index)
    for a in returns.columns:
        cum = (1 + returns[a]).cumprod()
        for w in REV_WINDOWS:
            out[f"d_rev{w}_{a}"] = -cum.pct_change(w)
    return out


def macro_signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly and weekly macro as correctly lagged step functions."""
    out = pd.DataFrame(index=index)
    for sid, name in [("CPIAUCSL", "d_cpi"), ("CPILFESL", "d_core_cpi"),
                      ("INDPRO", "d_indpro"), ("PAYEMS", "d_payems")]:
        try:
            s = _lagged(sid, index)
            out[f"{name}_yoy"] = s / s.shift(252) - 1
        except Exception:
            continue
    for sid, name in [("UNRATE", "d_unrate"), ("UMCSENT", "d_sentiment"),
                      ("NFCI", "d_nfci"), ("ANFCI", "d_anfci")]:
        try:
            s = _lagged(sid, index)
            out[name] = s
            out[f"{name}_chg"] = s.diff(21)
        except Exception:
            continue
    try:
        claims = _lagged("ICSA", index)
        out["d_claims"] = claims
        out["d_claims_z"] = ((claims - claims.rolling(252, min_periods=120).mean())
                             / claims.rolling(252, min_periods=120).std())
    except Exception:
        pass
    try:
        oil = get_series("DCOILWTICO").reindex(index).ffill()
        out["d_oil_mom63"] = oil.pct_change(63)
        out["d_oil_z"] = ((oil - oil.rolling(252, min_periods=120).mean())
                          / oil.rolling(252, min_periods=120).std())
    except Exception:
        pass
    return out


def cross_asset(returns: pd.DataFrame) -> pd.DataFrame:
    """Relationships between assets rather than properties of one."""
    out = pd.DataFrame(index=returns.index)
    if {"sp500", "ust10y"} <= set(returns.columns):
        out["d_stock_bond_corr"] = (returns["sp500"]
                                    .rolling(126, min_periods=60)
                                    .corr(returns["ust10y"]))
        out["d_stock_bond_corr_chg"] = out["d_stock_bond_corr"].diff(21)
    if {"gold", "sp500"} <= set(returns.columns):
        out["d_gold_vs_eq"] = ((1 + returns["gold"]).rolling(126).apply(np.prod, raw=True)
                               - (1 + returns["sp500"]).rolling(126).apply(np.prod, raw=True))
    if {"hy", "ig"} <= set(returns.columns):
        out["d_hy_vs_ig"] = ((1 + returns["hy"]).rolling(63).apply(np.prod, raw=True)
                             - (1 + returns["ig"]).rolling(63).apply(np.prod, raw=True))
    return out


def build(returns: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """
    Every daily signal, lagged so it is knowable before the return it
    predicts.
    """
    idx = returns.index
    blocks = [curve_signals(idx), credit_signals(idx),
              volatility_signals(idx, returns), momentum_signals(returns),
              reversal_signals(returns), macro_signals(idx), cross_asset(returns)]
    X = pd.concat([b for b in blocks if not b.empty], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]

    stale_cols = [c for c in X.columns
                  if any(c.endswith(f"_{a}") for a in STALE_ASSETS)]
    out = X.shift(lag_days)
    out[stale_cols] = X[stale_cols].shift(lag_days + 1)
    return out


def describe(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "coverage": X.notna().mean(),
        "first": [X[c].first_valid_index() for c in X.columns],
        "mean": X.mean(), "std": X.std(),
        "autocorr_1": [X[c].autocorr(1) for c in X.columns],
    })
