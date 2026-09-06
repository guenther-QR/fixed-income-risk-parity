"""
Regimes from a dynamic factor model rather than from two hand-picked series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.fred import get_series
from .schemes import _label

# Indicator blocks.
GROWTH_BLOCK = {
    "INDPRO": "yoy",         # industrial production
    "PAYEMS": "yoy",         # nonfarm payrolls
    "UNRATE": "neg_diff12",  # unemployment, inverted so up means good
    "HOUST": "yoy",          # housing starts
    "PERMIT": "yoy",         # building permits
    "MANEMP": "yoy",         # manufacturing employment
    "AWHMAN": "diff12",      # manufacturing weekly hours
    "TCU": "diff12",         # capacity utilisation
    "UMCSENT": "yoy",        # consumer sentiment
    "DSPIC96": "yoy",        # real disposable income
}

PRICE_BLOCK = {
    "CPIAUCSL": "yoy",       # headline CPI
    "CPILFESL": "yoy",       # core CPI
    "PPIACO": "yoy",         # producer prices, all commodities
    "PCEPI": "yoy",          # PCE deflator
    "PCEPILFE": "yoy",       # core PCE
    "AHETPI": "yoy",         # average hourly earnings
    "CPIENGSL": "yoy",       # energy CPI
    "CPIUFDSL": "yoy",       # food CPI
}


def _transform(s: pd.Series, how: str) -> pd.Series:
    if how == "yoy":
        return s / s.shift(12) - 1
    if how == "diff12":
        return s.diff(12)
    if how == "neg_diff12":
        return -s.diff(12)
    raise ValueError(how)


def build_block(spec: dict[str, str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch and transform one indicator block, lagged one month for
    publication.
    """
    cols = {}
    for sid, how in spec.items():
        try:
            raw = get_series(sid).resample("ME").last()
        except Exception:
            continue
        cols[sid] = _transform(raw, how)
    if not cols:
        raise RuntimeError("no indicators available")
    return pd.DataFrame(cols).shift(1).reindex(index)


def recursive_pc1(panel: pd.DataFrame, min_obs: int = 60,
                  min_coverage: float = 0.6) -> pd.Series:
    """
    First principal component, re-estimated each month on data available
    then.
    """
    out = pd.Series(index=panel.index, dtype=float)

    for i in range(min_obs, len(panel)):
        hist = panel.iloc[: i + 1]
        ok = hist.columns[hist.notna().mean() >= min_coverage]
        block = hist[ok].dropna()
        if len(block) < min_obs or block.shape[1] < 2:
            continue

        mu, sd = block.mean(), block.std()
        Z = ((block - mu) / sd.replace(0, np.nan)).dropna(axis=1, how="any")
        if Z.shape[1] < 2:
            continue

        # Eigendecomposition of the covariance of standardised data.
        vals, vecs = np.linalg.eigh(np.cov(Z.to_numpy(), rowvar=False))
        w = vecs[:, -1]

        if w.sum() < 0:              # anchor: loadings should point with the block
            w = -w

        score = Z.to_numpy() @ w
        # Reported in units of its own standard deviation so the factor stays
        # comparable across dates as the panel widens.
        out.iloc[i] = float(score[-1] / (np.std(score) + 1e-12))

    return out


def recursive_pc12(panel: pd.DataFrame, min_obs: int = 60,
                   min_coverage: float = 0.6) -> pd.DataFrame:
    """
    First two components of a joint panel, with signs anchored to reference
    blocks.
    """
    rows = {}

    for i in range(min_obs, len(panel)):
        hist = panel.iloc[: i + 1]
        ok = hist.columns[hist.notna().mean() >= min_coverage]
        block = hist[ok].dropna()
        if len(block) < min_obs or block.shape[1] < 4:
            continue

        Z = (block - block.mean()) / block.std().replace(0, np.nan)
        Z = Z.dropna(axis=1, how="any")
        if Z.shape[1] < 4:
            continue

        vals, vecs = np.linalg.eigh(np.cov(Z.to_numpy(), rowvar=False))
        order = np.argsort(vals)[::-1][:2]
        cols = list(Z.columns)
        scores = Z.to_numpy() @ vecs[:, order]

        loads = []
        for j in range(2):
            v = vecs[:, order[j]]
            g_abs = sum(abs(v[k]) for k, c in enumerate(cols) if c in GROWTH_BLOCK)
            p_abs = sum(abs(v[k]) for k, c in enumerate(cols) if c in PRICE_BLOCK)
            g_sgn = np.sign(sum(v[k] for k, c in enumerate(cols)
                                if c in GROWTH_BLOCK) or 1.0)
            p_sgn = np.sign(sum(v[k] for k, c in enumerate(cols)
                                if c in PRICE_BLOCK) or 1.0)
            loads.append(("growth", g_sgn) if g_abs > p_abs else ("inflation", p_sgn))

        if loads[0][0] == loads[1][0]:
            continue                              # both on one block: no quadrant

        rec = {"explained": float(vals[order].sum() / vals.sum())}
        for j, (kind, sign) in enumerate(loads):
            s = scores[:, j] * sign
            rec[kind] = float(s[-1] / (np.std(s) + 1e-12))
        rows[panel.index[i]] = rec

    return pd.DataFrame(rows).T.reindex(panel.index)


def block_factors(index: pd.DatetimeIndex, min_obs: int = 60) -> pd.DataFrame:
    """Growth and inflation diffusion indices, each the PC1 of its own block."""
    g = build_block(GROWTH_BLOCK, index)
    p = build_block(PRICE_BLOCK, index)
    return pd.DataFrame({"growth": recursive_pc1(g, min_obs),
                         "inflation": recursive_pc1(p, min_obs)}, index=index)


def joint_factors(index: pd.DatetimeIndex, min_obs: int = 60) -> pd.DataFrame:
    """Growth and inflation read off the first two components of one panel."""
    panel = pd.concat([build_block(GROWTH_BLOCK, index),
                       build_block(PRICE_BLOCK, index)], axis=1)
    return recursive_pc12(panel, min_obs)


def classify_factors(f: pd.DataFrame, growth_cut: float = 0.0,
                     infl_cut: float = 0.0) -> pd.Series:
    """Quadrants from factor scores."""
    out = _label(f["growth"] > growth_cut, f["inflation"] > infl_cut)
    out[f["growth"].isna() | f["inflation"].isna()] = np.nan
    return out


def variance_explained(index: pd.DatetimeIndex) -> pd.DataFrame:
    """How much of each block a single factor accounts for, on the full sample."""
    rows = {}
    for name, spec in [("growth", GROWTH_BLOCK), ("inflation", PRICE_BLOCK)]:
        panel = build_block(spec, index).dropna()
        if panel.empty:
            continue
        Z = (panel - panel.mean()) / panel.std()
        vals = np.linalg.eigvalsh(np.cov(Z.to_numpy(), rowvar=False))[::-1]
        rows[name] = {"n_series": panel.shape[1], "n_months": len(panel),
                      "pc1_share": vals[0] / vals.sum(),
                      "pc2_share": vals[1] / vals.sum()}
    return pd.DataFrame(rows).T
