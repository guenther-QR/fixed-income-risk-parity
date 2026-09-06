"""Principal components of the yield curve: level, slope and curvature."""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAMES = ["level", "slope", "curvature"]


def _orient(loadings: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """
    Fix each component's arbitrary sign so the factors are comparable over
    time.
    """
    out = loadings.copy()
    if out.shape[0] >= 1 and out[0].mean() < 0:
        out[0] *= -1
    if out.shape[0] >= 2 and out[1][-1] < 0:
        out[1] *= -1
    if out.shape[0] >= 3:
        belly = np.argmin(np.abs(maturities - np.median(maturities)))
        if out[2][belly] < 0:
            out[2] *= -1
    return out


def fit(zero: pd.DataFrame, n_components: int = 3) -> dict:
    """Decompose daily zero-rate changes into principal components."""
    changes = zero.diff().dropna(how="any")
    if changes.empty:
        raise ValueError("no complete rows of curve changes to decompose")

    X = changes.to_numpy(dtype=float)
    X_centred = X - X.mean(axis=0)

    # SVD rather than an explicit covariance eigendecomposition: better
    # conditioned when maturities are as collinear as these are.
    _, sv, vt = np.linalg.svd(X_centred, full_matrices=False)
    variance = sv ** 2 / (len(X_centred) - 1)
    explained = variance / variance.sum()

    mats = zero.columns.to_numpy(dtype=float)
    loadings = _orient(vt[:n_components], mats)
    scores = X_centred @ loadings.T

    names = FACTOR_NAMES[:n_components] + [
        f"pc{i + 1}" for i in range(len(FACTOR_NAMES), n_components)
    ]
    return {
        "loadings": pd.DataFrame(loadings, index=names, columns=zero.columns),
        "scores": pd.DataFrame(scores, index=changes.index, columns=names),
        "explained": pd.Series(explained[:n_components], index=names),
        "explained_all": pd.Series(explained),
    }


def descriptive_factors(zero: pd.DataFrame) -> pd.DataFrame:
    """
    The textbook level/slope/curvature proxies, for comparison with the PCA.
    """
    def at(m: float) -> pd.Series:
        cols = zero.columns.to_numpy(dtype=float)
        return zero.iloc[:, int(np.argmin(np.abs(cols - m)))]

    s2, s10, s30 = at(2.0), at(10.0), at(30.0)
    return pd.DataFrame({
        "level": (s2 + s10 + s30) / 3.0,
        "slope": s10 - s2,
        "curvature": 2.0 * s10 - s2 - s30,
    })
