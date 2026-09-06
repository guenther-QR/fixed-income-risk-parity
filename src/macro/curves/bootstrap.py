"""Bootstrap a zero-coupon curve from Treasury constant-maturity par yields."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# FRED constant-maturity series and their maturities in years.
CMT_SERIES = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0,
    "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0, "DGS7": 7.0,
    "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0,
}

# Short maturities carried on the curve in addition to the semiannual coupon
# grid.
MONEY_MARKET_GRID = (1 / 12, 0.25)


def bootstrap_one(maturities: np.ndarray, par_yields: np.ndarray,
                  horizon: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap one day's curve."""
    ok = np.isfinite(par_yields) & np.isfinite(maturities)
    m, y = maturities[ok], par_yields[ok] / 100.0
    if m.size < 4:
        raise ValueError("need at least four quoted maturities to bootstrap")

    order = np.argsort(m)
    m, y = m[order], y[order]

    # The grid always runs to the full `horizon`, even on days whose longest
    # quote is shorter: par yields beyond the last quote are held flat.

    # Money-market points (under six months) are bills: zero-coupon, quoted
    # coupon-equivalent, so they discount at simple interest and take no part
    # in the coupon recursion.
    mm_grid = np.array([t for t in MONEY_MARKET_GRID if t < 0.5 - 1e-9])
    coupon_grid = np.arange(0.5, horizon + 1e-9, 0.5)
    grid = np.concatenate([mm_grid, coupon_grid])

    # Interpolate par yields onto the grid; hold the curve flat beyond the
    # quotes.
    interp = PchipInterpolator(m, y, extrapolate=False)
    par = interp(grid)
    par = np.where(np.isnan(par) & (grid < m[0]), y[0], par)
    par = np.where(np.isnan(par) & (grid > m[-1]), y[-1], par)

    disc = np.empty_like(grid)
    running = 0.0
    for i, (t, c) in enumerate(zip(grid, par)):
        if t < 0.5 - 1e-9:
            disc[i] = 1.0 / (1.0 + c * t)          # bill: simple interest
        elif t <= 0.5 + 1e-9:
            disc[i] = 1.0 / (1.0 + c / 2.0)        # one period to redemption
            running += disc[i]
        else:
            disc[i] = (1.0 - (c / 2.0) * running) / (1.0 + c / 2.0)
            running += disc[i]

    return grid, disc


def zero_from_discount(grid: np.ndarray, disc: np.ndarray,
                       continuous: bool = True) -> np.ndarray:
    """Convert discount factors to zero rates, in percent."""
    if continuous:
        return -np.log(disc) / grid * 100.0
    return ((1.0 / disc) ** (1.0 / (2.0 * grid)) - 1.0) * 200.0


def forward_from_discount(grid: np.ndarray, disc: np.ndarray) -> np.ndarray:
    """Six-month forward rates implied by the discount curve, in percent."""
    fwd = np.empty_like(grid)
    fwd[0] = (1.0 / disc[0] - 1.0) * 200.0
    fwd[1:] = (disc[:-1] / disc[1:] - 1.0) * 200.0
    return fwd


def bootstrap_panel(cmt: pd.DataFrame, horizon: float = 30.0,
                    continuous: bool = True) -> dict[str, pd.DataFrame]:
    """Bootstrap every day in a CMT panel."""
    mats = np.array([CMT_SERIES[c] for c in cmt.columns])
    grid = np.concatenate([
        np.array([t for t in MONEY_MARKET_GRID if t < 0.5 - 1e-9]),
        np.arange(0.5, min(horizon, float(mats.max())) + 1e-9, 0.5),
    ])

    zeros, discs, fwds, index, longest = [], [], [], [], []
    for date, row in cmt.iterrows():
        vals = row.to_numpy(dtype=float)
        ok = np.isfinite(vals)
        if ok.sum() < 4:
            continue
        try:
            g, d = bootstrap_one(mats, vals, horizon)
        except (ValueError, FloatingPointError):
            continue
        if g.size != grid.size or not np.all(np.isfinite(d)) or np.any(d <= 0):
            continue
        zeros.append(zero_from_discount(g, d, continuous))
        discs.append(d)
        fwds.append(forward_from_discount(g, d))
        index.append(date)
        longest.append(float(mats[ok].max()))

    idx = pd.DatetimeIndex(index, name="date")
    return {
        "zero": pd.DataFrame(zeros, index=idx, columns=grid),
        "discount": pd.DataFrame(discs, index=idx, columns=grid),
        "forward": pd.DataFrame(fwds, index=idx, columns=grid),
        # Longest genuinely quoted maturity each day.
        "max_quoted": pd.Series(longest, index=idx, name="max_quoted"),
    }


def interp_zero(zero_row: pd.Series, maturity: float) -> float:
    """Zero rate at an arbitrary maturity, interpolated along one day's curve."""
    g = zero_row.index.to_numpy(dtype=float)
    v = zero_row.to_numpy(dtype=float)
    ok = np.isfinite(v)
    return float(np.interp(maturity, g[ok], v[ok]))


def discount_at(disc_row: pd.Series, maturity: float) -> float:
    """Discount factor at an arbitrary maturity."""
    g = disc_row.index.to_numpy(dtype=float)
    d = disc_row.to_numpy(dtype=float)
    ok = np.isfinite(d) & (d > 0)
    return float(np.exp(np.interp(maturity, g[ok], np.log(d[ok]))))
