"""Bootstrap a zero-coupon curve from Treasury constant-maturity par yields.

Treasury CMT quotes (FRED DGS*) are *par* yields on a semiannual coupon basis: a
bond paying y/2 per period and redeeming at par prices to exactly 100. Bootstrapping
inverts that relationship maturity by maturity to recover discount factors, and from
them zero rates and forward rates.

For a par yield y at maturity T (semiannual, so N = 2T periods):

    100 = (100 y / 2) * sum_{i=1..N} D(t_i)  +  100 * D(T)

Given D at every earlier coupon date, D(T) follows directly:

    D(T) = (1 - (y/2) * sum_{i=1..N-1} D(t_i)) / (1 + y/2)

Quotes exist only at a handful of maturities, so par yields are first interpolated
onto the semiannual grid. Interpolation happens in par-yield space (monotone
PCHIP), which keeps the curve smooth and avoids the oscillation a cubic spline can
produce at the short end.

Maturities under six months have no coupon before redemption; those points are
treated as simple-interest money-market quotes.
"""
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

# Short maturities carried on the curve in addition to the semiannual coupon grid.
MONEY_MARKET_GRID = (1 / 12, 0.25)


def bootstrap_one(maturities: np.ndarray, par_yields: np.ndarray,
                  horizon: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap one day's curve.

    `par_yields` are in percent at `maturities` (years). Returns the semiannual
    grid and the discount factor at each grid point.
    """
    ok = np.isfinite(par_yields) & np.isfinite(maturities)
    m, y = maturities[ok], par_yields[ok] / 100.0
    if m.size < 4:
        raise ValueError("need at least four quoted maturities to bootstrap")

    order = np.argsort(m)
    m, y = m[order], y[order]

    # The grid always runs to the full `horizon`, even on days whose longest quote
    # is shorter: par yields beyond the last quote are held flat. Truncating the
    # horizon per day instead would give each day a different grid length, and
    # `bootstrap_panel` would then silently drop every day that did not match the
    # panel-wide grid - which discarded everything before the 30-year's 1977
    # inception. Callers are told how far the real quotes reached via
    # `bootstrap_panel`'s `max_quoted`, so extrapolated regions can be refused.

    # Money-market points (under six months) are bills: zero-coupon, quoted
    # coupon-equivalent, so they discount at simple interest and take no part in
    # the coupon recursion. Excluding them would leave the curve undefined below
    # six months, and interpolating down to one month from a semiannual grid
    # silently returns the six-month rate instead.
    mm_grid = np.array([t for t in MONEY_MARKET_GRID if t < 0.5 - 1e-9])
    coupon_grid = np.arange(0.5, horizon + 1e-9, 0.5)
    grid = np.concatenate([mm_grid, coupon_grid])

    # Interpolate par yields onto the grid; hold the curve flat beyond the quotes.
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
    """
    Bootstrap every day in a CMT panel.

    `cmt` has FRED series ids as columns and dates as the index. Returns
    {"zero": ..., "discount": ..., "forward": ...} keyed by maturity in years.
    """
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
        # Longest genuinely quoted maturity each day. Anything beyond this on the
        # curve is flat extrapolation, not observation.
        "max_quoted": pd.Series(longest, index=idx, name="max_quoted"),
    }


def interp_zero(zero_row: pd.Series, maturity: float) -> float:
    """Zero rate at an arbitrary maturity, interpolated along one day's curve."""
    g = zero_row.index.to_numpy(dtype=float)
    v = zero_row.to_numpy(dtype=float)
    ok = np.isfinite(v)
    return float(np.interp(maturity, g[ok], v[ok]))


def discount_at(disc_row: pd.Series, maturity: float) -> float:
    """Discount factor at an arbitrary maturity.

    Interpolates in log-discount space, which is linear in the zero rate and so
    keeps implied forwards well behaved between grid points.
    """
    g = disc_row.index.to_numpy(dtype=float)
    d = disc_row.to_numpy(dtype=float)
    ok = np.isfinite(d) & (d > 0)
    return float(np.exp(np.interp(maturity, g[ok], np.log(d[ok]))))
