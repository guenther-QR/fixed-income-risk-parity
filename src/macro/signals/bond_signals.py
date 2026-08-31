"""Predictors of bond excess returns, under the physical measure.

A risk-neutral model cannot forecast excess returns - that is its defining
property, not a shortcoming. Everything here is estimated under P and is
knowable at the time it is used.

    carry_roll     coupon income plus the gain from sliding down the curve
    cp_factor      Cochrane-Piazzesi (2005): a single linear combination of forward
                   rates that predicts excess returns at every maturity. See the
                   note in `cochrane_piazzesi` - the loadings here do not reproduce
                   the paper's tent shape
    fb_spread      Fama-Bliss (1987): the forward rate minus the spot short rate
    term_premium   from our own affine model, and the NY Fed's published series
    level/slope/curvature   the curve's principal components

All are aligned to be usable at time t to predict returns from t to t+h, with no
look-ahead. The Cochrane-Piazzesi factor is the one that needs care: it is
estimated by regression, so in a backtest it must be re-estimated on an expanding
window rather than fitted once on the full sample. `cochrane_piazzesi_expanding`
does that; `cochrane_piazzesi` fits in-sample and is for reporting only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_rates(zero: pd.DataFrame, maturities=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """
    One-year forward rates f(t, n-1, n), from continuously compounded zeros.

        f(n-1, n) = n * y(n) - (n-1) * y(n-1)
    """
    grid = zero.columns.to_numpy(dtype=float)
    vals = zero.to_numpy(dtype=float)

    def y(m: float) -> np.ndarray:
        return np.array([np.interp(m, grid, row) for row in vals])

    out = {}
    for n in maturities:
        out[f"f{n}"] = n * y(n) - (n - 1) * y(n - 1) if n > 1 else y(1)
    return pd.DataFrame(out, index=zero.index)


def excess_returns(zero: pd.DataFrame, maturities=(2, 3, 4, 5),
                   horizon_months: int = 12) -> pd.DataFrame:
    """
    Realized log excess returns over the next `horizon_months`, by maturity.

    rx(n)_{t+1} = n*y_t(n) - (n-1)*y_{t+1}(n-1) - y_t(1), the standard
    Fama-Bliss construction. Indexed at t, so the value at t is the return
    *earned over the following horizon* - shift before using as a regressor.
    """
    grid = zero.columns.to_numpy(dtype=float)
    vals = zero.to_numpy(dtype=float) / 100.0

    def y(m: float) -> pd.Series:
        return pd.Series([np.interp(m, grid, row) for row in vals], index=zero.index)

    h = horizon_months
    y1 = y(1.0)
    out = {}
    for n in maturities:
        out[f"rx{n}"] = (n * y(n) - (n - 1) * y(n - 1).shift(-h) - y1).rename(f"rx{n}")
    return pd.DataFrame(out)


def cochrane_piazzesi(zero: pd.DataFrame, maturities=(2, 3, 4, 5),
                      horizon_months: int = 12) -> dict:
    """
    The Cochrane-Piazzesi single-factor predictor, fitted in sample.

    Regress the *average* excess return across maturities on the forward rates;
    the fitted value is the factor. A single number then carries most of the
    predictability at every maturity.

    On this project's sample the loadings do NOT reproduce the tent shape of the
    original paper. Cochrane and Piazzesi report a single-peaked pattern rising
    then falling across the forward curve; fitted on the 1990-2026 bootstrapped
    curve here, the loadings have two turning points and the R-squared is 22.5%
    against their roughly 35%. Two plausible reasons, neither verified: the sample
    is entirely post-Volcker and includes the zero-rate years, and the underlying
    curve is bootstrapped from constant-maturity quotes rather than taken from the
    Fama-Bliss discount bond file the paper uses. Treat the factor as a working
    predictor on this data, not as a replication of the published result.

    In sample only. Use `cochrane_piazzesi_expanding` inside a backtest.
    """
    fwd = forward_rates(zero, maturities=(1, 2, 3, 4, 5))
    rx = excess_returns(zero, maturities, horizon_months)
    avg = rx.mean(axis=1)

    d = pd.concat([avg.rename("y"), fwd], axis=1).dropna()
    A = np.column_stack([np.ones(len(d)), d[fwd.columns].to_numpy()])
    gamma, *_ = np.linalg.lstsq(A, d["y"].to_numpy(), rcond=None)

    full = np.column_stack([np.ones(len(fwd)), fwd.to_numpy()])
    factor = pd.Series(full @ gamma, index=fwd.index, name="cp_factor")

    fitted = A @ gamma
    ss_res = float(((d["y"].to_numpy() - fitted) ** 2).sum())
    ss_tot = float(((d["y"].to_numpy() - d["y"].mean()) ** 2).sum())
    return {"factor": factor, "gamma": gamma,
            "r2": 1 - ss_res / ss_tot, "forwards": fwd}


def cochrane_piazzesi_expanding(zero: pd.DataFrame, maturities=(2, 3, 4, 5),
                                horizon_months: int = 12,
                                min_obs: int = 120) -> pd.Series:
    """
    The CP factor re-estimated on an expanding window - the backtest-safe version.

    At each date the loadings come only from data whose outcome was already
    observed, which means excluding the last `horizon_months` of returns as well
    as everything after t. Fitting CP once on the full sample and using it
    throughout is a subtle but serious look-ahead: the tent shape itself was
    learned from the future.
    """
    fwd = forward_rates(zero, maturities=(1, 2, 3, 4, 5))
    rx = excess_returns(zero, maturities, horizon_months).mean(axis=1)

    out = pd.Series(index=fwd.index, dtype=float, name="cp_factor_oos")
    for i in range(min_obs, len(fwd)):
        usable = fwd.index[:i - horizon_months]        # outcome already realised
        if len(usable) < min_obs:
            continue
        d = pd.concat([rx.rename("y"), fwd], axis=1).loc[usable].dropna()
        if len(d) < min_obs:
            continue
        A = np.column_stack([np.ones(len(d)), d[fwd.columns].to_numpy()])
        gamma, *_ = np.linalg.lstsq(A, d["y"].to_numpy(), rcond=None)
        out.iloc[i] = float(np.r_[1.0, fwd.iloc[i].to_numpy()] @ gamma)
    return out


def fama_bliss_spreads(zero: pd.DataFrame, maturities=(2, 3, 4, 5)) -> pd.DataFrame:
    """Forward rate minus the one-year spot rate - the classic Fama-Bliss predictor."""
    fwd = forward_rates(zero, maturities=(1, 2, 3, 4, 5))
    return pd.DataFrame(
        {f"fb{n}": fwd[f"f{n}"] - fwd["f1"] for n in maturities}, index=zero.index
    )


def build_signal_panel(zero: pd.DataFrame, bond_returns: dict[float, pd.DataFrame],
                       term_premium: pd.Series | None = None,
                       nyfed_tp: pd.Series | None = None,
                       freq: str = "ME") -> pd.DataFrame:
    """
    Assemble every bond predictor into one panel, resampled to `freq`.

    Everything returned is knowable at the timestamp it carries.
    """
    z = zero.resample(freq).last().dropna(how="any")
    cp = cochrane_piazzesi(z)

    cols: dict[str, pd.Series] = {
        "cp_factor": cp["factor"],
        "level": z[[c for c in z.columns if 1.5 <= float(c) <= 30]].mean(axis=1),
        "slope": z[min(z.columns, key=lambda c: abs(float(c) - 10))]
                 - z[min(z.columns, key=lambda c: abs(float(c) - 0.5))],
    }
    ten = z[min(z.columns, key=lambda c: abs(float(c) - 10))]
    two = z[min(z.columns, key=lambda c: abs(float(c) - 2))]
    thirty = z[min(z.columns, key=lambda c: abs(float(c) - 30))]
    cols["curvature"] = 2 * ten - two - thirty

    cols |= {c: s for c, s in fama_bliss_spreads(z).items()}

    for m, r in bond_returns.items():
        cols[f"carry_roll_{m:g}y"] = (r["carry"] + r["rolldown"]).resample(freq).sum() * 12

    if term_premium is not None:
        cols["term_premium_ours"] = term_premium.resample(freq).last()
    if nyfed_tp is not None:
        cols["term_premium_nyfed"] = nyfed_tp.resample(freq).last()

    return pd.DataFrame(cols).dropna(how="all")
