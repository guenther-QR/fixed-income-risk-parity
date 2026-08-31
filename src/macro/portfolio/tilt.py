"""Bounded tilts: turning return forecasts into an allocation.

The obvious move - feed forecasts into a mean-variance optimizer as expected
returns - is the wrong one. Mean-variance is notoriously hypersensitive to
expected-return error: small changes in mu produce large swings in weights, and
the forecasts available here have out-of-sample R-squared around 1%. Optimizing
on them would amplify noise into turnover.

A bounded tilt fails gracefully instead. Start from a base allocation that needs
no forecast at all, deviate from it in proportion to signal strength, and cap how
far the deviation can go:

    w = clip( w_base + Lambda * z,  bounds )

where z is a cross-sectionally standardized forecast. If the forecasts are
worthless the portfolio stays near its base and loses only transaction costs; if
they carry information the tilt captures part of it. The asymmetry is deliberate:
the downside of a useless forecast is bounded by construction rather than by luck.

Structural constraints from the Phase 5 dependence map are applied on top, and
are design decisions with stated reasons rather than fitted parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TiltConfig:
    """Pre-committed in config/phase6.yaml; not tuned on results."""
    strength: float = 0.5                    # Lambda
    max_deviation_per_asset: float = 0.15
    max_total_deviation: float = 0.40
    long_only: bool = True
    allow_cash: bool = True
    structural_caps: dict[str, float] = field(default_factory=dict)
    grouped_caps: list[tuple[list[str], float]] = field(default_factory=list)


def cross_sectional_z(forecast: np.ndarray) -> np.ndarray:
    """
    Standardize forecasts across assets at a point in time.

    Cross-sectional rather than time-series, because the tilt is a relative
    decision: what matters is which asset looks better than the others now, not
    whether all of them look better than usual. A time-series z-score would push
    the whole portfolio into cash whenever the general level of forecasts fell,
    which is a market-timing bet the tilt is not meant to take.
    """
    f = np.asarray(forecast, dtype=float)
    if not np.isfinite(f).any():
        return np.zeros_like(f)
    mu = np.nanmean(f)
    sd = np.nanstd(f)
    if sd < 1e-12:
        return np.zeros_like(f)
    return np.nan_to_num((f - mu) / sd)


def apply_tilt(base: np.ndarray, forecast: np.ndarray, assets: list[str],
               cfg: TiltConfig) -> np.ndarray:
    """
    Tilt `base` toward assets with higher forecasts, subject to every bound.

    Order matters, and the structural caps must come **last**. Any operation that
    rescales weights afterwards - the deviation budget, a cash normalization -
    pulls a capped asset back toward its base and silently reopens the breach.
    An earlier version applied caps before the budget, and a 10% cap on high
    yield came out at 20.9%.

    A structural cap may itself create deviation from base, when the base
    allocation already exceeds the cap. That is intended: the cap is a hard
    constraint from the Phase 5 dependence analysis, and it overrides the
    deviation budget rather than negotiating with it.
    """
    base = np.asarray(base, dtype=float)
    z = cross_sectional_z(forecast)
    dev = cfg.strength * z * cfg.max_deviation_per_asset

    dev = np.clip(dev, -cfg.max_deviation_per_asset, cfg.max_deviation_per_asset)
    w = base + dev

    if cfg.long_only:
        w = np.maximum(w, 0.0)

    # Total deviation budget, applied before the caps so the caps survive it.
    excess = np.abs(w - base).sum()
    if excess > cfg.max_total_deviation and excess > 0:
        w = base + (w - base) * (cfg.max_total_deviation / excess)
        if cfg.long_only:
            w = np.maximum(w, 0.0)

    # Never lever. Done before the caps for the same reason.
    total = w.sum()
    if total > 1.0:
        w = w / total
    elif not cfg.allow_cash and total > 0:
        w = w / total

    # --- hard constraints, last so nothing can undo them ---------------------
    for asset, cap in cfg.structural_caps.items():
        if asset in assets:
            i = assets.index(asset)
            w[i] = min(w[i], cap)

    for group, cap in cfg.grouped_caps:
        idx = [assets.index(a) for a in group if a in assets]
        total = w[idx].sum()
        if total > cap and total > 0:
            w[idx] *= cap / total

    return w


def tilt_weight_fn(base_fn, forecast_panel: pd.DataFrame, assets: list[str],
                   cfg: TiltConfig):
    """
    Wrap a base weight function so the backtest engine can run the tilted version.

    The returned callable has the engine's signature. It looks up the forecast
    for the period being decided, which the caller must have built with the same
    no-look-ahead discipline as the engine itself - the wrapper cannot verify
    that, so `forecast_panel` must already be indexed by the period it predicts.
    """
    def fn(train: pd.DataFrame, rf: pd.Series | None = None) -> np.ndarray:
        base = np.asarray(base_fn(train, rf), dtype=float)

        # The decision at t is made from the window ending at t-1, so the
        # forecast used is the one indexed at the first period after the window.
        after = forecast_panel.index[forecast_panel.index > train.index[-1]]
        if len(after) == 0:
            return base
        row = forecast_panel.loc[after[0], assets]
        if row.isna().all():
            return base
        return apply_tilt(base, row.to_numpy(dtype=float), assets, cfg)

    return fn


def realized_deviation(held: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    """Total absolute deviation from base, period by period - the tilt's activity."""
    common = held.index.intersection(base.index)
    return (held.loc[common] - base.loc[common]).abs().sum(axis=1)
