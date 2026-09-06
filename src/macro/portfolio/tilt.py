"""Bounded tilts: turning return forecasts into an allocation."""
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
    """Standardize forecasts across assets at a point in time."""
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

    # --- hard constraints, last so nothing can undo them
    # ---------------------
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
    Wrap a base weight function so the backtest engine can run the tilted
    version.
    """
    def fn(train: pd.DataFrame, rf: pd.Series | None = None) -> np.ndarray:
        base = np.asarray(base_fn(train, rf), dtype=float)

        # The decision at t is made from the window ending at t-1, so the
        # forecast used is the one indexed at the first period after the
        # window.
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
