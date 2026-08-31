"""The signal library: everything the forecasting models are allowed to see.

Roughly 120 predictors across seven families. The number matters less than two
properties they all share:

**Everything is knowable when it is used.** Statistical releases are shifted by
their publication lag before entering, and every transform draws its parameters
from a trailing window. A signal that needs the full sample to compute - a
full-sample z-score being the usual offender - is not a signal, it is a leak.

**Breadth is a liability, not an asset.** Goyal and Welch (2008) tested most of
the published equity-premium predictors and found the majority fail out of
sample; several underperform the prevailing mean. Assembling 120 candidates and
picking the best would reliably produce something that looks excellent and
forecasts nothing. The library exists to be fed to *combination* and *shrinkage*
methods, which is why Phase 6 fits univariate-then-average and penalized models
rather than a kitchen-sink regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.fred import get_many, get_series
from . import transforms as tf

# Publication lag in months for each raw FRED series.
LAGS = {
    "CPIAUCSL": 1, "CPILFESL": 1, "PCEPILFE": 1, "INDPRO": 1, "UNRATE": 1,
    "PAYEMS": 1, "UMCSENT": 1, "HOUST": 1, "PERMIT": 1, "AWHMAN": 1,
    "NEWORDER": 1, "M2SL": 1, "ISRATIO": 2, "BAA": 1, "AAA": 1,
}

MACRO_SERIES = [
    "CPIAUCSL", "CPILFESL", "PCEPILFE", "INDPRO", "UNRATE", "PAYEMS",
    "UMCSENT", "HOUST", "PERMIT", "AWHMAN", "M2SL", "ICSA", "NFCI", "ANFCI",
    "BAA", "AAA", "DCOILWTICO", "T10Y3M",
]


def _month_end(s: pd.Series) -> pd.Series:
    return s.resample("ME").last()


# ------------------------------------------------------------------- families

def curve_signals(index: pd.DatetimeIndex, processed) -> pd.DataFrame:
    """Term-structure predictors, most of them built in Phase 2."""
    out = {}

    zero = pd.read_parquet(processed / "curve_zero.parquet")
    zero.columns = zero.columns.astype(float)
    zm = zero.resample("ME").last()

    def at(m: float) -> pd.Series:
        col = min(zm.columns, key=lambda c: abs(float(c) - m))
        return zm[col]

    out["curve_level"] = zm[[c for c in zm.columns if 2 <= float(c) <= 30]].mean(axis=1)
    out["curve_slope"] = at(10) - at(0.5)
    out["curve_curvature"] = 2 * at(10) - at(2) - at(30)
    out["curve_slope_2s10s"] = at(10) - at(2)
    out["short_rate"] = at(1 / 12)

    from .bond_signals import cochrane_piazzesi_expanding, fama_bliss_spreads, forward_rates
    fwd = forward_rates(zm)
    for c in fwd.columns:
        out[f"fwd_{c}"] = fwd[c]
    for c, s in fama_bliss_spreads(zm).items():
        out[c] = s
    out["cp_factor_oos"] = cochrane_piazzesi_expanding(zm, min_obs=120)

    for m in [2.0, 10.0, 30.0]:
        f = processed / f"bond_returns_{m:g}y.parquet"
        if f.exists():
            r = pd.read_parquet(f)
            out[f"carry_roll_{m:g}y"] = (
                (r["carry"] + r["rolldown"]).resample("ME").sum() * 12)
            out[f"mod_dur_{m:g}y"] = r["mod_dur"].resample("ME").last()

    df = pd.DataFrame(out).reindex(index)
    for c in ["curve_level", "curve_slope", "curve_curvature", "short_rate"]:
        df[f"{c}_chg"] = df[c].diff()
        df[f"{c}_z"] = tf.zscore(df[c])
    return df


def credit_signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Credit spreads. Moody's series reach 1919, unlike the ICE indices."""
    raw = get_many(["BAA", "AAA", "DGS10"])
    m = raw.resample("ME").last()

    out = pd.DataFrame(index=m.index)
    out["baa_aaa"] = m["BAA"] - m["AAA"]
    out["baa_10y"] = m["BAA"] - m["DGS10"]
    out["aaa_10y"] = m["AAA"] - m["DGS10"]
    for c in list(out.columns):
        out[f"{c}_chg"] = out[c].diff()
        out[f"{c}_chg3"] = out[c].diff(3)
        out[f"{c}_z"] = tf.zscore(out[c])
        out[f"{c}_rank"] = tf.rank_pct(out[c])
    # Lag the Moody's inputs; the derived columns inherit it.
    return out.shift(LAGS["BAA"]).reindex(index)


def volatility_signals(index: pd.DatetimeIndex, returns: pd.DataFrame) -> pd.DataFrame:
    """Implied and realized volatility, and the gap between them."""
    vix = _month_end(get_series("VIXCLS"))
    out = pd.DataFrame(index=index)
    out["vix"] = vix.reindex(index)
    out["vix_chg"] = out["vix"].diff()
    out["vix_z"] = tf.zscore(out["vix"])
    out["vix_rank"] = tf.rank_pct(out["vix"])

    for a in returns.columns:
        out[f"rvol_{a}"] = tf.realized_vol(returns[a], 12).reindex(index)
        out[f"rvol_{a}_z"] = tf.zscore(out[f"rvol_{a}"])

    # Variance risk premium proxy: implied minus subsequently realized equity vol.
    # Uses *trailing* realized so it stays causal.
    if "sp500" in returns:
        out["vrp_proxy"] = out["vix"] / 100.0 - tf.realized_vol(
            returns["sp500"], 12).reindex(index)
    return out


def macro_signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Growth, inflation, labour and financial conditions."""
    raw = get_many(MACRO_SERIES)
    m = raw.resample("ME").last()
    m = tf.apply_lag(m, LAGS)

    out = pd.DataFrame(index=m.index)
    for c, label in [("CPIAUCSL", "cpi"), ("CPILFESL", "core_cpi"),
                     ("PCEPILFE", "core_pce"), ("INDPRO", "indpro"),
                     ("PAYEMS", "payems"), ("M2SL", "m2"),
                     ("HOUST", "houst"), ("PERMIT", "permit")]:
        if c in m:
            out[f"{label}_yoy"] = tf.yoy(m[c])
            out[f"{label}_yoy_dev"] = tf.trend_deviation(out[f"{label}_yoy"])

    for c, label in [("UNRATE", "unrate"), ("UMCSENT", "sentiment"),
                     ("AWHMAN", "hours"), ("NFCI", "nfci"), ("ANFCI", "anfci"),
                     ("ICSA", "claims"), ("T10Y3M", "term_spread_3m")]:
        if c in m:
            out[label] = m[c]
            out[f"{label}_chg"] = m[c].diff()
            out[f"{label}_z"] = tf.zscore(m[c])

    if "UNRATE" in m:
        # Sahm rule: 3-month average unemployment minus its trailing 12-month low.
        u3 = m["UNRATE"].rolling(3).mean()
        out["sahm"] = u3 - u3.rolling(12).min()

    if "DCOILWTICO" in m:
        out["oil_yoy"] = tf.yoy(m["DCOILWTICO"])
        out["oil_z"] = tf.zscore(m["DCOILWTICO"])

    return out.reindex(index)


def momentum_signals(returns: pd.DataFrame) -> pd.DataFrame:
    """Trailing performance, per asset and cross-sectionally."""
    out = pd.DataFrame(index=returns.index)
    for a in returns.columns:
        for w in [1, 3, 6, 12]:
            out[f"mom{w}_{a}"] = tf.momentum(returns[a], w)
        # Twelve-month momentum skipping the most recent month, the standard
        # construction that separates trend from short-horizon reversal.
        out[f"mom12_1_{a}"] = tf.momentum(returns[a], 11).shift(1)

    if {"sp500", "ust10y"}.issubset(returns.columns):
        out["stock_bond_corr"] = tf.rolling_corr(returns["sp500"],
                                                 returns["ust10y"], 36)
        out["stock_bond_corr_chg"] = out["stock_bond_corr"].diff(3)
    if {"gold", "sp500"}.issubset(returns.columns):
        out["gold_vs_equity_12m"] = (tf.momentum(returns["gold"], 12)
                                     - tf.momentum(returns["sp500"], 12))
    return out


def regime_signals(index: pd.DatetimeIndex, processed) -> pd.DataFrame:
    """The Phase 3 regime states, as conditioning variables."""
    out = pd.DataFrame(index=index)
    f = processed / "regimes_monthly.parquet"
    if f.exists():
        reg = pd.read_parquet(f).reindex(index)
        for state in ["Goldilocks", "Reflation", "Stagflation", "Deflation"]:
            out[f"regime_{state.lower()}"] = (reg["regime"] == state).astype(float)
        out["growth_dev"] = reg["growth_dev"]
        out["inflation_dev"] = reg["inflation_dev"]

    f = processed / "markov_states.parquet"
    if f.exists():
        mk = pd.read_parquet(f).reindex(index)
        out["crisis_prob"] = mk.iloc[:, -1]
        out["crisis_prob_chg"] = out["crisis_prob"].diff(3)
    return out


def term_premium_signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Term premium from the NY Fed's published ACM estimate."""
    out = pd.DataFrame(index=index)
    try:
        tp = _month_end(get_series("THREEFYTP10")).reindex(index)
        out["term_premium"] = tp
        out["term_premium_chg"] = tp.diff()
        out["term_premium_z"] = tf.zscore(tp)
    except Exception:
        pass
    return out


# ------------------------------------------------------------------- assembly

def build(returns: pd.DataFrame, processed) -> pd.DataFrame:
    """
    The full signal panel, aligned to `returns` and lagged for publication.

    A final one-period shift is applied to everything, so a signal carrying
    timestamp t was knowable strictly before the return earned at t. That is
    belt-and-braces on top of the per-series publication lags, and it is the lag
    the backtest engine also enforces.
    """
    idx = returns.index
    blocks = [
        curve_signals(idx, processed),
        credit_signals(idx),
        volatility_signals(idx, returns),
        macro_signals(idx),
        momentum_signals(returns),
        regime_signals(idx, processed),
        term_premium_signals(idx),
    ]
    panel = pd.concat(blocks, axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated()]
    panel = panel.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are essentially constant or almost entirely missing.
    keep = [c for c in panel.columns
            if panel[c].notna().sum() > 120 and panel[c].std(skipna=True) > 1e-12]
    return panel[keep].shift(1)


def describe(panel: pd.DataFrame) -> pd.DataFrame:
    """Coverage and family membership, for the report."""
    families = {
        "curve": ("curve_", "fwd_", "fb", "cp_factor", "carry_roll", "mod_dur",
                  "short_rate"),
        "credit": ("baa", "aaa"),
        "volatility": ("vix", "rvol", "vrp"),
        "macro": ("cpi", "core_", "indpro", "payems", "m2_", "houst", "permit",
                  "unrate", "sentiment", "hours", "nfci", "anfci", "claims",
                  "term_spread", "sahm", "oil"),
        "momentum": ("mom", "stock_bond", "gold_vs"),
        "regime": ("regime_", "growth_dev", "inflation_dev", "crisis_prob"),
        "term_premium": ("term_premium",),
    }
    rows = []
    for c in panel.columns:
        fam = next((f for f, pre in families.items()
                    if any(c.startswith(p) for p in pre)), "other")
        s = panel[c].dropna()
        rows.append({"signal": c, "family": fam, "n_obs": len(s),
                     "first": s.index.min(), "last": s.index.max()})
    return pd.DataFrame(rows)
