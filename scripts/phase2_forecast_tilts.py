"""FI_26 Phase 3 - can return forecasts beat 1/N in fixed income?"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from macro.backtest import leverage, metrics, speclog  # noqa: E402
from macro.predict import forecast as fc  # noqa: E402
from macro.signals import library as lib, literature as lit  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

P2 = import_module("phase2_allocators")
PROCESSED = ROOT / "data/processed"
MIN_TRAIN = 120
DEV_END = "2015-12"
OOS = "1998-01"

# Approximate modified duration, used to turn a curve view into a carry signal
# and to report what the book's duration actually is.
DURATION = {"ust3m": 0.25, "ust2y": 1.9, "ust5y": 4.6, "ust10y": 8.4,
            "ust30y": 18.5, "ig_short": 2.5, "ig": 4.2, "ig_long": 12.0,
            "hy": 4.0, "mbs": 4.5, "muni": 5.0, "muni_hy": 7.5}


def carry_edges(r: pd.DataFrame) -> pd.DataFrame:
    """Carry and roll-down per asset, from the bootstrapped zero curve."""
    zero = pd.read_parquet(PROCESSED / "curve_zero.parquet")
    zero.columns = [float(c) for c in zero.columns]
    z = zero.resample("ME").last()
    grid = np.array(sorted(z.columns))
    out = pd.DataFrame(index=r.index)

    for a, m in [("ust3m", 0.25), ("ust2y", 2.0), ("ust5y", 5.0),
                 ("ust10y", 10.0), ("ust30y", 30.0)]:
        if a not in r.columns:
            continue
        y_now = z.apply(lambda row: np.interp(m, grid, row.to_numpy()), axis=1)
        y_roll = z.apply(
            lambda row: np.interp(max(m - 1 / 12, grid[0]), grid, row.to_numpy()),
            axis=1)
        carry = y_now / 12.0
        roll = (y_now - y_roll) * DURATION[a]
        out[a] = (carry + roll).reindex(r.index)

    # Funds: trailing 12-month return as a crude yield proxy.
    for a in r.columns:
        if a not in out.columns:
            out[a] = r[a].rolling(12, min_periods=6).mean()

    return out[list(r.columns)].shift(1)


def momentum_edges(r: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """Trailing return, skipping the most recent month for short-term reversal."""
    return ((1 + r).rolling(lookback - skip, min_periods=lookback - skip)
            .apply(np.prod, raw=True) - 1).shift(skip + 1)


def forecast_edges(r: pd.DataFrame, rf: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward univariate combination forecast per asset."""
    out = {}
    for a in r.columns:
        y = (r[a] - rf).dropna()
        F = fc.univariate_forecasts(y, X.reindex(y.index), MIN_TRAIN, 1)
        out[a] = fc.combine(F, "mean")
    return pd.DataFrame(out).reindex(r.index)


def main() -> int:
    r_full = pd.read_parquet(PROCESSED / "fi_returns.parquet")
    rf_full = pd.read_parquet(PROCESSED / "fi_rf.parquet")["rf"].reindex(r_full.index)
    r, rf = r_full.loc[:DEV_END], rf_full.loc[:DEV_END]

    X = pd.concat([lib.build(r_full, PROCESSED), lit.build(r_full, PROCESSED)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()].loc[:DEV_END]
    print(f"Signals: {X.shape[1]}   assets: {r.shape[1]}   months: {len(r)}\n")

    # ---- forecast skill first --------------------------------------------
    print("1. FORECAST SKILL PER ASSET (OOS R2 vs prevailing mean, from 1998)")
    oos = slice(OOS, DEV_END)
    fe = forecast_edges(r, rf, X)
    skill = {}
    for a in r.columns:
        y = (r[a] - rf).dropna()
        bench = fc.prevailing_mean(y, min_obs=MIN_TRAIN)
        skill[a] = fc.oos_r2(y[oos], fe[a][oos], bench[oos])
    S = pd.Series(skill).sort_values(ascending=False)
    for a, v in S.items():
        print(f"   {a:10s} {v:+7.2%}")
    print(f"\n   {(S > 0).sum()} of {len(S)} assets forecastable out of sample")
    print("   (Macro_26 for comparison: 5 of 7, but equities at +0.05%)")

    # ---- allocation -------------------------------------------------------
    print("\n2. ALLOCATION vs 1/N")
    ce = carry_edges(r)
    me = momentum_edges(r)

    nets, weights = {}, {}
    for name, fn, ed, rb in [
        ("1/N monthly", P2.equal_weight, None, 1),
        ("Risk parity", P2.risk_parity, None, 1),
        ("Carry tilt", P2.make_tilt(None, 0.5, 0.10), ce, 1),
        ("Momentum tilt", P2.make_tilt(None, 0.5, 0.10), me, 1),
        ("Forecast tilt", P2.make_tilt(None, 0.5, 0.10), fe, 1),
        ("Forecast tilt (strong)", P2.make_tilt(None, 1.0, 0.20), fe, 1),
        ("Carry + forecast", P2.make_tilt(None, 0.5, 0.10),
         (ce.rank(axis=1) + fe.rank(axis=1)) / 2.0, 1),
    ]:
        nets[name], weights[name] = P2.backtest(r, rf, fn, ed, rb, name)
        print(f"   ran {name}")

    tab = metrics.comparison_table(nets, rf)
    tab["vs_1N"] = tab["sharpe"] - tab.loc["1/N monthly", "sharpe"]
    print("\n   development, net of costs")
    print(tab[["cagr", "vol", "sharpe", "vs_1N", "max_drawdown"]]
          .to_string(float_format=lambda x: f"{x:9.4f}"))

    print("\n   at a common 4% risk target, financing charged at 50bp")
    print(leverage.comparison(nets, rf, target_vol=0.04)[
        ["sharpe_unlevered", "leverage_needed", "sharpe_levered", "cagr_levered"]]
        .to_string(float_format=lambda x: f"{x:9.4f}"))

    print("\n   average portfolio duration")
    dur = pd.Series(DURATION)
    for name, w in weights.items():
        print(f"     {name:24s} {float((w.mean() * dur.reindex(w.columns)).sum()):5.2f} years")

    for name in tab.index:
        speclog.record(speclog.Spec(
            phase="FI-3", family="forecast_allocation", name=name,
            config={"layer": "development", "n_signals": int(X.shape[1])},
            metrics={k: float(tab.loc[name, k]) for k in
                     ["sharpe", "cagr", "vol", "max_drawdown"]},
            n_periods=int(len(nets[name]))))

    pd.DataFrame(nets).to_parquet(PROCESSED / "fi_forecast_strategies.parquet")
    tab.to_parquet(PROCESSED / "fi_forecast_summary.parquet")
    S.to_frame("oos_r2").to_parquet(PROCESSED / "fi_forecast_skill.parquet")
    fe.to_parquet(PROCESSED / "fi_forecast_edges.parquet")
    print("\n" + speclog.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
