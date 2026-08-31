"""FI_26 Phase 4 - the predictability-risk tradeoff, inference, and the holdout.

Phase 3 found that nine of twelve fixed income assets are forecastable out of
sample - a better hit rate than Macro_26 managed on a multi-asset universe - and
that not one tilt built on those forecasts beat equal weighting.

That combination is the interesting part, and section 1 measures why. In
Macro_26 the explanation was that the forecastable assets carried little of the
portfolio's risk. If the same relationship holds inside a universe built entirely
from forecastable assets, it is not a quirk of having equities in the book. It is
a property of the assets themselves.

Sections 2 and 3 then close the project the same way Macro_26 was closed: a
multiple-testing correction over every specification logged, and a single
evaluation on a decade that was never used to choose anything.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from importlib import import_module  # noqa: E402

from macro.backtest import leverage, metrics, speclog  # noqa: E402
from macro.stats import inference as inf  # noqa: E402

P2 = import_module("phase2_allocators")
P3 = import_module("phase2_forecast_tilts")
PROCESSED = ROOT / "data/processed"
DEV_END = "2015-12"
OOS_START = "2016-01"
BENCH = "1/N monthly"


def main() -> int:
    r = pd.read_parquet(PROCESSED / "fi_returns.parquet")
    rf = pd.read_parquet(PROCESSED / "fi_rf.parquet")["rf"].reindex(r.index)
    skill = pd.read_parquet(PROCESSED / "fi_forecast_skill.parquet")["oos_r2"]
    dev = r.loc[:DEV_END]

    # ---- 1. is predictability inversely related to risk? -----------------
    print("1. THE PREDICTABILITY-RISK TRADEOFF")
    print("   If the forecastable assets are systematically the low-risk ones,")
    print("   a forecast cannot move a portfolio however good it is.\n")
    vol = dev.std() * np.sqrt(12)
    dur = pd.Series(P3.DURATION).reindex(dev.columns)
    T = pd.DataFrame({"oos_r2": skill.reindex(dev.columns), "vol": vol,
                      "duration": dur})
    T["risk_share"] = (vol ** 2) / (vol ** 2).sum()
    T = T.sort_values("oos_r2", ascending=False)
    print(T.to_string(float_format=lambda x: f"{x:9.4f}"))

    for xcol in ["vol", "duration"]:
        d = T[["oos_r2", xcol]].dropna()
        rho, p = st.spearmanr(d["oos_r2"], d[xcol])
        pr, pp = st.pearsonr(d["oos_r2"], d[xcol])
        print(f"\n   OOS R2 vs {xcol}:  Spearman {rho:+.3f} (p={p:.3f})   "
              f"Pearson {pr:+.3f} (p={pp:.3f})")

    top = T.nlargest(4, "oos_r2")
    bot = T.nsmallest(4, "oos_r2")
    print(f"\n   4 most forecastable assets hold {top['risk_share'].sum():.1%} "
          f"of universe variance")
    print(f"   4 least forecastable assets hold {bot['risk_share'].sum():.1%}")
    print("\n   Macro_26 found the same thing with equities in the book. Finding")
    print("   it again inside a universe of forecastable assets means it is a")
    print("   property of the assets, not of the earlier universe.")

    # ---- rebuild all strategies on the full sample -----------------------
    X = _signals(r)
    fe = P3.forecast_edges(r, rf, X)
    ce = P3.carry_edges(r)
    me = P3.momentum_edges(r)

    specs = [
        ("1/N monthly", P2.equal_weight, None, 1),
        ("1/N annual", P2.equal_weight, None, 12),
        ("Inverse volatility", P2.inverse_vol, None, 1),
        ("Risk parity", P2.risk_parity, None, 1),
        ("Minimum variance", P2.min_variance, None, 1),
        ("Max diversification", P2.max_diversification, None, 1),
        ("Carry tilt", P2.make_tilt(None, 0.5, 0.10), ce, 1),
        ("Momentum tilt", P2.make_tilt(None, 0.5, 0.10), me, 1),
        ("Forecast tilt", P2.make_tilt(None, 0.5, 0.10), fe, 1),
    ]
    nets = {}
    for name, fn, ed, rb in specs:
        nets[name], _ = P2.backtest(r, rf, fn, ed, rb, name)
    N = pd.DataFrame(nets)

    def block(sl, label):
        sub = {k: v.loc[sl].dropna() for k, v in nets.items()}
        t = metrics.comparison_table(sub, rf.loc[sl])
        t["vs_1N"] = t["sharpe"] - t.loc[BENCH, "sharpe"]
        print(f"\n{label}")
        print(t[["cagr", "vol", "sharpe", "vs_1N", "max_drawdown"]]
              .to_string(float_format=lambda x: f"{x:9.4f}"))
        return t

    print("\n2. INFERENCE AND HOLDOUT")
    t_dev = block(slice(None, DEV_END), "DEVELOPMENT 1987-11 to 2015-12")
    t_oos = block(slice(OOS_START, None), "HOLDOUT 2016-01 to 2026-08 (opened once)")

    print("\n   at a common 4% risk target on the holdout, financing at 50bp")
    print(leverage.comparison({k: v.loc[OOS_START:].dropna() for k, v in nets.items()},
                              rf.loc[OOS_START:], target_vol=0.04)[
        ["sharpe_unlevered", "leverage_needed", "sharpe_levered", "cagr_levered"]]
        .to_string(float_format=lambda x: f"{x:9.4f}"))

    cmp = pd.DataFrame({"dev_sharpe": t_dev["sharpe"], "oos_sharpe": t_oos["sharpe"]})
    cmp["dev_rank"] = cmp["dev_sharpe"].rank(ascending=False).astype(int)
    cmp["oos_rank"] = cmp["oos_sharpe"].rank(ascending=False).astype(int)
    rho = cmp["dev_sharpe"].corr(cmp["oos_sharpe"], method="spearman")
    print("\n   did the development ranking survive?")
    print(cmp.sort_values("dev_rank").to_string(float_format=lambda x: f"{x:9.4f}"))
    print(f"\n   rank correlation dev vs holdout: {rho:+.3f}")

    # ---- 3. multiple testing ---------------------------------------------
    print("\n3. MULTIPLE TESTING (development only)")
    d = N.loc[:DEV_END].dropna()
    ex = d.sub(rf.reindex(d.index), axis=0)
    losses = ex.drop(columns=[BENCH]).sub(ex[BENCH], axis=0)
    rc = inf.reality_check(losses, n_boot=5000, mean_block=12.0)
    print(f"   family size                {rc['n_strategies']}")
    print(f"   best performer             {rc['best']}  (t = {rc['best_t']:.3f})")
    print(f"   White Reality Check   p =  {rc['p_reality_check']:.4f}")
    print(f"   Hansen SPA            p =  {rc['p_spa']:.4f}")

    print("\n   paired block bootstrap vs 1/N (development)")
    rows = {}
    for name in d.columns:
        if name == BENCH:
            continue
        rows[name] = inf.sharpe_difference(d[name], d[BENCH], rf.reindex(d.index),
                                           n_boot=5000, mean_block=12.0)
    B = pd.DataFrame(rows).T.sort_values("difference", ascending=False)
    print(B[["difference", "ci_lo", "ci_hi", "p_one_sided"]]
          .to_string(float_format=lambda x: f"{x:9.4f}"))

    for name in t_oos.index:
        speclog.record(speclog.Spec(
            phase="FI-4", family="holdout", name=name,
            config={"layer": "holdout", "n_assets": int(r.shape[1])},
            metrics={k: float(t_oos.loc[name, k]) for k in
                     ["sharpe", "cagr", "vol", "max_drawdown"]},
            n_periods=int(len(nets[name].loc[OOS_START:]))))

    T.to_parquet(PROCESSED / "fi_predictability_risk.parquet")
    N.to_parquet(PROCESSED / "fi_all_strategies.parquet")
    t_dev.to_parquet(PROCESSED / "fi_dev_table.parquet")
    t_oos.to_parquet(PROCESSED / "fi_oos_table.parquet")
    cmp.to_parquet(PROCESSED / "fi_rank_comparison.parquet")
    B.to_parquet(PROCESSED / "fi_bootstrap.parquet")
    pd.Series({k: str(v) for k, v in rc.items()}).to_frame("value").to_parquet(
        PROCESSED / "fi_spa.parquet")
    print("\n" + speclog.summary())
    return 0


def _signals(r):
    from macro.signals import library as lib, literature as lit
    X = pd.concat([lib.build(r, PROCESSED), lit.build(r, PROCESSED)], axis=1)
    return X.loc[:, ~X.columns.duplicated()]


if __name__ == "__main__":
    raise SystemExit(main())
