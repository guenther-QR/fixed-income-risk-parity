"""Rebalancing rules, the covariance window, and whether the municipal funds belong.

Three questions, all of which the monthly build could not ask.

1.  The covariance window. The first daily engine used a rolling five years,
    which is 1,260 observations and looks generous. It is not: the earlier
    robustness work on monthly data had already flagged the sixty-month lookback
    as the one configuration that fails out of sample, and the daily version
    inherits that failure exactly. An expanding window recovers it. This is the
    single largest driver of the result and it has nothing to do with frequency.

2.  Rebalancing frequency. Daily data separates estimation from trading: the
    covariance is estimated on everything available, and the portfolio trades
    on whatever schedule costs justify. Four calendar frequencies are compared,
    with turnover measured as what was actually traded rather than as weight
    drift.

3.  The municipal funds. Both municipal holdings autocorrelate around 0.24 at
    daily frequency, which is stale pricing rather than predictability. That is
    not a bad choice of fund. Municipal bonds are thinly traded and marked by
    matrix pricing, so every municipal fund shows it, and switching tickers
    would move the number a little without changing the cause. The universe is
    kept whole and uncorrected; the composition test below records what that
    decision is worth.

Writes fi_rebal_summary, fi_rebal_strategies, fi_composition.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from importlib import import_module  # noqa: E402

from macro.backtest import metrics  # noqa: E402
from macro.data.yahoo import get_prices  # noqa: E402
from macro.portfolio import covariance as cov  # noqa: E402
from macro.stats import inference  # noqa: E402

RP = import_module("phase2_risk_parity")

P = ROOT / "data/processed"
PPY = 252
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"
OOS_START = "2016-01-01"

COST_BP = {"ust2y": 2, "ust5y": 3, "ust10y": 3, "ust30y": 4, "ig_short": 15,
           "ig": 20, "ig_long": 25, "hy": 40, "mbs": 12, "muni": 25,
           "muni_hy": 45}


def walk(r, rf, rates, weight_fn, *, lookback=None, every=21, band=None,
         nw=None):
    """Walk forward.

    every    calendar cadence in trading days. Ignored when `band` is set.
    band     no-trade threshold. Trade only when some weight has drifted more
             than this far from its target in absolute terms.
    lookback covariance window in days; None means expanding.
    nw       per-asset variance inflation for stale pricing.
    """
    assets = list(r.columns)
    n = len(assets)
    R = r.to_numpy()
    adj = None if nw is None else np.sqrt(nw.reindex(assets).to_numpy())
    nets, held, target, rows, dates, trades, traded = [], None, None, [], [], 0, 0.0

    for i in range(BURN_IN, len(r)):
        due = (held is None
               or (band is None and i % every == 0)
               or (band is not None and target is not None
                   and np.abs(held - target).max() > band))
        if due:
            lo = 0 if lookback is None else max(0, i - lookback)
            ex = r.iloc[lo:i].sub(rf.iloc[lo:i], axis=0).dropna()
            S = cov.ledoit_wolf(ex, periods_per_year=PPY)
            if adj is not None:
                S = S * np.outer(adj, adj)
            t = np.nan_to_num(weight_fn(S, assets), nan=0.0)
            t = np.ones(n) / n if t.sum() <= 0 else t / t.sum()
            pre = np.zeros(n) if held is None else held
            moved = float(np.abs(t - pre).sum())
            cost = float(np.abs(t - pre) @ rates)
            held, target = t, t
            trades += 1
            traded += 0.0 if pre.sum() == 0 else moved / 2.0   # one-way
        else:
            cost = 0.0
        nets.append(float(R[i] @ held) - cost)
        rows.append(held.copy())
        dates.append(r.index[i])
        g = held * (1 + R[i])
        held = g / g.sum() if g.sum() > 0 else held

    W = pd.DataFrame(rows, index=dates, columns=assets)
    # Turnover is what was actually traded, one way, annualised. Weight drift
    # between rebalances is not turnover: nothing was bought or sold.
    turn = float(traded / (len(W) / PPY))
    return pd.Series(nets, index=dates), W, trades, turn


def score(s, rf, bench, label, trades=np.nan, turn=np.nan):
    rf2 = rf.reindex(s.index)
    b = bench.reindex(s.index)
    out = {"strategy": label}
    for tag, sl in [("dev", slice(None, DEV_END)), ("oos", slice(OOS_START, None))]:
        m = metrics.performance(s.loc[sl], rf2.loc[sl], periods_per_year=PPY)
        bm = metrics.performance(b.loc[sl], rf2.loc[sl], periods_per_year=PPY)
        d = inference.sharpe_difference(s.loc[sl], b.loc[sl], rf=rf2.loc[sl],
                                        ppy=PPY)
        out[f"{tag}_ret"] = m["ann_return"]
        out[f"{tag}_vol"] = m["vol"]
        out[f"{tag}_sharpe"] = m["sharpe"]
        out[f"{tag}_vs_agg"] = m["sharpe"] - bm["sharpe"]
        out[f"{tag}_p"] = d["p_one_sided"]
    out["trades"] = trades
    out["turnover"] = turn
    return out


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    st = pd.read_parquet(P / "fi_daily_stats.parquet")
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([COST_BP[a] / 1e4 for a in assets])
    nw = st["nw_factor"]
    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)

    def hrp(S, cols):
        return RP.hierarchical_rp(S, list(cols))

    def erc(S, cols):
        return RP.erc_weights(S)

    rows, series = [], {}

    # ---- 1. the covariance window -----------------------------------------
    print("1. COVARIANCE WINDOW, hierarchical risk parity, monthly rebalancing")
    for lb, lbl in [(5 * PPY, "rolling 5y"), (10 * PPY, "rolling 10y"),
                    (None, "expanding")]:
        s, W, tr, tn = walk(r, rf, rates, hrp, lookback=lb, every=21)
        rows.append(score(s, rf, agg, f"HRP, {lbl}", tr, tn))
        series[f"HRP, {lbl}"] = s
        print(f"   {lbl:<14} dev {rows[-1]['dev_sharpe']:6.3f}   "
              f"oos {rows[-1]['oos_sharpe']:6.3f}   "
              f"oos vs Agg {rows[-1]['oos_vs_agg']:+.3f}  p={rows[-1]['oos_p']:.3f}")

    # ---- 2. rebalancing rules, expanding window ---------------------------
    print("\n2. REBALANCING RULES, expanding covariance window")
    # Calendar rebalancing only. Cost-optimised rules (no-trade bands, partial
    # adjustment toward the target) are a separate question and are left to a
    # later pass rather than mixed into the headline comparison.
    plans = [("daily", dict(every=1)), ("weekly", dict(every=5)),
             ("quarterly", dict(every=63)), ("annual", dict(every=252))]
    for fn, fname in [(hrp, "HRP"), (erc, "ERC")]:
        for lbl, kw in plans:
            s, W, tr, tn = walk(r, rf, rates, fn, lookback=None, **kw)
            name = f"{fname}, {lbl}"
            rows.append(score(s, rf, agg, name, tr, tn))
            series[name] = s
            print(f"   {name:<20} dev {rows[-1]['dev_sharpe']:6.3f}  "
                  f"oos {rows[-1]['oos_sharpe']:6.3f}  "
                  f"vs Agg {rows[-1]['oos_vs_agg']:+.3f}  "
                  f"trades {tr:5d}  turnover {tn:5.1%}")

    # ---- 3. does the result need the municipals? --------------------------
    print("\n3. COMPOSITION, HRP annual rebalancing, expanding window")
    comps = {
        "all 11 assets": assets,
        "drop municipals": [a for a in assets if not a.startswith("muni")],
        "drop all stale (muni + hy)": [a for a in assets
                                       if not a.startswith("muni") and a != "hy"],
        "all 11, stale-adjusted vol": assets,
    }
    comp_rows = []
    for lbl, cols in comps.items():
        use_nw = nw if lbl == "all 11, stale-adjusted vol" else None
        rr = r[cols]
        rt = np.array([COST_BP[a] / 1e4 for a in cols])
        s, W, tr, tn = walk(rr, rf, rt, hrp, lookback=None, every=252,
                            nw=use_nw)
        rec = score(s, rf, agg, lbl, tr, tn)
        rec["n_assets"] = len(cols)
        comp_rows.append(rec)
        series[f"HRP, {lbl}"] = s
        print(f"   {lbl:<28} n={len(cols):2d}  dev {rec['dev_sharpe']:6.3f}  "
              f"oos {rec['oos_sharpe']:6.3f}  vs Agg {rec['oos_vs_agg']:+.3f}  "
              f"p={rec['oos_p']:.3f}")

    T = pd.DataFrame(rows).set_index("strategy")
    T.to_parquet(P / "fi_rebal_summary.parquet")
    C = pd.DataFrame(comp_rows).set_index("strategy")
    C.to_parquet(P / "fi_composition.parquet")
    pd.DataFrame(series).to_parquet(P / "fi_rebal_strategies.parquet")

    # Score the benchmark on the same dates the strategies trade, not from the
    # first day of data: the five-year burn-in is not part of any result.
    live = agg.loc[r.index[BURN_IN]:]
    bm_d = metrics.performance(live.loc[:DEV_END], rf.loc[:DEV_END],
                               periods_per_year=PPY)["sharpe"]
    bm_o = metrics.performance(live.loc[OOS_START:], rf.loc[OOS_START:],
                               periods_per_year=PPY)["sharpe"]
    print(f"\nAgg benchmark: development {bm_d:.4f}, holdout {bm_o:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
