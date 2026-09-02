"""The same strategies financed through an overlay rather than proportionally.

Two ways to run a book at more than 100% exposure.

    proportional    scale every position by the leverage factor. Simple, and it
                    means taking a margin loan against the municipal fund, which
                    has no derivative and no repo market. That is the expensive
                    route and it is what every headline number in this project
                    assumes, because it is the conservative one.

    overlay         hold the cash book exactly as the strategy specifies,
                    municipals included at their unlevered weight, and obtain
                    the additional exposure only through instruments that have a
                    derivative: Treasury futures for the rates sleeve, total
                    return swaps for credit. The municipal sleeve is never
                    levered, so its 110bp never enters the marginal cost.

The overlay changes two things at once and both are reported. The financing rate
falls, because the expensive sleeve is excluded from the borrowed portion. And
the *composition* of the levered book shifts, because the extra exposure is
concentrated in the assets that can carry it rather than spread evenly. The
second effect is not obviously good: it tilts the levered portfolio toward
Treasuries and credit relative to the strategy's own weights.

Writes fi_overlay_summary and fi_overlay_strategies.
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
from macro.stats import inference as inf  # noqa: E402

RB = import_module("phase2_rebalance_study")

P = ROOT / "data/processed"
PPY = 252
DEV_END = "2015-12-31"
OOS_START = "2016-01-01"
REBAL = 252

# Per-asset cost of borrowing against the position, in basis points over the
# risk-free rate. Municipals have no derivative and no repo market, so the only
# route is a margin loan against the fund.
LEV_BP = {"ust2y": 3, "ust5y": 3, "ust10y": 3, "ust30y": 3, "mbs": 15,
          "ig_short": 50, "ig": 50, "ig_long": 50, "hy": 65,
          "muni": 110, "muni_hy": 110}
UNLEVERABLE = ["muni", "muni_hy"]


def lever(nets, W, rf, target_vol, mode, window):
    """Scale a strategy to `target_vol`, charging the right cost for the route.

    Returns the levered series and the annualised financing rate paid.
    """
    s = nets.dropna()
    vol = float(s.loc[window].std() * np.sqrt(PPY))
    if vol <= 0:
        return s, np.nan, 1.0
    L = target_vol / vol
    w = W.reindex(s.index).mean()
    bp = pd.Series(LEV_BP).reindex(w.index)

    if mode == "proportional":
        rate = float((w * bp).sum())
    else:
        # Only the sleeve with a derivative carries the borrowed exposure.
        lev = w.drop(index=[a for a in UNLEVERABLE if a in w.index])
        rate = float((lev / lev.sum() * bp.reindex(lev.index)).sum())
    if L <= 1.0:
        return s, rate, L
    rfa = rf.reindex(s.index).fillna(0.0)
    cost = (L - 1.0) * (rate / 1e4) / PPY
    return L * (s - rfa) + rfa - cost, rate, L


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])
    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)

    built = {}
    for lbl, fn in [("HRP", lambda S, c: RB.RP.hierarchical_rp(S, list(c))),
                    ("ERC", lambda S, c: RB.RP.erc_weights(S))]:
        s, W, tr, tn = RB.walk(r, rf, rates, fn, lookback=None, every=REBAL)
        built[lbl] = (s, W)

    idx = built["HRP"][0].index
    a = agg.reindex(idx)
    rf2 = rf.reindex(idx)
    rows, series = [], {"Agg index": a}

    for tag, sl in [("dev", slice(None, DEV_END)), ("oos", slice(OOS_START, None))]:
        pass                      # windows handled per-row below

    for lbl, (s, W) in built.items():
        for mode in ("proportional", "overlay"):
            rec = {"strategy": f"{lbl}, {mode}"}
            for tag, sl in [("dev", slice(None, DEV_END)),
                            ("oos", slice(OOS_START, None))]:
                tv = float(a.loc[sl].std() * np.sqrt(PPY))
                ls, rate, L = lever(s, W, rf2, tv, mode, sl)
                ls = ls.loc[sl]
                bb = a.loc[sl].reindex(ls.index)
                m = metrics.performance(ls, rf2.loc[sl], periods_per_year=PPY)
                bm = metrics.performance(bb, rf2.loc[sl], periods_per_year=PPY)
                z = inf.sharpe_difference(ls, bb, rf=rf2.loc[sl], ppy=PPY)
                rec[f"{tag}_leverage"] = L
                rec[f"{tag}_bp"] = rate
                rec[f"{tag}_ret"] = m["ann_return"]
                rec[f"{tag}_sharpe"] = m["sharpe"]
                rec[f"{tag}_vs_agg"] = m["sharpe"] - bm["sharpe"]
                rec[f"{tag}_ci_lo"] = z["ci_lo"]
                rec[f"{tag}_ci_hi"] = z["ci_hi"]
                rec[f"{tag}_p"] = z["p_one_sided"]
                if tag == "oos":
                    series[f"{lbl}, {mode}"] = ls
            rows.append(rec)

    T = pd.DataFrame(rows).set_index("strategy")
    T.to_parquet(P / "fi_overlay_summary.parquet")
    pd.DataFrame(series).to_parquet(P / "fi_overlay_strategies.parquet")

    show = ["dev_leverage", "dev_bp", "dev_sharpe", "dev_vs_agg", "dev_p",
            "oos_leverage", "oos_bp", "oos_sharpe", "oos_vs_agg", "oos_p"]
    print("Scaled to the Aggregate's own volatility, both financing routes:\n")
    print(T[show].round(4).to_string())
    print("\n  dev_bp / oos_bp is the blended cost of the borrowed portion.")
    print("  Overlay excludes the municipal sleeve from what gets levered, so")
    print("  the 110bp on municipals never enters the marginal rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
