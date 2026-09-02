"""The canonical daily portfolios: HRP and ERC, annual rebalancing, expanding
covariance, all eleven assets, no volatility correction.

These are the two strategies carried into the holdout, and this script saves the
series and the weight paths the reports read.
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
from macro.data.yahoo import get_prices  # noqa: E402

RB = import_module("phase2_rebalance_study")
P = ROOT / "data/processed"

DUR = {"ust2y": 1.9, "ust5y": 4.6, "ust10y": 8.4, "ust30y": 18.5,
       "ig_short": 2.5, "ig": 4.2, "ig_long": 12.0, "hy": 4.0, "mbs": 4.5,
       "muni": 5.0, "muni_hy": 7.5}


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])

    out, weights = {}, {}
    for lbl, fn in [("Hierarchical RP", lambda S, c: RB.RP.hierarchical_rp(S, list(c))),
                    ("Risk parity (ERC)", lambda S, c: RB.RP.erc_weights(S))]:
        s, W, tr, tn = RB.walk(r, rf, rates, fn, lookback=None, every=252)
        out[lbl] = s
        weights[lbl] = W
        print(f"{lbl:<20} {tr} trades, turnover {tn:.2%}")

    eq, EW, _, etn = RB.walk(r, rf, rates,
                             lambda S, c: np.ones(len(c)) / len(c),
                             lookback=None, every=252)
    out["Equal weight"] = eq
    out["Agg index"] = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)

    S = pd.DataFrame(out).dropna(how="any")
    S.to_parquet(P / "fi_canonical_strategies.parquet")
    weights["Hierarchical RP"].to_parquet(P / "fi_canonical_hrp_weights.parquet")
    weights["Risk parity (ERC)"].to_parquet(P / "fi_canonical_erc_weights.parquet")

    W = weights["Hierarchical RP"]
    dur = (W * pd.Series(DUR).reindex(W.columns)).sum(axis=1)
    dur.to_frame("duration").to_parquet(P / "fi_canonical_hrp_duration.parquet")
    print(f"\nHRP portfolio duration: mean {dur.mean():.2f}y, "
          f"range {dur.min():.2f} to {dur.max():.2f}")
    print("\nHRP average weights (%):")
    print((W.mean() * 100).round(2).sort_values(ascending=False).to_string())
    print(f"\nsaved {len(S):,} days, {S.index.min():%Y-%m-%d} to "
          f"{S.index.max():%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
