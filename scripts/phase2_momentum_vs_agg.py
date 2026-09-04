"""Every momentum design, scored against the Aggregate.

The calibration and rolling studies each measured their tilts against risk
parity, which answers whether the tilt improves on the project's best
portfolio. It does not answer whether the tilt is worth running at all. That
question is asked against the benchmark, so every design is rebuilt here and
scored against the Aggregate on both samples.

The rolling design is also built on equal weight as well as on risk parity, so
momentum can be seen standing on its own rather than only as an overlay.
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
MC = import_module("phase2_momentum_calibration")
RM = import_module("phase2_rolling_momentum")

P = ROOT / "data/processed"
PPY = 252
MONTH = 21
BURN_IN = 5 * PPY
SEL_WINDOW = 5 * PPY
DEV_END = "2015-12-31"
WINDOWS = MC.WINDOWS
TILT = MC.TILT
CAP = MC.CAP


def rolling_signal(r, cand, Zc, assets, idx):
    """Rolling 60-month argmax selection, re-chosen monthly."""
    keys = list(cand)
    fwd_np = {a: r[a].shift(-1).to_numpy() for a in assets}
    sig_np = {k: {a: cand[k][a].to_numpy() for a in assets} for k in keys}
    rows, dates = [], []
    for i in range(len(idx)):
        if i % MONTH:
            continue
        g = BURN_IN + i
        lo = max(0, g - SEL_WINDOW)
        row = {}
        for a in assets:
            yv = fwd_np[a][lo:g - 1]
            best, best_ic = keys[0], -np.inf
            for k in keys:
                xv = sig_np[k][a][lo:g - 1]
                m = np.isfinite(xv) & np.isfinite(yv)
                if m.sum() < MC.MIN_IC_OBS:
                    continue
                ic = RM.spearman(xv[m], yv[m])
                if np.isfinite(ic) and ic > best_ic:
                    best, best_ic = k, ic
            row[a] = best
        rows.append(row)
        dates.append(idx[i])
    CH = pd.DataFrame(rows, index=dates).reindex(idx).ffill()
    return pd.DataFrame({a: [Zc[CH.loc[t, a]].at[t, a] for t in idx]
                         for a in assets}, index=idx)


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])
    fwd = r.shift(-1)

    cand = MC.momentum_candidates(r)
    Zc = {k: MC.xs_z(v) for k, v in cand.items()}
    idx = r.index[BURN_IN:]
    burn_end = r.index[BURN_IN - 1]
    n = len(assets)
    print(f"{len(cand)} candidates, {n} assets, {idx[0]:%Y-%m} to {idx[-1]:%Y-%m}")

    out, turns = {}, {}
    W_by = {}
    for lbl, every in [("monthly", MONTH), ("annual", PPY)]:
        s, W, _, tn = RB.walk(
            r, rf, rates, lambda S_, c: RB.RP.hierarchical_rp(S_, list(c)),
            lookback=None, every=every)
        out[f"HRP, {lbl}"], turns[f"HRP, {lbl}"] = s, tn
        W_by[lbl] = W
    W_eq = pd.DataFrame(1.0 / n, index=r.index, columns=assets)
    out["Equal weight, annual"], turns["Equal weight, annual"] = RM.run(
        W_eq.reindex(idx), r, rates, PPY)
    print("  built references")

    def add(name, Z, base, every):
        W = MC.tilt_to_weights(Z, base, idx)
        out[name], turns[name] = RM.run(W, r, rates, every)
        print(f"  built {name}")

    # ---- A: one definition everywhere -----------------------------------
    momA = r.rolling(252).mean().shift(MC.SKIP_MONTH) * 252
    add("A. Uniform 12-1", MC.xs_z(momA), W_by["annual"], PPY)

    # ---- B: chosen once on the burn-in ----------------------------------
    IC0 = MC.ic_frame(cand, fwd, assets, burn_end)
    frozen = {a: (IC0.loc[a].dropna().idxmax()
                  if IC0.loc[a].notna().any() else f"momsharpe{WINDOWS[-1]}")
              for a in assets}
    Sf = pd.DataFrame({a: cand[frozen[a]][a] for a in assets}).reindex(idx)
    add("B. Frozen on burn-in", MC.xs_z(Sf), W_by["annual"], PPY)

    # ---- C and D: combine rather than choose ----------------------------
    for name, use_ic in [("C. IC-weighted combination", True),
                         ("D. Equal-weight combination", False)]:
        num = pd.DataFrame(0.0, index=idx, columns=assets)
        den = pd.DataFrame(0.0, index=idx, columns=assets)
        w = pd.DataFrame(1.0, index=assets, columns=list(cand))
        for i, t in enumerate(idx):
            if use_ic and i % PPY == 0:
                w = MC.ic_frame(cand, fwd, assets, t).clip(lower=0.0).fillna(0.0)
                if float(w.to_numpy().sum()) == 0.0:
                    w = pd.DataFrame(1.0, index=assets, columns=list(cand))
            for k in cand:
                wk = w[k].reindex(assets).fillna(0.0)
                num.loc[t] += Zc[k].loc[t].reindex(assets).fillna(0.0) * wk
                den.loc[t] += wk
        add(name, (num / den.replace(0.0, np.nan)).fillna(0.0),
            W_by["annual"], PPY)

    # ---- E: rolling 60-month selection ----------------------------------
    Zr = rolling_signal(r, cand, Zc, assets, idx)
    add("E. Rolling 60m, annual trading", Zr, W_by["annual"], PPY)
    add("F. Rolling 60m, monthly trading", Zr, W_by["monthly"], MONTH)
    add("G. Rolling 60m on equal weight, annual", Zr, W_eq, PPY)

    # ---- score against the Aggregate ------------------------------------
    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)
    D = pd.DataFrame(out).dropna(how="any")
    D["Agg index"] = agg.reindex(D.index)
    D = D.dropna()
    D.to_parquet(P / "fi_momentum_paths.parquet")
    dev = D.index <= pd.Timestamp(DEV_END)
    rf2 = rf.reindex(D.index)
    b = D["Agg index"]

    rows = []
    for c in D.columns:
        if c == "Agg index":
            continue
        row = {"strategy": c, "turnover": turns.get(c, np.nan)}
        for tag, msk in [("dev", dev), ("oos", ~dev)]:
            m = metrics.performance(D[c][msk], rf2[msk], periods_per_year=PPY)
            bm = metrics.performance(b[msk], rf2[msk], periods_per_year=PPY)
            d = inf.sharpe_difference(D[c][msk], b[msk], rf=rf2[msk], ppy=PPY)
            row[f"{tag}_sharpe"] = m["sharpe"]
            row[f"{tag}_vol"] = m["vol"]
            row[f"{tag}_vs_agg"] = m["sharpe"] - bm["sharpe"]
            row[f"{tag}_p"] = d["p_one_sided"]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "dev_sharpe", ascending=False)
    T.to_parquet(P / "fi_momentum_vs_agg.parquet")

    for tag, lbl in [("dev", "DEVELOPMENT"), ("oos", "HOLDOUT")]:
        bm = metrics.performance(b[dev if tag == "dev" else ~dev],
                                 rf2[dev if tag == "dev" else ~dev],
                                 periods_per_year=PPY)["sharpe"]
        print(f"\n{lbl}  (Agg Sharpe {bm:.4f})")
        print(T[[f"{tag}_sharpe", f"{tag}_vol", f"{tag}_vs_agg", f"{tag}_p",
                 "turnover"]].sort_values(f"{tag}_sharpe", ascending=False)
              .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
