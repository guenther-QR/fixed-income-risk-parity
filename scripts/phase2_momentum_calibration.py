"""Calibrating a momentum tilt without fitting it to the development sample.

The per-asset window search re-selected every year on all data available to
that point. That is walk-forward, but the winner of eight noisy rank
correlations is itself a noisy quantity, and the tilt reversed sign on the
holdout. Four designs, momentum only, each tilted around the annual
hierarchical risk parity weights and net of per-asset trading costs.

A.  Uniform. Asness, Moskowitz and Pedersen impose one momentum definition on
    all eight of their markets, the twelve month return skipping the most
    recent month, and decline to look for the best measure in each market. The
    faithful version here applies that definition identically to all eleven
    holdings, with no selection at all.

B.  Frozen. Each asset's window is chosen once, on the five year burn-in that
    precedes the backtest, and never revisited. Development then becomes a
    genuine out-of-sample test of the selection rather than the sample the
    selection was drawn from.

C.  Combined. No window is chosen. Every candidate contributes in proportion
    to its trailing rank correlation, which removes the variance that taking an
    argmax introduces.

D.  Averaged. The fully shrunk case: every candidate weighted equally, so the
    tilt rests on no estimated quantity at all.

Reported against hierarchical risk parity itself, which is the portfolio a tilt
has to beat to be worth running.
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
from macro.stats import inference as inf  # noqa: E402

RB = import_module("phase2_rebalance_study")
AM = import_module("09d_adaptive_momentum")

P = ROOT / "data/processed"
PPY = 252
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"
REBAL = 252
SKIP = 5
SKIP_MONTH = 21          # the standard definition skips a month, not a week
WINDOWS = [21, 42, 63, 126, 189, 252, 378, 504]
TILT = 0.5
CAP = 0.15
MIN_IC_OBS = 252


def momentum_candidates(r):
    """Momentum only: cumulative return and its risk-adjusted form."""
    out = {}
    for w in WINDOWS:
        out[f"mom{w}"] = r.rolling(w).mean().shift(SKIP) * w
        out[f"momsharpe{w}"] = (r.rolling(w).mean()
                                / r.rolling(w).std().replace(0, np.nan)).shift(SKIP)
    return out


def xs_z(df):
    """Cross-sectional z-score, so windows of different scale are comparable."""
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
    return z.fillna(0.0)


def ic_at(sig, fwd, upto):
    """Rank correlation using only pairs whose forward return is known by upto.

    The signal is observed through upto, but the return it predicts is the next
    day's, so the final observation has to be dropped or the estimate peeks one
    day into the future.
    """
    x = sig.loc[:upto]
    if len(x) < 2:
        return np.nan
    x = x.iloc[:-1]
    d = pd.concat([x, fwd.reindex(x.index)], axis=1).dropna()
    if len(d) < MIN_IC_OBS:
        return np.nan
    return d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman")


def ic_frame(cand, fwd, assets, upto):
    return pd.DataFrame({k: {a: ic_at(cand[k][a], fwd[a], upto) for a in assets}
                         for k in cand})


def tilt_to_weights(Z, base, idx):
    b = base.reindex(idx).ffill()
    W = (b + TILT * Z.reindex(idx).fillna(0.0) * b).clip(lower=0.0)
    W = W.clip(upper=b + CAP)
    return W.div(W.sum(axis=1), axis=0)


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])
    fwd = r.shift(-1)

    cand = momentum_candidates(r)
    Zc = {k: xs_z(v) for k, v in cand.items()}
    idx = r.index[BURN_IN:]
    burn_end = r.index[BURN_IN - 1]
    print(f"{len(cand)} momentum candidates, {len(assets)} assets")
    print(f"burn-in {r.index[0]:%Y-%m} to {burn_end:%Y-%m}, "
          f"backtest {idx[0]:%Y-%m} to {idx[-1]:%Y-%m}")

    s_hrp, W_hrp, _, tn_hrp = RB.walk(
        r, rf, rates, lambda S, c: RB.RP.hierarchical_rp(S, list(c)),
        lookback=None, every=REBAL)
    out, turns = {"HRP, annual": s_hrp}, {"HRP, annual": tn_hrp}

    # ---- A. one definition everywhere, no selection ----------------------
    momA = r.rolling(252).mean().shift(SKIP_MONTH) * 252
    out["A. Uniform 12-1, all assets"], turns["A. Uniform 12-1, all assets"] = \
        AM.run(tilt_to_weights(xs_z(momA), W_hrp, idx), r, rates)
    print("  built A. Uniform 12-1, all assets")

    # ---- B. chosen once on the burn-in, then frozen -----------------------
    IC0 = ic_frame(cand, fwd, assets, burn_end)
    frozen = {}
    for a in assets:
        row = IC0.loc[a].dropna()
        frozen[a] = row.idxmax() if len(row) else f"momsharpe{WINDOWS[-1]}"
    Sf = pd.DataFrame({a: cand[frozen[a]][a] for a in assets}).reindex(idx)
    out["B. Frozen on burn-in"], turns["B. Frozen on burn-in"] = AM.run(
        tilt_to_weights(xs_z(Sf), W_hrp, idx), r, rates)
    print("  built B. Frozen on burn-in")

    # ---- C and D. combine rather than choose ------------------------------
    for label, use_ic in [("C. IC-weighted combination", True),
                          ("D. Equal-weight combination", False)]:
        num = pd.DataFrame(0.0, index=idx, columns=assets)
        den = pd.DataFrame(0.0, index=idx, columns=assets)
        w = pd.DataFrame(1.0, index=assets, columns=list(cand))
        for i, t in enumerate(idx):
            if use_ic and i % REBAL == 0:
                w = ic_frame(cand, fwd, assets, t).clip(lower=0.0).fillna(0.0)
                if float(w.to_numpy().sum()) == 0.0:
                    w = pd.DataFrame(1.0, index=assets, columns=list(cand))
            for k in cand:
                wk = w[k].reindex(assets).fillna(0.0)
                num.loc[t] += Zc[k].loc[t].reindex(assets).fillna(0.0) * wk
                den.loc[t] += wk
        Z = (num / den.replace(0.0, np.nan)).fillna(0.0)
        out[label], turns[label] = AM.run(
            tilt_to_weights(Z, W_hrp, idx), r, rates)
        print(f"  built {label}")

    # ---- score ------------------------------------------------------------
    S = pd.DataFrame(out).dropna(how="any")
    dev = S.index <= pd.Timestamp(DEV_END)
    rf2 = rf.reindex(S.index)
    base = S["HRP, annual"]

    rows = []
    for c in S.columns:
        if c == "HRP, annual":
            continue
        for lbl, msk in [("development", dev), ("HOLDOUT", ~dev)]:
            m = metrics.performance(S[c][msk], rf2[msk], periods_per_year=PPY)
            h = metrics.performance(base[msk], rf2[msk], periods_per_year=PPY)
            d = inf.sharpe_difference(S[c][msk], base[msk], rf=rf2[msk], ppy=PPY)
            rows.append({"strategy": c, "sample": lbl, "sharpe": m["sharpe"],
                         "hrp": h["sharpe"], "vs_hrp": m["sharpe"] - h["sharpe"],
                         "p_magnitude": d["p_one_sided"],
                         "turnover": turns.get(c, np.nan)})
    T = pd.DataFrame(rows).set_index(["strategy", "sample"])
    T.to_parquet(P / "fi_momentum_calibration.parquet")
    print()
    print(T.round(4).to_string())
    print("\nWindow frozen on the burn-in, per asset:")
    print(pd.Series(frozen).to_string())
    print("\nBurn-in rank correlations of the chosen window:")
    print(pd.Series({a: IC0.loc[a, frozen[a]] for a in assets}).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
