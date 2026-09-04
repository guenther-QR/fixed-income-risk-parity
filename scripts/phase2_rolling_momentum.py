"""Rolling-window momentum selection, re-chosen and retraded every month.

The frozen design picked each asset's window once on the burn-in and never
revisited it. This is the other way round: the selection window is always the
trailing sixty months, it steps forward one month at a time, and the winner
trades the month that follows. Old evidence leaves the window as new evidence
enters, so the choice tracks whatever has been working recently rather than
averaging over four decades.

Run continuously from the end of the burn-in through the holdout.

Because the tilt retrades monthly, the comparison is against risk parity
rebalanced monthly as well as annually. Charging a monthly tilt against an
annually rebalanced base would credit the tilt with the effect of trading more
often.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from importlib import import_module  # noqa: E402

from macro.backtest import metrics  # noqa: E402
from macro.stats import inference as inf  # noqa: E402

RB = import_module("phase2_rebalance_study")

P = ROOT / "data/processed"
PPY = 252
MONTH = 21
BURN_IN = 5 * PPY          # also the length of the rolling selection window
SEL_WINDOW = 5 * PPY       # sixty months
DEV_END = "2015-12-31"
SKIP = 5
WINDOWS = [21, 42, 63, 126, 189, 252, 378, 504]
TILT = 0.5
CAP = 0.15
MIN_IC_OBS = 252


def momentum_candidates(r):
    out = {}
    for w in WINDOWS:
        out[f"mom{w}"] = r.rolling(w).mean().shift(SKIP) * w
        out[f"momsharpe{w}"] = (r.rolling(w).mean()
                                / r.rolling(w).std().replace(0, np.nan)).shift(SKIP)
    return out


def xs_z(df):
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
    return z.fillna(0.0)


def spearman(x, y):
    """Rank correlation on two aligned arrays, NaNs already removed."""
    if len(x) < MIN_IC_OBS:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def run(W, r, rates, every):
    """Backtest a weight path, rebalancing every `every` days, net of costs."""
    idx = W.index
    R = r.reindex(idx).to_numpy()
    Wt = W.to_numpy()
    n = W.shape[1]
    nets, held, traded = [], None, 0.0
    for i in range(len(idx)):
        if i % every == 0 or held is None:
            t = Wt[i]
            t = np.ones(n) / n if not np.isfinite(t).all() or t.sum() <= 0 \
                else t / t.sum()
            pre = np.zeros(n) if held is None else held
            traded += 0.0 if pre.sum() == 0 else float(np.abs(t - pre).sum()) / 2
            cost = float(np.abs(t - pre) @ rates)
            held = t
        else:
            cost = 0.0
        nets.append(float(R[i] @ held) - cost)
        g = held * (1 + R[i])
        held = g / g.sum() if g.sum() > 0 else held
    return pd.Series(nets, index=idx), traded / (len(idx) / PPY)


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])

    cand = momentum_candidates(r)
    keys = list(cand)
    Zc = {k: xs_z(v) for k, v in cand.items()}
    idx = r.index[BURN_IN:]
    print(f"{len(keys)} candidates, rolling {SEL_WINDOW}-day selection window, "
          f"re-chosen every {MONTH} days")
    print(f"backtest {idx[0]:%Y-%m} to {idx[-1]:%Y-%m}")

    # numpy views for the selection loop
    fwd_np = {a: r[a].shift(-1).to_numpy() for a in assets}
    sig_np = {k: {a: cand[k][a].to_numpy() for a in assets} for k in keys}
    off = BURN_IN

    chosen_rows, sel_dates = [], []
    picks = {a: [] for a in assets}
    for i in range(len(idx)):
        if i % MONTH == 0:
            g = off + i
            lo = max(0, g - SEL_WINDOW)
            row = {}
            for a in assets:
                yv = fwd_np[a][lo:g - 1]
                best, best_ic = keys[0], -np.inf
                for k in keys:
                    xv = sig_np[k][a][lo:g - 1]
                    m = np.isfinite(xv) & np.isfinite(yv)
                    if m.sum() < MIN_IC_OBS:
                        continue
                    ic = spearman(xv[m], yv[m])
                    if np.isfinite(ic) and ic > best_ic:
                        best, best_ic = k, ic
                row[a] = best
            chosen_rows.append(row)
            sel_dates.append(idx[i])
            for a in assets:
                picks[a].append(row[a])
        chosen_rows and None
    CH = pd.DataFrame(chosen_rows, index=sel_dates).reindex(idx).ffill()
    print(f"  {len(sel_dates)} monthly selections made")

    S = pd.DataFrame({a: [Zc[CH.loc[t, a]].at[t, a] for t in idx]
                      for a in assets}, index=idx)

    # bases: risk parity rebalanced monthly and annually
    out, turns = {}, {}
    W_by = {}
    for lbl, every in [("monthly", MONTH), ("annual", PPY)]:
        s_hrp, W_hrp, _, tn = RB.walk(
            r, rf, rates, lambda S_, c: RB.RP.hierarchical_rp(S_, list(c)),
            lookback=None, every=every)
        out[f"HRP, {lbl}"] = s_hrp
        turns[f"HRP, {lbl}"] = tn
        W_by[lbl] = W_hrp
        print(f"  built HRP, {lbl}")

    for lbl, every in [("monthly", MONTH), ("annual", PPY)]:
        b = W_by[lbl].reindex(idx).ffill()
        W = (b + TILT * S * b).clip(lower=0.0)
        W = W.clip(upper=b + CAP)
        W = W.div(W.sum(axis=1), axis=0)
        name = f"Rolling 60m selection, {lbl} trading"
        out[name], turns[name] = run(W, r, rates, every)
        print(f"  built {name}")

    # ---- score -----------------------------------------------------------
    D = pd.DataFrame(out).dropna(how="any")
    dev = D.index <= pd.Timestamp(DEV_END)
    rf2 = rf.reindex(D.index)
    pairs = [("Rolling 60m selection, monthly trading", "HRP, monthly"),
             ("Rolling 60m selection, annual trading", "HRP, annual")]

    rows = []
    for c, bench in pairs:
        for lbl, msk in [("development", dev), ("HOLDOUT", ~dev)]:
            m = metrics.performance(D[c][msk], rf2[msk], periods_per_year=PPY)
            h = metrics.performance(D[bench][msk], rf2[msk], periods_per_year=PPY)
            d = inf.sharpe_difference(D[c][msk], D[bench][msk],
                                      rf=rf2[msk], ppy=PPY)
            rows.append({"strategy": c, "sample": lbl, "sharpe": m["sharpe"],
                         "base": h["sharpe"],
                         "vs_base": m["sharpe"] - h["sharpe"],
                         "p_magnitude": d["p_one_sided"],
                         "turnover": turns[c]})
    for bench in ["HRP, monthly", "HRP, annual"]:
        for lbl, msk in [("development", dev), ("HOLDOUT", ~dev)]:
            m = metrics.performance(D[bench][msk], rf2[msk], periods_per_year=PPY)
            rows.append({"strategy": bench, "sample": lbl,
                         "sharpe": m["sharpe"], "base": np.nan,
                         "vs_base": np.nan, "p_magnitude": np.nan,
                         "turnover": turns[bench]})
    T = pd.DataFrame(rows).set_index(["strategy", "sample"]).sort_index()
    T.to_parquet(P / "fi_rolling_momentum.parquet")
    print()
    print(T.round(4).to_string())

    CHd = CH.loc[CH.index <= pd.Timestamp(DEV_END)]
    print("\nHow often the monthly choice changed, per asset "
          f"(out of {len(sel_dates)} selections):")
    ch = pd.Series({a: int((CH[a] != CH[a].shift()).sum() - 1) for a in assets})
    print(ch.to_string())
    print("\nMost common window per asset, development:")
    print(pd.Series({a: CHd[a].mode().iloc[0] for a in assets}).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
