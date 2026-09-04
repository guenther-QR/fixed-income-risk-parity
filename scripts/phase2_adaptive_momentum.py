"""Momentum with the lookback window chosen per asset, walk forward.

A fixed twelve-month momentum window is a guess. Asness, Moskowitz and Pedersen
document momentum across asset classes but the horizon at which it works is not
the same everywhere, and on a bond universe the assets differ enormously in
duration and liquidity. This tests whether letting each asset pick its own
window does better than imposing one.

The selection is done on trailing data only. At each rebalance date, for every
asset, every candidate window is scored on the history available up to that
date; the best-scoring window is then used to form that asset's signal for the
period ahead. No window is chosen with knowledge of the returns it is used to
predict.

Four constructions are compared:

    fixed 12-1        the conventional twelve month window skipping the most
                      recent month, applied to every asset
    adaptive raw      each asset uses whichever window had the best trailing
                      information coefficient
    adaptive Sharpe   the same, but the signal is trailing return divided by
                      trailing volatility rather than raw return
    conviction        adaptive, with assets whose chosen window scored better in
                      selection given proportionally more weight

Development sample only. Nothing here is carried into the holdout.
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
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"
REBAL = 252
SKIP = 5                     # days skipped, to avoid short-term reversal
WINDOWS = [21, 42, 63, 126, 189, 252, 378, 504]

TILT_STRENGTH = 0.5
TILT_CAP = 0.15


def signals(r, window, kind):
    """Trailing momentum over `window` days, skipping the last SKIP."""
    roll = r.rolling(window).mean().shift(SKIP)
    if kind == "sharpe":
        vol = r.rolling(window).std().shift(SKIP)
        return roll / vol.replace(0, np.nan)
    return roll * window


def trailing_ic(sig, fwd, upto):
    """Rank correlation between a signal and next-day return, to `upto`."""
    a = sig.loc[:upto].dropna()
    b = fwd.reindex(a.index).dropna()
    a = a.reindex(b.index)
    if len(a) < PPY:
        return np.nan
    return float(a.corr(b, method="spearman"))


def build_weights(r, kind, adaptive, conviction=False):
    """Weight path from momentum, with the window chosen per asset if adaptive."""
    assets = list(r.columns)
    fwd = r.shift(-1)
    sig = {w: signals(r, w, kind) for w in WINDOWS}
    idx = r.index[BURN_IN:]
    rows, chosen_log = [], []
    cur_win = {a: 252 for a in assets}
    cur_score = {a: 0.0 for a in assets}

    for i, t in enumerate(idx):
        if i % REBAL == 0:
            for a in assets:
                if adaptive:
                    best, best_ic = 252, -np.inf
                    for w in WINDOWS:
                        ic = trailing_ic(sig[w][a], fwd[a], t)
                        if np.isfinite(ic) and ic > best_ic:
                            best, best_ic = w, ic
                    cur_win[a] = best
                    cur_score[a] = max(best_ic, 0.0) if np.isfinite(best_ic) else 0.0
                else:
                    cur_win[a] = 252
                    cur_score[a] = 1.0
            chosen_log.append({"date": t,
                               **{a: cur_win[a] for a in assets}})
        row = {a: sig[cur_win[a]][a].get(t, np.nan) for a in assets}
        rows.append(row)

    S = pd.DataFrame(rows, index=idx)
    Z = S.sub(S.mean(axis=1), axis=0).div(S.std(axis=1).replace(0, np.nan), axis=0)
    Z = Z.fillna(0.0)
    if conviction:
        # Scale each asset's tilt by how well its window scored in selection, so
        # an asset with no usable window barely moves off equal weight.
        conf = pd.Series(cur_score).reindex(Z.columns).fillna(0.0)
        if conf.max() > 0:
            Z = Z.mul(conf / conf.max(), axis=1)

    n = len(assets)
    base = 1.0 / n
    W = (base + TILT_STRENGTH * Z * base).clip(
        lower=max(base - TILT_CAP, 0.0), upper=base + TILT_CAP)
    W = W.div(W.sum(axis=1), axis=0)
    return W, pd.DataFrame(chosen_log).set_index("date") if chosen_log else None


def run(W, r, rates):
    idx = W.index
    R = r.reindex(idx).to_numpy()
    Wt = W.to_numpy()
    n = W.shape[1]
    nets, held, traded = [], None, 0.0
    for i in range(len(idx)):
        if i % REBAL == 0 or held is None:
            t = Wt[i]
            t = np.ones(n) / n if not np.isfinite(t).all() or t.sum() <= 0 else t / t.sum()
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
    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)

    specs = [
        ("Fixed 12-1", "raw", False, False),
        ("Adaptive window, raw", "raw", True, False),
        ("Adaptive window, Sharpe", "sharpe", True, False),
        ("Adaptive window, conviction", "sharpe", True, True),
    ]
    out, turns, logs = {}, {}, {}
    for name, kind, adaptive, conv in specs:
        W, log = build_weights(r, kind, adaptive, conv)
        s, tn = run(W, r, rates)
        out[name] = s
        turns[name] = tn
        if log is not None:
            logs[name] = log
        print(f"  built {name}")

    # Risk parity on the same dates, for reference.
    s_hrp, _, _, tn_hrp = RB.walk(r, rf, rates,
                                  lambda S, c: RB.RP.hierarchical_rp(S, list(c)),
                                  lookback=None, every=REBAL)
    out["HRP, annual"] = s_hrp
    turns["HRP, annual"] = tn_hrp

    S = pd.DataFrame(out).dropna(how="any")
    S["Agg index"] = agg.reindex(S.index)
    S = S.dropna()
    dev = S.index <= pd.Timestamp(DEV_END)
    rf2 = rf.reindex(S.index)
    b = S["Agg index"]
    bm = metrics.performance(b[dev], rf2[dev], periods_per_year=PPY)["sharpe"]

    rows = []
    for c in S.columns:
        if c == "Agg index":
            continue
        m = metrics.performance(S[c][dev], rf2[dev], periods_per_year=PPY)
        d = inf.sharpe_difference(S[c][dev], b[dev], rf=rf2[dev], ppy=PPY)
        rows.append({"strategy": c, "dev_sharpe": m["sharpe"],
                     "dev_vol": m["vol"], "vs_agg": m["sharpe"] - bm,
                     "p": d["p_one_sided"], "turnover": turns.get(c, np.nan)})
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "dev_sharpe", ascending=False)
    T.to_parquet(P / "fi_adaptive_momentum.parquet")

    print(f"\nAgg development Sharpe: {bm:.4f}")
    print(f"Window {S.index.min():%Y-%m-%d} to {min(S.index.max(), pd.Timestamp(DEV_END)):%Y-%m-%d}\n")
    print(T.round(4).to_string())

    if "Adaptive window, Sharpe" in logs:
        L = logs["Adaptive window, Sharpe"]
        L.to_parquet(P / "fi_adaptive_windows.parquet")
        print("\nWindow chosen per asset, most common over the sample (days):")
        print(L.mode().iloc[0].to_string())
        print("\nHow often the choice changed, per asset:")
        print((L != L.shift()).sum().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
