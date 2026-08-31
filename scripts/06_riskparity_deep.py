"""Why does fixed income risk parity work, and is the reason one worth having?

It is the only result in either project to clear all three tests: it beats 1/N by
+0.086 Sharpe with a bootstrap interval excluding zero, Hansen SPA p = 0.026, it
survives the leverage charge, and it holds on the sealed holdout. That makes it
worth understanding rather than just reporting.

The obvious suspicion is that it is a duration bet wearing a diversification
costume. Risk parity underweights volatile assets, volatility in fixed income is
almost entirely duration, so the strategy systematically holds less duration than
equal weight. If that is all it does, an investor who wants less duration can
simply hold less duration - no covariance matrix required - and the result is a
presentation artefact rather than a finding.

Section 3 is the test that decides it. Every strategy is compared against a
**duration-matched** equal-weight benchmark: 1/N levered or de-levered to the
same portfolio duration as the strategy, funded at the risk-free rate. If risk
parity still wins against that, it is doing something beyond duration selection.
If it does not, the honest description changes.

The other sections establish robustness, because a result that survives one
covariance estimator and one lookback is not established:

    1  where the return comes from - attribution by asset and by risk group
    2  robustness across estimator, lookback, rebalancing and cost
    3  the duration-matched test
    4  variants - inverse vol, equal risk contribution, hierarchical risk
       parity, group risk budgeting
    5  daily frequency, since the fixed income universe is fund-based and its
       stale marks matter most for exactly this strategy
"""
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from importlib import import_module  # noqa: E402

from macro.backtest import leverage, metrics, speclog  # noqa: E402
from macro.portfolio import covariance as cov  # noqa: E402
from macro.stats import inference as inf  # noqa: E402

P2 = import_module("02_allocation_engine")
P3 = import_module("03_forecast_allocation")
PROCESSED = ROOT / "data/processed"
DEV_END = "2015-12"
OOS_START = "2016-01"
MIN_TRAIN = 60
DURATION = P3.DURATION


def load(drop_bill=True):
    r = pd.read_parquet(PROCESSED / "fi_returns.parquet")
    if drop_bill and "ust3m" in r.columns:
        r = r.drop(columns=["ust3m"])
    rf = pd.read_parquet(PROCESSED / "fi_rf.parquet")["rf"].reindex(r.index)
    return r, rf


def erc_weights(S: np.ndarray, iters: int = 500) -> np.ndarray:
    n = S.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        rc = w * (S @ w)
        w = np.maximum(w * ((rc.sum() / n) / np.where(rc > 1e-16, rc, 1e-16)) ** 0.5,
                       1e-9)
        w = w / w.sum()
    return w


def hierarchical_rp(S: np.ndarray, cols: list) -> np.ndarray:
    """
    Lopez de Prado's hierarchical risk parity, simplified.

    Cluster by correlation distance, then allocate down the tree by inverse
    variance. Included because HRP is the standard answer to the criticism that
    risk parity ignores the correlation structure it is handed - it uses the
    hierarchy rather than inverting a matrix that may be near-singular.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    C = np.clip(C, -1, 1)
    D = np.sqrt(np.maximum(0.5 * (1 - C), 0))
    np.fill_diagonal(D, 0.0)
    try:
        link = linkage(squareform(D, checks=False), "single")
        order = list(leaves_list(link))
    except Exception:
        order = list(range(len(cols)))

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            c0, c1 = clusters[i], clusters[i + 1]
            v0 = _cluster_var(S, c0)
            v1 = _cluster_var(S, c1)
            alpha = 1 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    out = np.zeros(len(cols))
    for idx, val in w.items():
        out[idx] = val
    return out / out.sum() if out.sum() > 0 else np.ones(len(cols)) / len(cols)


def _cluster_var(S, idx):
    sub = S[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(sub)
    iv = iv / iv.sum()
    return float(iv @ sub @ iv)


def group_risk_budget(S, cols, groups, budget=None) -> np.ndarray:
    """Equal risk across *groups* (curve / credit / securitized), then within."""
    gm = pd.Series(groups).reindex(cols)
    uniq = [g for g in gm.dropna().unique()]
    budget = budget or {g: 1.0 / len(uniq) for g in uniq}
    w = np.zeros(len(cols))
    for g in uniq:
        idx = [i for i, c in enumerate(cols) if gm.get(c) == g]
        if not idx:
            continue
        sub = S[np.ix_(idx, idx)]
        wi = erc_weights(sub)
        gvol = np.sqrt(wi @ sub @ wi)
        w[idx] = wi * (budget[g] / max(gvol, 1e-9))
    return w / w.sum() if w.sum() > 0 else np.ones(len(cols)) / len(cols)


def backtest(r, rf, weight_fn, rebalance=1, cost_mult=1.0, lookback=None,
             cov_kind="ledoit_wolf"):
    assets = list(r.columns)
    rates = np.array([P2.COST_BP.get(a, 20) / 1e4 for a in assets]) * cost_mult
    nets, held, W = [], None, []
    for i in range(len(r)):
        if i < MIN_TRAIN:
            nets.append(np.nan)
            W.append([np.nan] * len(assets))
            continue
        if held is None or i % rebalance == 0:
            lo = 0 if lookback is None else max(0, i - lookback)
            ex = r.iloc[lo:i].sub(rf.iloc[lo:i], axis=0).dropna()
            if cov_kind == "sample":
                S = cov.sample(ex)
            elif cov_kind == "ewma":
                lam = 0.97
                w_ = lam ** np.arange(len(ex))[::-1]
                w_ = w_ / w_.sum()
                Xc = ex.to_numpy() - (ex.to_numpy() * w_[:, None]).sum(0)
                S = (Xc * w_[:, None]).T @ Xc * 12
            else:
                S = cov.ledoit_wolf(ex)
            target = weight_fn(S, assets)
            target = np.nan_to_num(target, nan=0.0)
            if target.sum() <= 0:
                target = np.ones(len(assets)) / len(assets)
            target = target / target.sum()
            pre = np.zeros(len(assets)) if held is None else held
            cost = float(np.abs(target - pre) @ rates)
            held = target
        else:
            cost = 0.0
        nets.append(float(r.iloc[i].to_numpy() @ held) - cost)
        W.append(held.copy())
        grown = held * (1 + r.iloc[i].to_numpy())
        held = grown / grown.sum() if grown.sum() > 0 else held
    return (pd.Series(nets, index=r.index).dropna(),
            pd.DataFrame(W, index=r.index, columns=assets).dropna())


def duration_matched(bench_ret, bench_W, target_W, rf, dur):
    """
    Equal weight scaled to the strategy's duration, funded at the risk-free rate.

    This is the benchmark that makes the comparison about allocation rather than
    about how much interest-rate risk was taken. Scaling below 1 means holding
    cash alongside; above 1 means borrowing, which is charged at the risk-free
    rate here because the financing spread is handled separately.
    """
    d_b = (bench_W * dur.reindex(bench_W.columns)).sum(axis=1)
    d_s = (target_W * dur.reindex(target_W.columns)).sum(axis=1)
    k = (d_s / d_b.replace(0, np.nan)).reindex(bench_ret.index).ffill().clip(0, 5)
    return k * bench_ret + (1 - k) * rf.reindex(bench_ret.index)


def main() -> int:
    r, rf = load()
    groups = pd.read_parquet(PROCESSED / "fi_groups.parquet")["group"].to_dict()
    dur = pd.Series(DURATION)
    dev, oos = slice(None, DEV_END), slice(OOS_START, None)
    print(f"Universe {r.shape[1]} assets (3M bill removed), {len(r)} months\n")

    eq = lambda S, c: np.ones(len(c)) / len(c)
    rp = lambda S, c: erc_weights(S)
    iv = lambda S, c: (1 / np.sqrt(np.diag(S))) / (1 / np.sqrt(np.diag(S))).sum()
    hrp = lambda S, c: hierarchical_rp(S, c)
    grb = lambda S, c: group_risk_budget(S, c, groups)

    core = {}
    W = {}
    for name, fn in [("1/N", eq), ("Risk parity (ERC)", rp),
                     ("Inverse volatility", iv),
                     ("Hierarchical RP", hrp), ("Group risk budget", grb)]:
        core[name], W[name] = backtest(r, rf, fn)
        print(f"  ran {name}")

    t = metrics.comparison_table({k: v.loc[dev] for k, v in core.items()},
                                 rf.loc[dev])
    t["vs_1N"] = t["sharpe"] - t.loc["1/N", "sharpe"]
    t["duration"] = pd.Series({k: float((W[k].loc[dev].mean() *
                                         dur.reindex(W[k].columns)).sum())
                               for k in W})
    print("\n1. VARIANTS (development, net of costs)")
    print(t[["cagr", "vol", "sharpe", "vs_1N", "max_drawdown", "duration"]]
          .to_string(float_format=lambda x: f"{x:9.4f}"))

    # ---- 2. is it just duration? -----------------------------------------
    print("\n2. THE DURATION-MATCHED TEST")
    print("   1/N rescaled to each strategy's own duration, funded at the")
    print("   risk-free rate. If the edge is duration selection, it vanishes.\n")
    rows = {}
    for name in core:
        if name == "1/N":
            continue
        dm = duration_matched(core["1/N"], W["1/N"], W[name], rf, dur)
        common = core[name].index.intersection(dm.dropna().index)
        s_dev = core[name].loc[common].loc[dev]
        b_dev = dm.loc[common].loc[dev]
        d = inf.sharpe_difference(s_dev, b_dev, rf.loc[dev],
                                  n_boot=3000, mean_block=12.0)
        d_raw = inf.sharpe_difference(s_dev, core["1/N"].loc[common].loc[dev],
                                      rf.loc[dev], n_boot=3000, mean_block=12.0)
        rows[name] = {
            "vs plain 1/N": d_raw["difference"],
            "vs duration-matched 1/N": d["difference"],
            "dm CI low": d["ci_lo"], "dm CI high": d["ci_hi"],
            "dm p": d["p_one_sided"],
            "strategy duration": float((W[name].loc[dev].mean() *
                                        dur.reindex(W[name].columns)).sum()),
        }
    D = pd.DataFrame(rows).T
    print(D.to_string(float_format=lambda x: f"{x:9.4f}"))

    # ---- 3. robustness ---------------------------------------------------
    print("\n3. ROBUSTNESS OF RISK PARITY")
    rob = []
    for ck, lb, rb, cm in product(["ledoit_wolf", "sample", "ewma"],
                                  [None, 60, 120], [1, 3, 12], [1.0]):
        s, w = backtest(r, rf, rp, rebalance=rb, cost_mult=cm,
                        lookback=lb, cov_kind=ck)
        b, _ = backtest(r, rf, eq, rebalance=rb, cost_mult=cm)
        common = s.index.intersection(b.index)
        e = (metrics.performance(s.loc[common].loc[dev], rf.loc[dev])["sharpe"]
             - metrics.performance(b.loc[common].loc[dev], rf.loc[dev])["sharpe"])
        eo = (metrics.performance(s.loc[common].loc[oos], rf.loc[oos])["sharpe"]
              - metrics.performance(b.loc[common].loc[oos], rf.loc[oos])["sharpe"])
        rob.append({"cov": ck, "lookback": lb or "expanding", "rebal_months": rb,
                    "dev_edge": e, "oos_edge": eo})
    R = pd.DataFrame(rob)
    print(R.pivot_table(index=["cov", "lookback"], columns="rebal_months",
                        values="dev_edge")
          .to_string(float_format=lambda x: f"{x:9.4f}"))
    print(f"\n   positive on development: {(R.dev_edge > 0).sum()} of {len(R)}")
    print(f"   positive on holdout:     {(R.oos_edge > 0).sum()} of {len(R)}")
    print(f"   mean dev edge {R.dev_edge.mean():+.4f}   "
          f"mean holdout edge {R.oos_edge.mean():+.4f}")

    # ---- 4. attribution ---------------------------------------------------
    print("\n4. WHERE THE WEIGHTS AND THE RETURN COME FROM")
    wm = pd.DataFrame({k: v.loc[dev].mean() for k, v in W.items()})
    wm["group"] = pd.Series(groups).reindex(wm.index)
    print(wm.to_string(float_format=lambda x: f"{x:7.3f}"))
    rpw = W["Risk parity (ERC)"].loc[dev]
    contrib = (rpw * r.loc[dev]).sum() / (rpw * r.loc[dev]).sum().sum()
    print("\n   share of total return by asset (risk parity):")
    print(contrib.sort_values(ascending=False)
          .to_string(float_format=lambda x: f"{x:7.1%}"))

    # ---- 5. holdout -------------------------------------------------------
    print("\n5. HOLDOUT")
    t_oos = metrics.comparison_table({k: v.loc[oos] for k, v in core.items()},
                                     rf.loc[oos])
    t_oos["vs_1N"] = t_oos["sharpe"] - t_oos.loc["1/N", "sharpe"]
    print(t_oos[["cagr", "vol", "sharpe", "vs_1N", "max_drawdown"]]
          .to_string(float_format=lambda x: f"{x:9.4f}"))
    print("\n   at a common 4.5% risk target, 50bp financing (holdout)")
    print(leverage.comparison({k: v.loc[oos] for k, v in core.items()},
                              rf.loc[oos], target_vol=0.045)[
        ["sharpe_unlevered", "leverage_needed", "sharpe_levered"]]
        .to_string(float_format=lambda x: f"{x:9.4f}"))

    for name in t.index:
        speclog.record(speclog.Spec(
            phase="FI-6", family="riskparity_deep", name=name,
            config={"layer": "development", "universe": list(r.columns)},
            metrics={k: float(t.loc[name, k]) for k in
                     ["sharpe", "cagr", "vol", "max_drawdown"]},
            n_periods=int(len(core[name].loc[dev]))))

    pd.DataFrame(core).to_parquet(PROCESSED / "fi_rp_variants.parquet")
    t.to_parquet(PROCESSED / "fi_rp_dev.parquet")
    t_oos.to_parquet(PROCESSED / "fi_rp_oos.parquet")
    D.to_parquet(PROCESSED / "fi_rp_duration_test.parquet")
    R.assign(lookback=R["lookback"].astype(str)).to_parquet(
        PROCESSED / "fi_rp_robustness.parquet")
    wm.to_parquet(PROCESSED / "fi_rp_weights.parquet")
    print("\n" + speclog.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
