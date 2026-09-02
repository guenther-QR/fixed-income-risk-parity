"""Take the best development performer from each model class to the holdout.

The forecast-driven work in this project is reported as a group failure, which
is fair but not very informative. A reader wants to see the specific model that
looked best in development and what happened to it, because that is the shape
the failure actually takes: not models that never worked, but models that worked
until they were asked to work on data nobody had seen.

Four classes, each represented by whichever member scored highest on the
development sample:

    return regression   forecast a mean, then optimise or tilt on it
    signal tilt         carry and momentum, which need no statistical estimate
    regime conditional  split the sample by macro state and estimate within it
    risk only           use the covariance matrix and nothing else

The regime models are built here rather than imported, because the parent
project's regime work was on a multi-asset book and the claim that it also fails
on fixed income should be demonstrated rather than asserted. Regime labels come
from Macro_26, which classifies each month by growth and inflation.

Everything is measured against the Bloomberg Aggregate, which is the benchmark
the rest of this project uses.

Writes fi_class_summary.parquet and fi_class_strategies.parquet.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from macro.backtest import metrics  # noqa: E402
from macro.portfolio import covariance as cov  # noqa: E402
from macro.stats import inference  # noqa: E402

P = ROOT / "data/processed"
MACRO = ROOT.parent / "Macro_26/data/processed"

START = "1987-11"
DEV_END = "2015-12-31"
OOS_START = "2016-01-01"
BENCH = "Agg index (VBMFX)"
MIN_TRAIN = 60
MIN_REGIME = 24          # months of a regime before its own estimate is usable

# Per-asset round-trip cost in basis points, matching the rest of the project.
COST_BP = {"ust2y": 2, "ust5y": 3, "ust10y": 3, "ust30y": 4, "ig_short": 15,
           "ig": 20, "ig_long": 25, "hy": 40, "mbs": 12, "muni": 25,
           "muni_hy": 45}


def erc_weights(S, iters=500):
    """Equal risk contribution by fixed-point iteration."""
    n = len(S)
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = S @ w
        rc = w * mrc
        if rc.sum() <= 0:
            break
        w = w * (rc.mean() / np.maximum(rc, 1e-12)) ** 0.5
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
    return w


def walk(r, rf, weight_fn, rates, labels=None):
    """Walk forward. `weight_fn(excess_frame) -> weights` sees only the past."""
    assets = list(r.columns)
    nets, held = [], None
    for i in range(len(r)):
        if i < MIN_TRAIN:
            nets.append(np.nan)
            continue
        ex = r.iloc[:i].sub(rf.iloc[:i], axis=0).dropna()
        if labels is not None:
            # Only months that shared the current regime label, and only if
            # there are enough of them to estimate anything from.
            now = labels.iloc[i]
            same = labels.iloc[:i] == now
            sub = ex[same.reindex(ex.index).fillna(False).to_numpy()]
            if len(sub) >= MIN_REGIME:
                ex = sub
        target = weight_fn(ex)
        target = np.nan_to_num(target, nan=0.0)
        if target.sum() <= 0:
            target = np.ones(len(assets)) / len(assets)
        target = target / target.sum()
        pre = np.zeros(len(assets)) if held is None else held
        cost = float(np.abs(target - pre) @ rates)
        held = target
        nets.append(float(r.iloc[i].to_numpy() @ held) - cost)
        grown = held * (1 + r.iloc[i].to_numpy())
        held = grown / grown.sum() if grown.sum() > 0 else held
    return pd.Series(nets, index=r.index).dropna()


def regime_labels(index):
    try:
        g = pd.read_parquet(MACRO / "regimes_monthly.parquet")["regime"]
    except Exception:
        return None
    g = g.reindex(index, method="ffill")
    return g if g.notna().sum() > len(index) * 0.5 else None


def main() -> int:
    r = pd.read_parquet(P / "fi_returns.parquet")
    rf = pd.read_parquet(P / "fi_rf.parquet").squeeze()
    bench = pd.read_parquet(P / "fi_bench_final.parquet")

    assets = [c for c in r.columns if c != "ust3m"]
    r = r[assets].loc[START:].dropna()
    rf = rf.reindex(r.index)
    rates = np.array([COST_BP.get(a, 20) / 1e4 for a in assets])
    labels = regime_labels(r.index)
    print(f"Universe {len(assets)} assets, {len(r)} months from {r.index.min():%Y-%m}")
    print(f"Regime labels: {'available' if labels is not None else 'MISSING'}")
    if labels is not None:
        print(" ", labels.value_counts().to_dict(), "\n")

    def max_sharpe_on(ex):
        """Long-only maximum Sharpe from the sample mean and Ledoit-Wolf cov."""
        mu = ex.mean().to_numpy() * 12
        S = cov.ledoit_wolf(ex)
        try:
            w = np.linalg.solve(S + np.eye(len(S)) * 1e-8, mu)
        except np.linalg.LinAlgError:
            return np.ones(len(mu)) / len(mu)
        w = np.clip(w, 0, None)
        return w if w.sum() > 0 else np.ones(len(mu)) / len(mu)

    built = {}
    # --- regime conditional -------------------------------------------------
    if labels is not None:
        built["Regime covariance RP"] = walk(
            r, rf, lambda ex: erc_weights(cov.ledoit_wolf(ex)), rates, labels)
        built["Regime mean max Sharpe"] = walk(
            r, rf, max_sharpe_on, rates, labels)
    # --- unconditional counterparts, so the regime split is isolated --------
    built["Risk parity (ERC)"] = walk(
        r, rf, lambda ex: erc_weights(cov.ledoit_wolf(ex)), rates)
    built["Max Sharpe on estimated means"] = walk(r, rf, max_sharpe_on, rates)

    # --- everything already computed elsewhere ------------------------------
    for fname, keep in [
        ("fi_all_strategies", ["Carry tilt", "Momentum tilt", "Forecast tilt"]),
        ("fi_designed_strategies", ["Max Sharpe (forecast)",
                                    "Max utility g=3 (forecast)",
                                    "Target vol (forecast)", "Black-Litterman"]),
        ("fi_rp_variants", ["Hierarchical RP"]),
    ]:
        try:
            d = pd.read_parquet(P / f"{fname}.parquet")
        except Exception:
            continue
        for c in keep:
            if c in d.columns:
                built[c] = d[c].dropna()

    S = pd.DataFrame(built)
    S[BENCH] = bench[BENCH]
    S = S.loc[START:].dropna(how="any")
    S.to_parquet(P / "fi_class_strategies.parquet")
    print(f"Assembled {S.shape[1] - 1} strategies on {len(S)} common months, "
          f"{S.index.min():%Y-%m} to {S.index.max():%Y-%m}\n")

    CLASS = {
        "Forecast tilt": "return regression",
        "Max Sharpe (forecast)": "return regression",
        "Max utility g=3 (forecast)": "return regression",
        "Target vol (forecast)": "return regression",
        "Black-Litterman": "return regression",
        "Max Sharpe on estimated means": "return regression",
        "Carry tilt": "signal tilt",
        "Momentum tilt": "signal tilt",
        "Regime covariance RP": "regime conditional",
        "Regime mean max Sharpe": "regime conditional",
        "Risk parity (ERC)": "risk only",
        "Hierarchical RP": "risk only",
    }

    rf2 = rf.reindex(S.index)
    dev = S.index <= pd.Timestamp(DEV_END)
    oos = S.index >= pd.Timestamp(OOS_START)
    rows = []
    for c in S.columns:
        if c == BENCH:
            continue
        d = metrics.performance(S[c][dev], rf2[dev])
        o = metrics.performance(S[c][oos], rf2[oos])
        bd = metrics.performance(S[BENCH][dev], rf2[dev])["sharpe"]
        bo = metrics.performance(S[BENCH][oos], rf2[oos])["sharpe"]
        bs = inference.sharpe_difference(S[c][oos], S[BENCH][oos], rf=rf2[oos])
        rows.append({
            "strategy": c, "class": CLASS.get(c, "other"),
            "dev_sharpe": d["sharpe"], "dev_vs_agg": d["sharpe"] - bd,
            "oos_sharpe": o["sharpe"], "oos_vs_agg": o["sharpe"] - bo,
            "oos_p": bs["p_one_sided"],
            "decay": (d["sharpe"] - bd) - (o["sharpe"] - bo),
        })
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "dev_vs_agg", ascending=False)
    T.to_parquet(P / "fi_class_summary.parquet")

    print(f"Agg benchmark: development {bd:.4f}, holdout {bo:.4f}\n")
    print("All candidates, sorted by development edge over the Agg:")
    print(T.round(4).to_string(), "\n")

    best = T.reset_index().sort_values("dev_vs_agg", ascending=False)
    best = best.groupby("class", as_index=False).first().set_index("class")
    best.to_parquet(P / "fi_class_best.parquet")
    print("Best development performer in each class, and its holdout result:")
    print(best.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
