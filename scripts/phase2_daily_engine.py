"""Daily walk-forward engine, with rebalancing frequency as a free choice.

The point of moving to daily data is that estimation frequency and trading
frequency stop being the same decision. The covariance matrix is estimated on
every day available; the portfolio trades on whatever schedule costs justify.
This script runs the risk-based allocators across four rebalancing frequencies
so the tradeoff is measured rather than assumed.

Three things the monthly build could not do:

    covariance      an 11x11 matrix has 66 free parameters. Five years of
                    monthly data is 60 observations, fewer than the parameters
                    being estimated. Five years of daily data is about 1,260.

    stale pricing   three of the seven funds autocorrelate above 0.23 at daily
                    frequency. Left uncorrected, a daily covariance understates
                    their variance and risk parity overweights them. The
                    Newey-West adjustment from the build script is applied to
                    the covariance here, so the correction flows into weights.

    turnover        with monthly data, monthly rebalancing was the only option.
                    Here it is one of four, and the cost of each is charged.

Writes fi_daily_strategies, fi_daily_summary, fi_daily_weights.
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
BUILD = import_module("phase1_build_daily")

P = ROOT / "data/processed"
PPY = 252
BURN_IN = 5 * PPY                      # five years, matching the monthly build
DEV_END = "2015-12-31"
OOS_START = "2016-01-01"
BENCH = "Agg index"
LOOKBACK = 5 * PPY                     # rolling covariance window

# Round-trip cost in basis points, charged on realised turnover.
COST_BP = {"ust2y": 2, "ust5y": 3, "ust10y": 3, "ust30y": 4, "ig_short": 15,
           "ig": 20, "ig_long": 25, "hy": 40, "mbs": 12, "muni": 25,
           "muni_hy": 45}

FREQ = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}


def rebalance_flags(index, every: int):
    f = np.zeros(len(index), dtype=bool)
    f[::every] = True
    return f


def walk(r, rf, weight_fn, rates, every: int, nw: pd.Series | None = None):
    """Estimate on all history, trade on a schedule, drift in between."""
    assets = list(r.columns)
    n = len(assets)
    flags = rebalance_flags(r.index, every)
    R = r.to_numpy()
    nets, held, rows, dates = [], None, [], []
    adj = None if nw is None else np.sqrt(nw.reindex(assets).to_numpy())

    for i in range(BURN_IN, len(r)):
        if flags[i] or held is None:
            lo = max(0, i - LOOKBACK)
            ex = r.iloc[lo:i].sub(rf.iloc[lo:i], axis=0).dropna()
            S = cov.ledoit_wolf(ex, periods_per_year=PPY)
            if adj is not None:
                # Mark up variance for autocorrelation, preserving correlations.
                S = S * np.outer(adj, adj)
            target = np.nan_to_num(weight_fn(S, assets), nan=0.0)
            if target.sum() <= 0:
                target = np.ones(n) / n
            target = target / target.sum()
            pre = np.zeros(n) if held is None else held
            cost = float(np.abs(target - pre) @ rates)
            held = target
        else:
            cost = 0.0
        nets.append(float(R[i] @ held) - cost)
        rows.append(held.copy())
        dates.append(r.index[i])
        grown = held * (1 + R[i])
        held = grown / grown.sum() if grown.sum() > 0 else held
    return (pd.Series(nets, index=dates),
            pd.DataFrame(rows, index=dates, columns=assets))


def turnover_of(W, every):
    """Annualised one-way turnover from a weight path."""
    tr = W.iloc[::every].diff().abs().sum(axis=1).dropna()
    return float(tr.mean() * (PPY / every) / 2.0)


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    stats = pd.read_parquet(P / "fi_daily_stats.parquet")

    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([COST_BP.get(a, 20) / 1e4 for a in assets])
    nw = stats["nw_factor"]

    print(f"Daily universe: {len(r):,} days, {len(assets)} assets, "
          f"{r.index.min():%Y-%m-%d} to {r.index.max():%Y-%m-%d}")
    print(f"Burn-in {BURN_IN:,} trading days, so trading starts "
          f"{r.index[BURN_IN]:%Y-%m-%d}\n")

    # Benchmarks, on the same daily grid.
    px = get_prices(["VBMFX"])
    agg = px["VBMFX"].pct_change().reindex(r.index)
    eq = pd.Series(1.0 / len(assets), index=r.index)

    def erc(S, cols):
        return RP.erc_weights(S)

    def hrp(S, cols):
        return RP.hierarchical_rp(S, list(cols))

    strat, weights, turns = {}, {}, {}
    for label, fn in [("HRP", hrp), ("ERC", erc)]:
        for fname, every in FREQ.items():
            for tag, adj in [("", nw), (", no stale adj", None)]:
                if tag and fname != "monthly":
                    continue            # the correction is shown at one freq
                name = f"{label}, {fname}{tag}"
                s, W = walk(r, rf, fn, rates, every, adj)
                strat[name] = s
                turns[name] = turnover_of(W, every)
                if label == "HRP" and fname == "monthly" and not tag:
                    weights["HRP"] = W
                print(f"  {name:<30} turnover {turns[name]:5.1%}")

    # Equal weight, rebalanced monthly, as the naive reference.
    ew, EW = walk(r, rf, lambda S, c: np.ones(len(c)) / len(c), rates, 21, None)
    strat["Equal weight"] = ew
    turns["Equal weight"] = turnover_of(EW, 21)
    strat[BENCH] = agg

    S = pd.DataFrame(strat).dropna(how="any")
    S.to_parquet(P / "fi_daily_strategies.parquet")
    weights["HRP"].to_parquet(P / "fi_daily_weights.parquet")

    dev = S.index <= pd.Timestamp(DEV_END)
    oos = S.index >= pd.Timestamp(OOS_START)
    rf2 = rf.reindex(S.index)
    bd = metrics.performance(S[BENCH][dev], rf2[dev], periods_per_year=PPY)["sharpe"]
    bo = metrics.performance(S[BENCH][oos], rf2[oos], periods_per_year=PPY)["sharpe"]

    rows = []
    for c in S.columns:
        d = metrics.performance(S[c][dev], rf2[dev], periods_per_year=PPY)
        o = metrics.performance(S[c][oos], rf2[oos], periods_per_year=PPY)
        row = {"strategy": c, "dev_return": d["ann_return"], "dev_vol": d["vol"],
               "dev_sharpe": d["sharpe"], "dev_vs_agg": d["sharpe"] - bd,
               "oos_sharpe": o["sharpe"], "oos_vs_agg": o["sharpe"] - bo,
               "turnover": turns.get(c, np.nan)}
        if c != BENCH:
            row["dev_p"] = inference.sharpe_difference(
                S[c][dev], S[BENCH][dev], rf=rf2[dev], ppy=PPY)["p_one_sided"]
            row["oos_p"] = inference.sharpe_difference(
                S[c][oos], S[BENCH][oos], rf=rf2[oos], ppy=PPY)["p_one_sided"]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "dev_sharpe", ascending=False)
    T.to_parquet(P / "fi_daily_summary.parquet")

    print(f"\nCommon window {S.index.min():%Y-%m-%d} to {S.index.max():%Y-%m-%d}"
          f", {len(S):,} days")
    print(f"Agg: development {bd:.4f}, holdout {bo:.4f}\n")
    print(T.round(4).to_string())
    print("\nHRP average weights, monthly rebalancing, stale-adjusted:")
    print((weights["HRP"].mean() * 100).round(2).sort_values(
        ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
