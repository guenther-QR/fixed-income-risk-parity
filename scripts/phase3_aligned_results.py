"""Recompute every headline comparison on a common start date, and on a
common volatility.

Two problems with the earlier results table, both of which flattered or
disadvantaged strategies for reasons that have nothing to do with the strategies.

1. Start dates. Equal weight and the 2s10s barbell need no covariance estimate,
   so they ran from 1982-11. Risk parity, hierarchical risk parity and inverse
   volatility need one, so a 60-month burn-in pushed them to 1987-11.

   The direction of that bias is not the obvious one. Those five extra years
   hold the highest absolute bond returns in the sample, 11.63% a year for
   equal weight against 6.87% after, so the instinct is that dropping them
   should hurt the benchmark. It helps it. Cash paid 7.45% over the stub, so
   11.63% is only 4.18% of excess return, earned at 6.64% volatility in the
   unstable rate environment after the Volcker disinflation: a Sharpe of 0.575
   against 0.805 for the period that follows. The extra window was dragging
   equal weight's Sharpe *down*. Aligning therefore makes the benchmark harder
   and shrinks every edge by roughly six basis points of Sharpe.

   The fix is to start every series, benchmarks included, on the first date the
   slowest strategy can trade.

2. Volatility. A growth-of-1 chart puts the lowest-volatility line at the
   bottom, which is the opposite of the Sharpe ranking whenever the low
   volatility strategy is the better one. Comparing risk parity to equal weight
   at their natural volatilities is not an apples-to-apples comparison; risk
   parity is simply a smaller position.

   The fix is to lever every series to the same volatility target, charging a
   50bp financing spread on the borrowed portion, and to chart that. A reader
   then sees the ranking they would actually experience if they sized the
   strategies to the same risk.

Writes fi_aligned_*.parquet, which the reports read.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro.backtest import leverage, metrics  # noqa: E402
from macro.stats import inference  # noqa: E402

P = ROOT / "data/processed"

DEV_END = "2015-12-31"
OOS_START = "2016-01-01"
SPREAD_BP = 50.0
BENCH = "1/N"

ORDER = ["Hierarchical RP", "Risk parity (ERC)", "Inverse volatility",
         "1/N", "Agg index (VBMFX)", "2s10s barbell 50/50"]


def windows(idx):
    return {"dev": idx <= pd.Timestamp(DEV_END),
            "oos": idx >= pd.Timestamp(OOS_START),
            "full": pd.Series(True, index=idx).to_numpy()}


def stats_block(nets, rf, mask, tag):
    rows = {}
    for c in nets.columns:
        s = nets[c][mask].dropna()
        m = metrics.performance(s, rf.reindex(s.index))
        rows[c] = {f"{tag}_cagr": m["cagr"], f"{tag}_vol": m["vol"],
                   f"{tag}_sharpe": m["sharpe"], f"{tag}_dd": m["max_drawdown"],
                   f"{tag}_n": len(s)}
    t = pd.DataFrame(rows).T
    t[f"{tag}_vs_1N"] = t[f"{tag}_sharpe"] - t.loc[BENCH, f"{tag}_sharpe"]
    return t


def main() -> int:
    nets = pd.read_parquet(P / "fi_bench_final.parquet")
    rf = pd.read_parquet(P / "fi_rf.parquet").squeeze()

    # ---------------------------------------------------------- 1. align
    starts = nets.apply(lambda s: s.first_valid_index())
    common = starts.max()
    print("Series start dates before alignment:")
    for k, v in starts.sort_values().items():
        print(f"    {k:<24} {v:%Y-%m}")
    print(f"  common start: {common:%Y-%m}")
    print(f"  dropped {(nets.index < common).sum()} months from the long series\n")

    A = nets.loc[common:].dropna(how="any")
    rf = rf.reindex(A.index)
    w = windows(A.index)

    T = pd.concat([stats_block(A, rf, w["dev"], "dev"),
                   stats_block(A, rf, w["oos"], "oos"),
                   stats_block(A, rf, w["full"], "full")], axis=1)
    T = T.reindex([c for c in ORDER if c in T.index])
    T.to_parquet(P / "fi_aligned_table.parquet")

    print("Aligned Sharpe, and the edge the misalignment was worth:")
    old = pd.read_parquet(P / "fi_paper_table.parquet")
    for c in T.index:
        if c in old.index:
            print(f"    {c:<24} dev {old.loc[c, 'dev_sharpe']:.4f} -> "
                  f"{T.loc[c, 'dev_sharpe']:.4f}   "
                  f"edge {old.loc[c, 'dev_vs_1N']:+.4f} -> "
                  f"{T.loc[c, 'dev_vs_1N']:+.4f}")
    print()

    # ------------------------------------------------ 2. common volatility
    # Target is the benchmark's own volatility over the window, so the
    # comparison is "same risk as equal weight" rather than an arbitrary number.
    lev_rows, curves = {}, {}
    for tag, mask in w.items():
        target = float(A[BENCH][mask].std() * np.sqrt(12))
        for c in A.columns:
            s = A[c][mask].dropna()
            vol = float(s.std() * np.sqrt(12))
            L = leverage.required_leverage(vol, target)
            ls = leverage.lever_series(s, rf.reindex(s.index), L,
                                       spread_bp=SPREAD_BP)
            m = metrics.performance(ls, rf.reindex(ls.index))
            lev_rows.setdefault(c, {}).update({
                f"{tag}_leverage": L, f"{tag}_lev_cagr": m["cagr"],
                f"{tag}_lev_vol": m["vol"], f"{tag}_lev_sharpe": m["sharpe"],
                f"{tag}_lev_dd": m["max_drawdown"]})
            if tag == "full":
                curves[c] = ls
    L = pd.DataFrame(lev_rows).T.reindex([c for c in ORDER if c in lev_rows])
    for tag in ["dev", "oos", "full"]:
        L[f"{tag}_lev_vs_1N"] = (L[f"{tag}_lev_sharpe"]
                                 - L.loc[BENCH, f"{tag}_lev_sharpe"])
    L.to_parquet(P / "fi_aligned_levered.parquet")
    C = pd.DataFrame(curves)[[c for c in ORDER if c in curves]]
    C.to_parquet(P / "fi_aligned_curves.parquet")

    tv = float(A[BENCH].std() * np.sqrt(12))
    print(f"Levered to {BENCH}'s own {tv:.2%} volatility, {SPREAD_BP:.0f}bp "
          f"financing, full sample:")
    for c in L.index:
        print(f"    {c:<24} x{L.loc[c, 'full_leverage']:.2f}  "
              f"CAGR {L.loc[c, 'full_lev_cagr']:.2%}  "
              f"Sharpe {L.loc[c, 'full_lev_sharpe']:.3f}")
    print()

    # ------------------------------------------------------- 3. inference
    boot = {}
    for c in A.columns:
        if c == BENCH:
            continue
        for tag, mask in w.items():
            d = inference.sharpe_difference(A[c][mask], A[BENCH][mask],
                                            rf=rf[mask])
            boot.setdefault(c, {}).update({
                f"{tag}_edge": d["difference"], f"{tag}_lo": d["ci_lo"],
                f"{tag}_hi": d["ci_hi"], f"{tag}_p": d["p_one_sided"]})
    Bt = pd.DataFrame(boot).T.reindex(
        [c for c in ORDER if c in boot])
    Bt.to_parquet(P / "fi_aligned_bootstrap.parquet")

    print("Block bootstrap against 1/N on the aligned window:")
    for c in Bt.index:
        for tag in ["dev", "oos", "full"]:
            print(f"    {c:<24} {tag:<5} {Bt.loc[c, f'{tag}_edge']:+.4f}  "
                  f"[{Bt.loc[c, f'{tag}_lo']:+.4f}, "
                  f"{Bt.loc[c, f'{tag}_hi']:+.4f}]  p={Bt.loc[c, f'{tag}_p']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
