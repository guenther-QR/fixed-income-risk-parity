"""Put every comparison on a common start date and a common volatility."""
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
# Financing, charged per asset rather than as one blended rate.
#
# Transaction costs in this project are already per asset, because a Treasury
# bill and a high yield municipal do not trade at the same spread. Financing has
# the same property and for the same reason: what it costs to lever a position
# depends on what instrument carries it. A Treasury finances through repo or a
# futures basis at a few basis points. A municipal bond fund has no derivative
# and no futures contract, so it finances through a prime broker margin loan at
# a hundred basis points or more.
#
# The assumption throughout is an institutional book, a fund or bank desk with
# access to repo, listed futures and cleared swaps, not a retail margin account.
#
# Sources for the ranges are in the reports. Mid-range values are used here.
FINANCING_BP = {
    # Treasuries: SOFR is constructed from Treasury general collateral repo,
    # so financing them is close to definitionally flat to the reference rate.
    "ust2y": 3.0, "ust5y": 3.0, "ust10y": 3.0, "ust30y": 3.0,
    # Agency mortgages fund through TBA dollar rolls and agency repo, which
    # trade a few basis points wide of Treasury GC.
    "mbs": 15.0,
    # Investment grade credit: total return swap at SOFR + 30 to 75bp, plus a
    # 10 to 25bp agent fee where one applies.
    "ig_short": 50.0, "ig": 50.0, "ig_long": 50.0,
    # High yield prices wider than investment grade on the same structure.
    "hy": 65.0,
    # Municipals are the expensive leg and the reason a blended rate misleads.
    "muni": 110.0, "muni_hy": 110.0,
}

# The two benchmarks are single instruments rather than baskets, so their
# financing is stated directly.
BENCH_FINANCING_BP = {"2s10s barbell 50/50": 3.0, "Agg index (VBMFX)": 40.0}

# Sensitivity grid: a flat spread applied to every asset, so the per-asset
# result can be located against the simpler assumption.
SPREADS = [0.0, 25.0, 50.0, 100.0, 150.0]
# The Aggregate index is the benchmark a fixed income mandate is actually
# measured against, so it is the primary comparison and the volatility target.
# Equal weight is kept as a second reference because it is the naive allocation
# any method has to beat to justify itself, but nobody is benchmarked to it.
#
# The Agg is never levered. Levering a benchmark to match a strategy inverts the
# question being asked: the mandate is fixed and the strategy has to fit it, not
# the other way round. So every strategy is scaled to the Agg's volatility and
# the Agg is held as it comes.
BENCH = "Agg index (VBMFX)"
BENCH2 = "1/N"

ORDER = ["Hierarchical RP", "Risk parity (ERC)",
         "Agg index (VBMFX)", "1/N", "2s10s barbell 50/50"]

# The benchmarks are the yardstick, not candidates.
STRATEGIES = ["Hierarchical RP", "Risk parity (ERC)"]


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
    t[f"{tag}_vs_agg"] = t[f"{tag}_sharpe"] - t.loc[BENCH, f"{tag}_sharpe"]
    t[f"{tag}_vs_1N"] = t[f"{tag}_sharpe"] - t.loc[BENCH2, f"{tag}_sharpe"]
    return t


# Sleeves that cannot be levered synthetically.
UNLEVERABLE = ["muni", "muni_hy"]


def blended_spreads():
    """
    What each strategy pays to finance leverage, under two implementations.
    """
    try:
        W = pd.read_parquet(P / "fi_rp_weights.parquet")
    except Exception:
        return None, None, None
    W = W.drop(columns=[c for c in ["group"] if c in W.columns])
    bp = pd.Series(FINANCING_BP).reindex(W.index)
    if bp.isna().any():
        print("  missing a financing rate for:", list(bp[bp.isna()].index))
        return None, None, None

    proportional = W.mul(bp, axis=0).sum()

    lev = W.drop(index=[a for a in UNLEVERABLE if a in W.index])
    lev = lev.div(lev.sum())                    # renormalise the levered sleeve
    overlay = lev.mul(bp.reindex(lev.index), axis=0).sum()

    detail = pd.DataFrame({"financing bp": bp})
    for c in W.columns:
        detail[c] = W[c]
    routes = pd.DataFrame({"proportional_bp": proportional,
                           "overlay_bp": overlay})
    return proportional, detail, routes


def main() -> int:
    nets = pd.read_parquet(P / "fi_bench_final.parquet")
    rf = pd.read_parquet(P / "fi_rf.parquet").squeeze()

    # ---------------------------------------------------------- 1.
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

    print("Aligned Sharpe against both benchmarks:")
    for c in T.index:
        if True:
            print(f"    {c:<24} dev {T.loc[c, 'dev_sharpe']:.4f}  "
                  f"vs Agg {T.loc[c, 'dev_vs_agg']:+.4f}  "
                  f"vs 1/N {T.loc[c, 'dev_vs_1N']:+.4f}")
    print()

    # ------------------------------------------------ 2.
    blended, detail, routes = blended_spreads()
    if blended is not None:
        print("Financing charged per asset, blended by each strategy's weights:")
        print(detail.round(4).to_string())
        print()
        print("Cost of the borrowed portion, by implementation:")
        print(routes.round(1).to_string())
        print("  proportional: scale every position, finance each at its own rate")
        print("  overlay:      leave", "/".join(UNLEVERABLE), "at cash weight,")
        print("                lever only what has a derivative")
        print()
        detail.to_parquet(P / "fi_financing_detail.parquet")
        routes.to_parquet(P / "fi_financing_routes.parquet")

    def spread_for(name):
        """A strategy pays what its own holdings cost to finance."""
        if name in BENCH_FINANCING_BP:
            return BENCH_FINANCING_BP[name]
        if blended is None or name not in blended.index:
            return 40.0
        return float(blended[name])

    lev_rows, curves = {}, {}
    for tag, mask in w.items():
        target = float(A[BENCH][mask].std() * np.sqrt(12))
        for c in A.columns:
            s = A[c][mask].dropna()
            vol = float(s.std() * np.sqrt(12))
            # The benchmark defines the risk budget, so it is held as it
            # comes.
            L = 1.0 if c == BENCH else leverage.required_leverage(vol, target)
            ls = leverage.lever_series(s, rf.reindex(s.index), L,
                                       spread_bp=spread_for(c))
            m = metrics.performance(ls, rf.reindex(ls.index))
            lev_rows.setdefault(c, {}).update({
                f"{tag}_spread_bp": spread_for(c),
                f"{tag}_leverage": L, f"{tag}_lev_cagr": m["cagr"],
                f"{tag}_lev_vol": m["vol"], f"{tag}_lev_sharpe": m["sharpe"],
                f"{tag}_lev_dd": m["max_drawdown"]})
            if tag == "full":
                curves[c] = ls
    L = pd.DataFrame(lev_rows).T.reindex([c for c in ORDER if c in lev_rows])
    for tag in ["dev", "oos", "full"]:
        L[f"{tag}_lev_vs_agg"] = (L[f"{tag}_lev_sharpe"]
                                  - L.loc[BENCH, f"{tag}_lev_sharpe"])
        L[f"{tag}_lev_vs_1N"] = (L[f"{tag}_lev_sharpe"]
                                 - L.loc[BENCH2, f"{tag}_lev_sharpe"])
    L.to_parquet(P / "fi_aligned_levered.parquet")
    C = pd.DataFrame(curves)[[c for c in ORDER if c in curves]]
    C.to_parquet(P / "fi_aligned_curves.parquet")

    tv = float(A[BENCH].std() * np.sqrt(12))
    print(f"Levered to {BENCH}'s own {tv:.2%} volatility, per-asset financing, "
          f"full sample:")
    for c in L.index:
        print(f"    {c:<24} x{L.loc[c, 'full_leverage']:.2f}  "
              f"@{L.loc[c, 'full_spread_bp']:5.1f}bp  "
              f"CAGR {L.loc[c, 'full_lev_cagr']:.2%}  "
              f"Sharpe {L.loc[c, 'full_lev_sharpe']:.3f}")
    print()

    # ----------------------------------------- 2b.
    base = metrics.performance(A[BENCH], rf)["sharpe"]
    rows = []
    for c in A.columns:
        if c == BENCH:
            continue
        sr = A[c]
        lv = leverage.required_leverage(float(sr.std() * np.sqrt(12)), tv)
        if lv <= 1.0:
            continue
        lo, hi = 0.0, 600.0
        for _ in range(60):                     # bisect on the spread
            mid = (lo + hi) / 2
            sh = metrics.performance(
                leverage.lever_series(sr, rf, lv, spread_bp=mid), rf)["sharpe"]
            lo, hi = (mid, hi) if sh > base else (lo, mid)
        paid = spread_for(c)
        rows.append({"strategy": c, "leverage": lv, "pays_bp": paid,
                     "breakeven_bp": lo, "headroom_bp": lo - paid})
    BE = pd.DataFrame(rows).set_index("strategy")
    BE.to_parquet(P / "fi_financing_breakeven.parquet")
    print(f"Breakeven financing spread, against {BENCH} at {base:.3f}:")
    print(BE.round(1).to_string())
    print()

    # ------------------------------------------------------- 3.
    boots = {}
    for bench, fname in [(BENCH, "fi_aligned_bootstrap"),
                         (BENCH2, "fi_aligned_bootstrap_1n")]:
        boot = {}
        for c in A.columns:
            if c == bench:
                continue
            for tag, mask in w.items():
                d = inference.sharpe_difference(A[c][mask], A[bench][mask],
                                                rf=rf[mask])
                boot.setdefault(c, {}).update({
                    f"{tag}_edge": d["difference"], f"{tag}_lo": d["ci_lo"],
                    f"{tag}_hi": d["ci_hi"], f"{tag}_p": d["p_one_sided"]})
        boots[bench] = pd.DataFrame(boot).T.reindex(
            [c for c in STRATEGIES if c in boot])
        boots[bench].to_parquet(P / f"{fname}.parquet")
    Bt = boots[BENCH]

    # ------------------------------------- 4.
    rows = {}
    full = w["full"]
    target = float(A[BENCH][full].std() * np.sqrt(12))
    for bp in SPREADS:
        for c in A.columns:
            s = A[c][full].dropna()
            lv = leverage.required_leverage(float(s.std() * np.sqrt(12)), target)
            ls = leverage.lever_series(s, rf.reindex(s.index), lv, spread_bp=bp)
            rows.setdefault(c, {})[f"{bp:.0f}bp"] = metrics.performance(
                ls, rf.reindex(ls.index))["sharpe"]
    S = pd.DataFrame(rows).T.reindex([c for c in ORDER if c in rows])
    S.to_parquet(P / "fi_aligned_spread_sensitivity.parquet")
    print("Levered Sharpe by financing spread, full sample:")
    print(S.round(3).to_string())
    print()

    for bench, frame in boots.items():
        print(f"Block bootstrap against {bench} on the aligned window:")
        for c in frame.index:
            for tag in ["dev", "oos", "full"]:
                print(f"    {c:<24} {tag:<5} {frame.loc[c, f'{tag}_edge']:+.4f}  "
                      f"[{frame.loc[c, f'{tag}_lo']:+.4f}, "
                      f"{frame.loc[c, f'{tag}_hi']:+.4f}]  "
                      f"p={frame.loc[c, f'{tag}_p']:.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
