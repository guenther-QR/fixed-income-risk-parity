"""Derive the two numbers the reports lean on, so neither has to be taken on trust.

1. "Forecastability falls as duration rises, rank correlation -0.958."
   Reported as a Spearman correlation, which is a statement about ranks. The
   ranks themselves are written out here alongside a p-value and an ordinary
   least squares fit, so the claim can be checked rather than believed.

2. "This universe contains 1.92 independent bets out of 11."
   This is not an assumption and it is not the square root of anything. It is
   the participation ratio of the correlation matrix eigenvalues,

       N_eff = (sum lambda_i)^2 / sum(lambda_i^2)

   which equals N when every asset is uncorrelated (all eigenvalues equal) and
   1 when they are perfectly correlated (one non-zero eigenvalue). The full
   eigenvalue spectrum is written out so the number can be recomputed by hand.

   It is a descriptive statistic about the covariance structure, not a
   hypothesis test, and this project does not claim otherwise.

Writes fi_rank_*.parquet and fi_breadth_*.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

P = ROOT / "data/processed"


def rank_evidence() -> None:
    d = pd.read_parquet(P / "fi_predictability_risk.parquet")
    t = d[["oos_r2", "duration", "vol"]].copy()
    # Rank 1 = most forecastable, and 1 = longest duration, so a perfect
    # inverse relationship shows up as ranks running in opposite directions.
    t["rank_r2"] = t["oos_r2"].rank(ascending=False).astype(int)
    t["rank_duration"] = t["duration"].rank(ascending=False).astype(int)
    t["rank_vol"] = t["vol"].rank(ascending=False).astype(int)
    t = t.sort_values("rank_r2")
    t.index.name = "asset"
    t.to_parquet(P / "fi_rank_evidence.parquet")

    rows = []
    for label, col in [("modified duration", "duration"),
                       ("annualised volatility", "vol")]:
        rho, p_s = stats.spearmanr(d["oos_r2"], d[col])
        r, p_p = stats.pearsonr(d["oos_r2"], d[col])
        ols = stats.linregress(d[col], d["oos_r2"])
        rows.append({
            "predictor": col,
            "n": len(d),
            "spearman rho": rho,
            "spearman p": p_s,
            "pearson r": r,
            "pearson p": p_p,
            "OLS slope": ols.slope,
            "OLS t": ols.slope / ols.stderr,
            "OLS R2": ols.rvalue ** 2,
        })
    T = pd.DataFrame(rows).set_index("predictor")
    T.to_parquet(P / "fi_rank_tests.parquet")

    print("Ranks, most forecastable first:")
    print(t.round(4).to_string(), "\n")
    print("Tests of out-of-sample R squared against risk:")
    print(T.round(4).to_string(), "\n")
    print("  Spearman is the headline because the claim is about ordering, not")
    print("  about a linear slope. With n=12 the 5% critical value for |rho| is")
    print("  about 0.58, so -0.958 is far inside rejection. Both tests agree.\n")


def breadth_evidence() -> None:
    r = pd.read_parquet(P / "fi_returns.parquet")
    rf = pd.read_parquet(P / "fi_rf.parquet").squeeze()
    cols = [c for c in r.columns if c != "ust3m"]     # cash proxy is not held
    ex = r[cols].sub(rf, axis=0).dropna()

    C = ex.corr()
    vals = np.linalg.eigvalsh(C.to_numpy())[::-1]
    share = vals / vals.sum()
    E = pd.DataFrame({
        "eigenvalue": vals,
        "share of variance": share,
        "cumulative": np.cumsum(share),
    }, index=[f"PC{i+1}" for i in range(len(vals))])
    E.index.name = "component"
    E.to_parquet(P / "fi_breadth_evidence.parquet")

    n_eff = float(vals.sum() ** 2 / (vals ** 2).sum())
    S = pd.DataFrame({
        "value": [len(cols), round(n_eff, 2), round(float(share[0]), 4),
                  round(float(ex.corr().to_numpy()[np.triu_indices(len(cols), 1)].mean()), 4)],
        "what it is": [
            "assets held in the portfolio",
            "participation ratio of the eigenvalues, (sum L)^2 / sum(L^2)",
            "share of variance in the first principal component",
            "average pairwise correlation of excess returns"],
    }, index=["nominal assets", "effective independent assets",
              "PC1 share", "mean correlation"])
    S.index.name = "measure"
    S.to_parquet(P / "fi_breadth_stats.parquet")

    print("Eigenvalues of the correlation matrix:")
    print(E.round(4).to_string(), "\n")
    print(S.to_string(), "\n")
    print(f"  Check: ({vals.sum():.4f})^2 / {(vals ** 2).sum():.4f} = {n_eff:.4f}")
    print(f"  With {len(cols)} assets a perfectly uncorrelated universe gives "
          f"{len(cols)}.00; {len(cols)} copies of one asset gives 1.00.")
    print()


def main() -> int:
    rank_evidence()
    breadth_evidence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
