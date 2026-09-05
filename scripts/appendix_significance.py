"""The same results under a textbook test and under the one this project uses.

Every p-value in the project comes from a stationary block bootstrap, which
makes no distributional assumption and preserves whatever serial dependence the
returns carry. The conventional alternative is the Jobson-Korkie test with
Memmel's correction, the standard parametric test for a difference between two
Sharpe ratios. It assumes returns are independent and normally distributed.

Neither is wrong. They answer the same question under different assumptions,
and showing both makes it possible to see what the stricter test costs. Bond
returns are neither independent nor normal, so where the two disagree the
bootstrap is the one to believe.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from macro.stats import inference as inf  # noqa: E402

P = ROOT / "data/processed"
PPY, BLOCK = 12, 12
DEV_END = "2015-12-31"
BENCH = "Agg index"


def jobson_korkie(a, b):
    """Memmel-corrected z for the difference between two Sharpe ratios.

    Assumes the two return series are independent draws from a bivariate
    normal, which bond returns are not. That is the point of showing it
    beside the bootstrap.
    """
    n = len(a)
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = float(np.corrcoef(a, b)[0, 1])
    theta = (1.0 / n) * (2.0 * (1.0 - rho)
                         + 0.5 * (sa ** 2 + sb ** 2 - 2.0 * sa * sb * rho ** 2))
    if theta <= 0:
        return np.nan, np.nan
    z = (sa - sb) / np.sqrt(theta)
    return float(z), float(st.norm.sf(abs(z)))       # one sided


def main() -> int:
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    rf = (1.0 + rf).resample("ME").prod() - 1.0
    D = pd.read_parquet(P / "fi_holdout_paths.parquet")
    D = (1.0 + D).resample("ME").prod() - 1.0
    rf = rf.reindex(D.index)
    dev = D.index <= pd.Timestamp(DEV_END)

    rows = []
    for c in D.columns:
        if c == BENCH:
            continue
        row = {"strategy": c}
        for tag, msk in [("dev", dev), ("oos", ~dev),
                         ("full", pd.Series(True, index=D.index))]:
            x = (D[c][msk] - rf[msk]).dropna()
            y = (D[BENCH][msk] - rf[msk]).reindex(x.index)
            sh_a = x.mean() / x.std(ddof=1) * np.sqrt(PPY)
            sh_b = y.mean() / y.std(ddof=1) * np.sqrt(PPY)
            z, p_jk = jobson_korkie(x.to_numpy(), y.to_numpy())
            p_bs = inf.sharpe_difference(D[c][msk], D[BENCH][msk], rf=rf[msk],
                                         ppy=PPY, mean_block=BLOCK)["p_one_sided"]
            row[f"{tag}_edge"] = sh_a - sh_b
            row[f"{tag}_t"] = z
            row[f"{tag}_p_t"] = p_jk
            row[f"{tag}_p_boot"] = p_bs
        rows.append(row)
    T = pd.DataFrame(rows).set_index("strategy")
    T.to_parquet(P / "fi_significance_comparison.parquet")

    print(f"{len(D)} months, {D.index[0]:%Y-%m} to {D.index[-1]:%Y-%m}\n")
    for tag, lbl in [("dev", "DEVELOPMENT"), ("oos", "HOLDOUT"),
                     ("full", "FULL SAMPLE")]:
        print(f"=== {lbl} ===")
        print(T[[f"{tag}_edge", f"{tag}_t", f"{tag}_p_t",
                 f"{tag}_p_boot"]].round(4).to_string())
        n_t = (T[f"{tag}_p_t"] < 0.05).sum()
        n_b = (T[f"{tag}_p_boot"] < 0.05).sum()
        print(f"  significant at 5%: {n_t} under the t test, "
              f"{n_b} under the bootstrap\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
