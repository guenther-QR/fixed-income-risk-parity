"""
Recession prediction, and the two traps that make it look easier than it is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.fred import get_series

# Months before an NBER classification is public.
RECOGNITION_LAG = 12

# GDP is published about one quarter after the reference period, plus
# revisions.
GDP_LAG = 4


def nber_recession(index: pd.DatetimeIndex) -> pd.Series:
    """NBER recession indicator, monthly, 1 during a contraction."""
    return get_series("USREC").resample("ME").last().reindex(index).ffill()


def gdp_contraction(index: pd.DatetimeIndex) -> pd.Series:
    """Quarters of negative real GDP growth, forward-filled to months."""
    g = get_series("GDPC1").resample("QE").last()
    neg = (g.pct_change() < 0).astype(float)
    return neg.resample("ME").ffill().reindex(index).ffill()


def forward_target(label: pd.Series, horizon: int) -> pd.Series:
    """1 if a recession occurs at any point in the next `horizon` months."""
    fwd = label.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    return fwd.reindex(label.index)


def visible_target(label: pd.Series, horizon: int,
                   lag: int = RECOGNITION_LAG) -> pd.Series:
    """The forward target as it would have been *knowable* for training."""
    return forward_target(label, horizon).shift(lag)


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


@dataclass
class WalkForwardClassifier:
    """Expanding- or rolling-window classification, refit each month."""
    model: str = "logit"
    window: int | None = None
    min_train: int = 120
    seed: int = 20260830

    def _fit_predict(self, Xtr, ytr, Xte):
        if len(np.unique(ytr)) < 2:
            return float(ytr.mean())
        Xtr, Xte = _standardise(Xtr, Xte)

        if self.model == "logit":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=2000, C=1.0)
        elif self.model == "logit_l1":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=4000, C=0.1, penalty="l1",
                                   solver="liblinear")
        elif self.model == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=300, max_depth=4,
                                       min_samples_leaf=12, n_jobs=-1,
                                       random_state=self.seed)
        elif self.model == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                           learning_rate=0.05,
                                           random_state=self.seed)
        else:
            raise ValueError(self.model)

        m.fit(Xtr, ytr)
        return float(m.predict_proba(Xte)[0, 1])

    def run(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        d = pd.concat([X, y.rename("_y")], axis=1)
        Xc = d[X.columns]
        out = pd.Series(index=X.index, dtype=float)

        for i in range(self.min_train, len(X)):
            lo = 0 if self.window is None else max(0, i - self.window)
            tr = d.iloc[lo:i].dropna()
            if len(tr) < self.min_train // 2:
                continue
            xte = Xc.iloc[[i]]
            if xte.isna().any(axis=1).iloc[0]:
                continue
            out.iloc[i] = self._fit_predict(
                tr[X.columns].to_numpy(), tr["_y"].to_numpy(), xte.to_numpy())
        return out


def auc(y_true: pd.Series, score: pd.Series) -> float:
    """Area under the ROC curve, computed from ranks."""
    d = pd.concat([y_true.rename("y"), score.rename("s")], axis=1).dropna()
    y, s = d["y"].to_numpy(), d["s"].to_numpy()
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y_true: pd.Series, prob: pd.Series) -> float:
    d = pd.concat([y_true.rename("y"), prob.rename("p")], axis=1).dropna()
    return float(((d["p"] - d["y"]) ** 2).mean())


def classification_report(y_true: pd.Series, prob: pd.Series,
                          threshold: float = 0.5) -> dict:
    d = pd.concat([y_true.rename("y"), prob.rename("p")], axis=1).dropna()
    pred = (d["p"] >= threshold).astype(int)
    tp = int(((pred == 1) & (d["y"] == 1)).sum())
    fp = int(((pred == 1) & (d["y"] == 0)).sum())
    fn = int(((pred == 0) & (d["y"] == 1)).sum())
    tn = int(((pred == 0) & (d["y"] == 0)).sum())
    return {
        "auc": auc(d["y"], d["p"]), "brier": brier(d["y"], d["p"]),
        "base_rate": float(d["y"].mean()),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "n_signals": int(pred.sum()), "n_obs": int(len(d)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
