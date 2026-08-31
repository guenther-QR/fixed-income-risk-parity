"""Recession prediction, and the two traps that make it look easier than it is.

The idea is appealing: recessions are when equities lose money and bonds and gold
make it, so a classifier that fires before one would be worth more than any of
the return forecasts in this project. Classification is also a far easier
statistical problem than return prediction - the target is persistent, the base
rate is stable, and the yield curve has a genuine and long-documented
relationship to it.

Two things make published recession classifiers far less useful than they look.

**The label is not known when it happens.** The NBER dates recessions in
retrospect, announcing a peak roughly seven to twelve months after the fact and a
trough later still. A model trained on `USREC` as it exists today is trained on
labels that nobody had at the time. `RECOGNITION_LAG` enforces the delay: at date
t the model may only learn from labels through t minus the lag. Without it, a
2008 model is trained on the knowledge that 2008 was a recession.

**A good classifier is not a good trading signal.** Recessions are rare - about
14% of months since 1980 - so a model that never fires is right 86% of the time,
and accuracy is meaningless. Worse, equity markets lead the economy: the S&P
typically bottoms *during* a recession, so a signal that correctly fires at the
NBER peak is often selling near the low. Area under the ROC curve measures
whether the model ranks recessionary months above expansionary ones; it says
nothing about whether acting on it makes money. Both are reported, and they
disagree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.fred import get_series

# Months before an NBER classification is public. The average announcement lag
# for a peak is about seven months and for a trough about fifteen; twelve is a
# defensible middle and is deliberately conservative.
RECOGNITION_LAG = 12

# GDP is published about one quarter after the reference period, plus revisions.
GDP_LAG = 4


def nber_recession(index: pd.DatetimeIndex) -> pd.Series:
    """NBER recession indicator, monthly, 1 during a contraction."""
    return get_series("USREC").resample("ME").last().reindex(index).ffill()


def gdp_contraction(index: pd.DatetimeIndex) -> pd.Series:
    """
    Quarters of negative real GDP growth, forward-filled to months.

    A looser target than the NBER definition and a more mechanical one, which
    makes it a useful robustness check: if a model only works on one of the two
    labels, it is probably fitting the idiosyncrasies of that label.
    """
    g = get_series("GDPC1").resample("QE").last()
    neg = (g.pct_change() < 0).astype(float)
    return neg.resample("ME").ffill().reindex(index).ffill()


def forward_target(label: pd.Series, horizon: int) -> pd.Series:
    """
    1 if a recession occurs at any point in the next `horizon` months.

    Framed as "will there be a recession soon" rather than "is there one now",
    which is the question an allocator actually faces.
    """
    fwd = label.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    return fwd.reindex(label.index)


def visible_target(label: pd.Series, horizon: int,
                   lag: int = RECOGNITION_LAG) -> pd.Series:
    """
    The forward target as it would have been *knowable* for training.

    Shifting the whole target by the recognition lag means a model fitted at date
    t sees outcomes only through t minus lag. This is the single most important
    line in the module: without it, every out-of-sample number below is fiction.
    """
    return forward_target(label, horizon).shift(lag)


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


@dataclass
class WalkForwardClassifier:
    """
    Expanding- or rolling-window classification, refit each month.

    `window` of None is an expanding window; an integer keeps only that many
    trailing months, which matters if the relationship between the curve and the
    business cycle has changed - and the 2020 recession, which no curve-based
    model saw coming, is a strong hint that it has.
    """
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
    """
    Area under the ROC curve, computed from ranks.

    Equivalent to the Mann-Whitney statistic: the probability that a randomly
    chosen recessionary month is scored above a randomly chosen expansionary one.
    0.5 is a coin flip.
    """
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
