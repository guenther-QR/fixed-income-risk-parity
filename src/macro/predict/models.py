"""Machine-learning forecasters with purged walk-forward validation.

Standard k-fold cross-validation is invalid here and would inflate every result.
Two reasons, both structural:

    Overlapping outcomes. A 12-month forward return at t shares eleven months of
    data with the one at t+1. Splitting randomly puts near-identical observations
    in both training and test folds, so the model is scored partly on data it
    trained on.

    Time ordering. A random split trains on the future to predict the past, which
    no live strategy can do.

Lopez de Prado's purging and embargoing fixes both: training observations whose
outcome window overlaps the test period are removed, and a further gap is left
between train and test so that serial correlation cannot bridge them.

Every model here refits on a cadence rather than every period. That is partly
cost - a gradient booster refit 380 times is slow - and partly signal: monthly
refitting of a model this flexible produces weights that churn without adding
information.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MLConfig:
    min_train: int = 180
    refit_every: int = 12
    embargo: int = 12
    horizon: int = 1
    max_features: int = 60          # cap on signals fed to the model


def purged_train_index(index: pd.DatetimeIndex, decision_pos: int,
                       horizon: int, embargo: int, min_train: int) -> slice | None:
    """
    The training window usable for a decision at `decision_pos`.

    Ends `horizon + embargo` periods before the decision: `horizon` because an
    outcome starting later than that has not been observed, and `embargo` to stop
    serial correlation carrying information across the boundary.
    """
    end = decision_pos - horizon - embargo
    if end < min_train:
        return None
    return slice(0, end)


def _select_features(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    """
    Rank signals by absolute correlation with the target, on training data only.

    Selection is part of the model and must therefore happen inside the training
    window. Choosing features on the full sample is a leak that survives every
    other precaution.
    """
    corr = X.apply(lambda c: c.corr(y))
    return corr.abs().sort_values(ascending=False).head(k).index.tolist()


def walk_forward(y: pd.Series, X: pd.DataFrame, model_factory,
                 cfg: MLConfig) -> pd.Series:
    """
    Fit `model_factory()` on a purged expanding window, refitting on a cadence.

    Returns a forecast series indexed by the period predicted, matching the
    convention in `forecast.py`.
    """
    from sklearn.preprocessing import StandardScaler

    idx = X.index
    out = pd.Series(index=idx, dtype=float)
    model = None
    features: list[str] = []
    scaler = None
    last_fit = -10 ** 9

    for pos in range(cfg.min_train, len(idx)):
        sl = purged_train_index(idx, pos, cfg.horizon, cfg.embargo, cfg.min_train)
        if sl is None:
            continue

        if pos - last_fit >= cfg.refit_every or model is None:
            Xtr = X.iloc[sl]
            ytr = y.iloc[sl]
            ok = Xtr.notna().all(axis=1) & ytr.notna()
            Xtr, ytr = Xtr[ok], ytr[ok]
            if len(Xtr) < cfg.min_train // 2:
                continue

            features = _select_features(Xtr, ytr, cfg.max_features)
            scaler = StandardScaler().fit(Xtr[features])

            # The target is standardized too, and the prediction rescaled back.
            # A penalty expressed on the raw return scale means something
            # different for every asset: alpha = 0.001 is negligible against
            # gold's 4.8% monthly volatility and substantial against the 2-year's
            # 0.46%. Left unscaled, the elastic net overfitted gold so badly its
            # forecast varied twice as much as the returns it was forecasting,
            # for an out-of-sample R-squared of -354%.
            y_mu, y_sd = float(ytr.mean()), float(ytr.std())
            y_sd = y_sd if y_sd > 1e-12 else 1.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model_factory()
                model.fit(scaler.transform(Xtr[features]), (ytr - y_mu) / y_sd)
            last_fit = pos

        row = X.iloc[[pos]][features]
        if row.isna().any(axis=1).iloc[0]:
            continue
        out.iloc[pos] = float(model.predict(scaler.transform(row))[0]) * y_sd + y_mu

    return out


# ------------------------------------------------------------------ factories

def elastic_net(alpha: float = 0.10, l1_ratio: float = 0.5):
    from sklearn.linear_model import ElasticNet
    return lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)


def ridge(alpha: float = 10.0):
    from sklearn.linear_model import Ridge
    return lambda: Ridge(alpha=alpha)


def random_forest(n_estimators: int = 300, max_depth: int = 4):
    from sklearn.ensemble import RandomForestRegressor
    return lambda: RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=20, random_state=0, n_jobs=-1)


def gradient_boosting(n_estimators: int = 200, max_depth: int = 2,
                      learning_rate: float = 0.02):
    from sklearn.ensemble import GradientBoostingRegressor
    return lambda: GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.7, random_state=0)


def shallow_tree(max_depth: int = 3):
    """
    A single shallow CART.

    Included for interpretability rather than accuracy: a three-level tree can be
    read as a handful of if-then rules, and a rule a portfolio manager can state
    out loud is worth more than a marginal gain in fit.
    """
    from sklearn.tree import DecisionTreeRegressor
    return lambda: DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=24,
                                         random_state=0)


MODELS = {
    "elastic_net": elastic_net,
    "ridge": ridge,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
    "shallow_tree": shallow_tree,
}
