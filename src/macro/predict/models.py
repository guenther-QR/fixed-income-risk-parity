"""Machine-learning forecasters with purged walk-forward validation."""
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
    """The training window usable for a decision at `decision_pos`."""
    end = decision_pos - horizon - embargo
    if end < min_train:
        return None
    return slice(0, end)


def _select_features(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    """
    Rank signals by absolute correlation with the target, on training data
    only.
    """
    corr = X.apply(lambda c: c.corr(y))
    return corr.abs().sort_values(ascending=False).head(k).index.tolist()


def walk_forward(y: pd.Series, X: pd.DataFrame, model_factory,
                 cfg: MLConfig) -> pd.Series:
    """
    Fit `model_factory()` on a purged expanding window, refitting on a
    cadence.
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

            # The target is standardized too, and the prediction rescaled
            # back.
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
    """A single shallow CART."""
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
