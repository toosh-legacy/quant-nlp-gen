"""The model zoo: an escalating ladder of forecasters, weakest first.

The ladder exists so no result can be reported without context. A number is only
interesting relative to what the previous rung already achieved:

1. **persistence** — next log-vol equals current log-vol. Free, and genuinely hard to beat
   because volatility is extremely persistent. Any model that cannot beat this is noise.
2. **HAR** — Ridge on the four multi-scale realized-vol components (Corsi 2009). The
   standard academic benchmark; frequently beats far fancier models.
3. **ridge** — Ridge on the full feature set. Isolates what the extra factor families add
   while holding model capacity fixed and linear.
4. **gbm** — gradient-boosted trees on the full feature set. Adds nonlinearity and
   interactions; also the rung most able to overfit, which the walk-forward will expose.

Every estimator implements the same two-method interface so the backtest code never
branches on model type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from volradar import config

HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q"]

# Fixed GBM settings (not tuned): shared across every fit so the tuning grid only varies
# the three parameters that actually matter for capacity.
GBM_FIXED = dict(subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                 random_state=config.RANDOM_SEED, n_jobs=-1, tree_method="hist")
GBM_DEFAULT = dict(n_estimators=400, max_depth=3, learning_rate=0.05)

# Deliberately conservative grid. The market panel has only ~4k daily rows, so deep trees
# memorize; capping depth at 4 is a modelling decision made on the validation slice, not a
# convenience.
GBM_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in (300, 600) for d in (2, 3, 4) for lr in (0.03, 0.05)
]


@dataclass
class Fitted:
    """A trained model plus everything needed to reproduce and serve it."""
    name: str
    columns: list[str]
    estimator: object = None
    scaler: object = None
    params: dict = field(default_factory=dict)

    def predict(self, df) -> np.ndarray:
        if self.name == "persistence":
            return df[self.columns[0]].to_numpy(float)
        X = df[self.columns].to_numpy(float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(self.estimator.predict(X), dtype=float)


def fit_persistence(_train, _features, h: int) -> Fitted:
    """No fitting required — the prediction IS the current realized vol."""
    return Fitted(name="persistence", columns=[config.persist_col(h)])


def _fit_ridge(train, columns, h, name, alpha=1.0) -> Fitted:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X = train[columns].to_numpy(float)
    y = train[config.fwd_col(h)].to_numpy(float)
    sc = StandardScaler().fit(X)
    est = Ridge(alpha=alpha).fit(sc.transform(X), y)
    return Fitted(name=name, columns=list(columns), estimator=est, scaler=sc,
                  params={"alpha": alpha})


def fit_har(train, _features, h: int) -> Fitted:
    return _fit_ridge(train, HAR_COLS, h, "HAR")


def fit_ridge(train, features, h: int) -> Fitted:
    return _fit_ridge(train, features, h, "ridge")


def fit_gbm(train, features, h: int, params: dict | None = None) -> Fitted:
    from xgboost import XGBRegressor

    p = dict(params or GBM_DEFAULT)
    est = XGBRegressor(**GBM_FIXED, **p)
    est.fit(train[features].to_numpy(float), train[config.fwd_col(h)].to_numpy(float))
    return Fitted(name="gbm", columns=list(features), estimator=est, params=p)


def tune_gbm(fit_df, val_df, features: list[str], h: int, quiet: bool = False) -> dict:
    """Pick GBM capacity on a held-out time slice of the training period.

    Never touches the test period. The validation slice is itself chronologically after
    the fitting slice, so the selection mimics how the model will actually be used.
    """
    from sklearn.metrics import r2_score

    y_val = val_df[config.fwd_col(h)].to_numpy(float)
    best, best_score = GBM_DEFAULT, -np.inf
    for p in GBM_GRID:
        pred = fit_gbm(fit_df, features, h, p).predict(val_df)
        s = r2_score(y_val, pred)
        if s > best_score:
            best_score, best = s, p
    if not quiet:
        print(f"    h={h:>2} tuned {best}  (val R2={best_score:.3f})")
    return best


# The ladder, in reporting order.
LADDER = [
    ("persistence", fit_persistence),
    ("HAR", fit_har),
    ("ridge", fit_ridge),
    ("gbm", fit_gbm),
]
