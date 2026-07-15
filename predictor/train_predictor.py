"""Train the movement predictors, for each prediction horizon.

For a horizon h (in trading days) we predict the direction of the return from close[t] to
close[t+h]. The three honest feature variants the project is about:

  * price-only     — technical factors, no text
  * sentiment-only — the daily sentiment factor, no price
  * combined        — both

and two model families:

  * LogisticRegression (`class_weight="balanced"`) — the simple, transparent baseline.
  * XGBoost, hyperparameter-tuned on the dev split — gradient-boosted trees, what quant
    teams reach for on tabular multi-factor data.

Two disciplines keep the numbers honest:
  * Standardization / tuning use ONLY train and dev. Test is never touched here.
  * Embargo: because a horizon-h label's outcome window overlaps its neighbors', we keep a
    row for horizon h only if its outcome day (t+h) lands in the SAME split as its feature
    day (t). That prevents a train example's outcome from overlapping a dev/test example's
    outcome, which would otherwise inflate the score.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# The three feature sets that define the experiment.
FEATURE_SETS: dict[str, list[str]] = {
    "price_only": config.PRICE_FEATURE_COLS,
    "sentiment_only": config.SENTIMENT_FEATURE_COLS,
    "combined": config.PRICE_FEATURE_COLS + config.SENTIMENT_FEATURE_COLS,
}

# Fixed XGBoost settings shared across the tuning search.
XGB_BASE = dict(
    random_state=config.RANDOM_SEED,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist",
)

# Sensible default hyperparameters, used when the dev split is too small to tune on (which
# happens for the longest horizon, whose outcome window overruns the short StockNet dev
# window and gets fully embargoed).
XGB_DEFAULT = dict(
    max_depth=3, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=1.0,
)

# Minimum dev rows required to trust a hyperparameter search.
MIN_DEV_ROWS = 100


@dataclass
class TrainedModel:
    """A fitted estimator plus the scaler + columns it expects, so evaluation can apply it
    to the test split identically."""
    name: str            # e.g. "xgboost/combined"
    family: str          # "logreg" | "xgboost"
    feature_set: str     # "price_only" | "sentiment_only" | "combined"
    horizon: int
    columns: list[str]
    scaler: object | None
    estimator: object


def load_combined() -> pd.DataFrame:
    if not config.COMBINED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{config.COMBINED_FEATURES_PATH} not found. Run features/build_features.py first."
        )
    return pd.read_parquet(config.COMBINED_FEATURES_PATH)


def embargoed_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Rows usable for horizon h, with the overlapping-window embargo (purging) applied.

    The label must be defined, and then:
      * train/dev rows are kept only if their outcome day (t+h) lands in the SAME split as
        the feature day — this purges rows whose h-day outcome window would spill forward
        into a later split and contaminate it.
      * test rows are kept as long as the label is defined. Their outcome day is allowed to
        extend past the test window (using real later prices we loaded): that is NOT
        contamination, because nothing is trained on the test split. Requiring test labels
        to also stay inside the short 3-month test window would needlessly discard most
        long-horizon test rows.
    Adds an integer working target column `y`.
    """
    label_defined = df[config.movement_col(h)].notna()
    same_split = df["split"] == df[config.label_split_col(h)]
    is_test = df["split"] == "test"
    out = df[label_defined & (same_split | is_test)].copy()
    out["y"] = out[config.movement_col(h)].astype(int)
    return out.reset_index(drop=True)


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)


def _fit_logreg(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_SEED, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf


def _fit_xgboost(X_train, y_train, params: dict):
    from xgboost import XGBClassifier

    clf = XGBClassifier(**XGB_BASE, **params)
    clf.fit(X_train, y_train)
    return clf


def tune_xgboost(train: pd.DataFrame, dev: pd.DataFrame, cols: list[str]) -> tuple[dict, float]:
    """Small randomized search over XGBoost params, selected by dev-split MCC. Returns the
    best params (including a class-imbalance-aware scale_pos_weight) and its dev MCC.

    We select on MCC, not accuracy, because on a ~50/50 target accuracy barely moves; MCC is
    what we actually care about. Tuning touches only train (fit) and dev (score) — never test.
    """
    from sklearn.metrics import matthews_corrcoef
    from sklearn.model_selection import ParameterSampler

    X_tr, y_tr = train[cols].to_numpy(dtype=float), train["y"].to_numpy()
    X_dv, y_dv = dev[cols].to_numpy(dtype=float), dev["y"].to_numpy()

    # scale_pos_weight counteracts class imbalance (ratio of negatives to positives).
    n_pos = max(1, int((y_tr == 1).sum()))
    n_neg = int((y_tr == 0).sum())
    spw = n_neg / n_pos

    grid = {
        "max_depth": [2, 3, 4],
        "n_estimators": [200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [1.0, 3.0],
        "scale_pos_weight": [1.0, spw],
    }

    best_params: dict | None = None
    best_mcc = -2.0
    for params in ParameterSampler(grid, n_iter=24, random_state=config.RANDOM_SEED):
        clf = _fit_xgboost(X_tr, y_tr, params)
        mcc = matthews_corrcoef(y_dv, clf.predict(X_dv))
        if mcc > best_mcc:
            best_mcc, best_params = mcc, params
    assert best_params is not None
    return best_params, best_mcc


def train_all(df: pd.DataFrame, h: int) -> tuple[list[TrainedModel], pd.DataFrame]:
    """Fit every (family x feature-set) combination for horizon h. Returns the trained
    models and the embargoed frame (so the caller can pull its test split)."""
    from sklearn.preprocessing import StandardScaler

    frame = embargoed_frame(df, h)
    train = split_frame(frame, "train")
    dev = split_frame(frame, "dev")
    y_train = train["y"].to_numpy()

    models: list[TrainedModel] = []
    for fs_name, cols in FEATURE_SETS.items():
        X_train_raw = train[cols].to_numpy(dtype=float)

        # Linear model: standardized inputs, scaler fit on train only.
        scaler = StandardScaler().fit(X_train_raw)
        logreg = _fit_logreg(scaler.transform(X_train_raw), y_train)
        models.append(TrainedModel(
            name=f"logreg/{fs_name}", family="logreg", feature_set=fs_name, horizon=h,
            columns=cols, scaler=scaler, estimator=logreg,
        ))

        # XGBoost: tuned on dev when dev is large enough, else sensible defaults.
        if len(dev) >= MIN_DEV_ROWS:
            best_params, dev_mcc = tune_xgboost(train, dev, cols)
            note = f"tuned, dev MCC={dev_mcc:+.4f}"
        else:
            best_params, note = XGB_DEFAULT, f"defaults (dev too small: {len(dev)} rows)"
        xgb = _fit_xgboost(X_train_raw, y_train, best_params)
        models.append(TrainedModel(
            name=f"xgboost/{fs_name}", family="xgboost", feature_set=fs_name, horizon=h,
            columns=cols, scaler=None, estimator=xgb,
        ))
        print(f"  h={h:>2} {fs_name:14s}: logreg + xgboost ({note})")

    return models, frame


def predict(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    """Apply a trained model to a frame, using the exact columns/scaler it was fit with."""
    X = df[model.columns].to_numpy(dtype=float)
    if model.scaler is not None:
        X = model.scaler.transform(X)
    return model.estimator.predict(X)


if __name__ == "__main__":
    # Standalone sanity run for the default horizon: train + report train-split accuracy.
    from sklearn.metrics import accuracy_score

    frame_df = load_combined()
    trained, frame = train_all(frame_df, config.DEFAULT_HORIZON)
    tr = split_frame(frame, "train")
    for m in trained:
        acc = accuracy_score(tr["y"], predict(m, tr))
        print(f"  {m.name:24s} train-acc={acc:.3f}")
