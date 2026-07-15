"""Step 6 (training half) — Train the movement predictors.

Trains the three honest variants the whole project is about, for each of two model
families:

  * price-only     — technical factors, no text
  * sentiment-only — the daily sentiment factor, no price
  * combined        — both

Families:
  * LogisticRegression — the simple, transparent baseline you should always try first.
  * XGBoost            — gradient-boosted trees, what quant teams actually reach for on
                          tabular multi-factor data; it can pick up nonlinear interactions
                          a linear model can't.

Standardization: the linear model is sensitive to feature scale, so we fit a
StandardScaler on the TRAIN split only and reuse it — fitting the scaler on dev/test
would leak distributional information from the future into training.

Nothing here touches the test split — that is reserved for evaluate.py, run once at the
very end.
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


@dataclass
class TrainedModel:
    """A fitted estimator plus the scaler and columns it expects, so evaluate.py can apply
    it to the test split identically."""
    name: str            # e.g. "xgboost/combined"
    family: str          # "logreg" | "xgboost"
    feature_set: str     # "price_only" | "sentiment_only" | "combined"
    columns: list[str]
    scaler: object | None
    estimator: object


def load_combined() -> pd.DataFrame:
    if not config.COMBINED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{config.COMBINED_FEATURES_PATH} not found. Run features/build_features.py first."
        )
    return pd.read_parquet(config.COMBINED_FEATURES_PATH)


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)


def _fit_logreg(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    clf.fit(X_train, y_train)
    return clf


def _fit_xgboost(X_train, y_train):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,          # shallow trees resist overfitting on a noisy, near-random target
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )
    clf.fit(X_train, y_train)
    return clf


def train_all(df: pd.DataFrame) -> list[TrainedModel]:
    """Fit every (family x feature-set) combination on the train split. The scaler for the
    linear models is fit on train only to avoid leaking future statistics."""
    from sklearn.preprocessing import StandardScaler

    train = split_frame(df, "train")
    y_train = train[config.TARGET_COL].to_numpy()

    models: list[TrainedModel] = []
    for fs_name, cols in FEATURE_SETS.items():
        X_train_raw = train[cols].to_numpy(dtype=float)

        # Linear model: standardized inputs.
        scaler = StandardScaler().fit(X_train_raw)
        X_train_std = scaler.transform(X_train_raw)
        logreg = _fit_logreg(X_train_std, y_train)
        models.append(TrainedModel(
            name=f"logreg/{fs_name}", family="logreg", feature_set=fs_name,
            columns=cols, scaler=scaler, estimator=logreg,
        ))

        # Trees: scale-invariant, so no scaler needed.
        xgb = _fit_xgboost(X_train_raw, y_train)
        models.append(TrainedModel(
            name=f"xgboost/{fs_name}", family="xgboost", feature_set=fs_name,
            columns=cols, scaler=None, estimator=xgb,
        ))
        print(f"  trained {fs_name}: logreg + xgboost")

    return models


def predict(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    """Apply a trained model to a frame, using the exact columns/scaler it was fit with."""
    X = df[model.columns].to_numpy(dtype=float)
    if model.scaler is not None:
        X = model.scaler.transform(X)
    return model.estimator.predict(X)


if __name__ == "__main__":
    # Standalone sanity run: train and report train-split accuracy (not a real metric,
    # just confirms the models fit). Real evaluation lives in evaluate.py.
    from sklearn.metrics import accuracy_score

    frame = load_combined()
    print("Split sizes:", frame["split"].value_counts().to_dict())
    trained = train_all(frame)
    tr = split_frame(frame, "train")
    for m in trained:
        acc = accuracy_score(tr[config.TARGET_COL], predict(m, tr))
        print(f"  {m.name:24s} train-acc={acc:.3f}")
