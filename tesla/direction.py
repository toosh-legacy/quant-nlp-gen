"""TSLA direction prediction — the honest, expected-to-be-weak result.

Same question as the parent project ("how far ahead can we predict up/down?"),
same discipline (chronological walk-forward folds, overlapping-window embargo,
MCC vs a majority baseline), but on ONE stock over 2015-2026. We report MCC
mean +/- std across yearly folds, per horizon, for three feature variants:

  * price      — technical + HAR + range-based factors
  * sentiment  — the DistilBERT Musk/Tesla-tweet factor only
  * combined   — both

Prior expectation (stated before the numbers): a single stock strips out the
cross-sectional signal and leaves far fewer rows than the 88-stock run, so we
expect direction MCC near zero with WIDE confidence intervals — i.e. no reliable
edge. The value is measuring that honestly, not manufacturing one.

Run:
    python tesla/direction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from predictor.analysis import _fit_logreg, _proba  # noqa: E402  (reuse parent helpers)
from predictor.evaluate import score  # noqa: E402
from tesla.features import (  # noqa: E402
    DIR_PRICE_FEATURES, SENT_FEATURES, build_features,
)

FEATURE_SETS = {
    "price": DIR_PRICE_FEATURES,
    "sentiment": SENT_FEATURES,
    "combined": DIR_PRICE_FEATURES + SENT_FEATURES,
}

# Yearly out-of-sample folds. Train = all prior rows whose h-day outcome also lands
# before the fold start (embargo); test = the fold year.
FOLD_YEARS = list(range(2017, 2027))


def _embargoed_train(df: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    ld = config.label_date_col(h)
    mv = config.movement_col(h)
    return df[df[mv].notna() & (df["date"] < cutoff) & (df[ld].notna()) & (df[ld] < cutoff)]


def walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(horizon, feature-set) yearly walk-forward. Returns per-fold rows and a
    mean +/- std summary (incl. the majority baseline accuracy)."""
    per_fold = []
    for h in config.TSLA_DIR_HORIZONS:
        mv = config.movement_col(h)
        sub = df[df[mv].notna()].copy()
        sub["y"] = sub[mv].astype(int)
        for yr in FOLD_YEARS:
            cutoff = pd.Timestamp(f"{yr}-01-01")
            nxt = pd.Timestamp(f"{yr + 1}-01-01")
            train = _embargoed_train(sub, h, cutoff)
            test = sub[(sub["date"] >= cutoff) & (sub["date"] < nxt)]
            if len(train) < 150 or len(test) < 20 or test["y"].nunique() < 2 or train["y"].nunique() < 2:
                continue
            base = float(max(test["y"].mean(), 1 - test["y"].mean()))
            for fs, cols in FEATURE_SETS.items():
                scaler, clf = _fit_logreg(train.assign(y=train["y"]), cols)
                preds = (_proba(scaler, clf, test, cols) > 0.5).astype(int)
                m = score(test["y"].to_numpy(), preds)
                per_fold.append({"horizon": h, "features": fs, "year": yr,
                                 "n_test": len(test), "baseline": base,
                                 "accuracy": m["accuracy"], "mcc": m["mcc"]})

    pf = pd.DataFrame(per_fold)
    rows = []
    for h in config.TSLA_DIR_HORIZONS:
        for fs in FEATURE_SETS:
            s = pf[(pf["horizon"] == h) & (pf["features"] == fs)]
            if s.empty:
                continue
            rows.append({"horizon": h, "features": fs, "n_folds": len(s),
                         "baseline_acc": s["baseline"].mean(),
                         "acc_mean": s["accuracy"].mean(),
                         "mcc_mean": s["mcc"].mean(), "mcc_std": s["mcc"].std(ddof=0)})
    return pf, pd.DataFrame(rows)


def main() -> None:
    df = build_features()
    print(f"TSLA direction: {len(df)} rows, "
          f"{df['date'].min().date()}..{df['date'].max().date()}\n")

    pf, summary = walk_forward(df)
    print("=== Walk-forward direction: MCC mean +/- std across yearly folds ===")
    print("  (baseline = majority-class accuracy; MCC ~ 0 within +/-std => no reliable edge)\n")
    for h in config.TSLA_DIR_HORIZONS:
        print(f"h = {h:>2} trading day(s):")
        for fs in FEATURE_SETS:
            r = summary[(summary["horizon"] == h) & (summary["features"] == fs)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"    {fs:9s} ({int(r['n_folds'])} folds):  "
                  f"acc {r['acc_mean']:.3f} (base {r['baseline_acc']:.3f})   "
                  f"MCC {r['mcc_mean']:+.3f} +/- {r['mcc_std']:.3f}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pf.to_csv(config.TSLA_DIRECTION_RESULTS, index=False)
    print(f"\nWrote per-fold direction results -> {config.TSLA_DIRECTION_RESULTS}")


if __name__ == "__main__":
    main()
