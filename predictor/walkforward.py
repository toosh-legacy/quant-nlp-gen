"""Walk-forward evaluation — the single biggest credibility upgrade.

A single fixed test window gives one point estimate that could be luck. Here we roll several
consecutive out-of-sample windows through 2014-2016 and report the MEAN +/- STD of accuracy
and MCC per horizon. If "predictability rises with horizon" survives across windows, it's a
real pattern; if the spread is large, that honesty shows up as a wide confidence interval.

Each fold:
  * test  = feature days inside the window [ts, te); labels may extend past te using real
            later prices (not contamination — nothing trains on test).
  * train = feature days strictly before ts whose h-day outcome also lands before ts. That
            per-fold purge is the same overlapping-window embargo, applied at the fold
            boundary so a training label never overlaps the test window.

Uses LogisticRegression/combined (fast, no per-fold tuning) so this stays quick.

Run:
    python predictor/walkforward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from predictor.analysis import COMBINED_COLS, _fit_logreg, _proba  # noqa: E402
from predictor.evaluate import score  # noqa: E402
from predictor.train_predictor import load_combined  # noqa: E402

# Quarterly out-of-sample windows within the 2014-2016 feature range. The last one is the
# official StockNet test window.
FOLD_STARTS = ["2014-10-01", "2015-01-01", "2015-04-01", "2015-07-01", "2015-10-01"]
FOLD_LENGTH = pd.Timedelta(days=91)  # ~one quarter


def _fold_frames(df: pd.DataFrame, h: int, ts: pd.Timestamp):
    """Return (train, test) DataFrames with a 'y' target for one fold at horizon h."""
    te = ts + FOLD_LENGTH
    mv, ld = config.movement_col(h), config.label_date_col(h)
    labeled = df[mv].notna()

    # Train: everything before the window whose outcome also lands before the window (purge).
    train = df[labeled & (df["date"] < ts) & (df[ld] < ts)].copy()
    # Test: feature days inside the window with a defined label.
    test = df[labeled & (df["date"] >= ts) & (df["date"] < te)].copy()
    for part in (train, test):
        part["y"] = part[mv].astype(int)
    return train, test


def walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-fold results, per-horizon summary with mean +/- std)."""
    per_fold = []
    for h in config.HORIZONS:
        for ts_str in FOLD_STARTS:
            ts = pd.Timestamp(ts_str)
            train, test = _fold_frames(df, h, ts)
            if len(train) < 200 or len(test) < 30 or test["y"].nunique() < 2 or train["y"].nunique() < 2:
                continue  # fold too small / single-class — skip, noted via n_folds
            scaler, clf = _fit_logreg(train, COMBINED_COLS)
            preds = (_proba(scaler, clf, test, COMBINED_COLS) > 0.5).astype(int)
            m = score(test["y"].to_numpy(), preds)
            per_fold.append({"horizon": h, "fold": ts_str, "n_test": len(test),
                             "accuracy": m["accuracy"], "mcc": m["mcc"]})

    per_fold_df = pd.DataFrame(per_fold)
    summary_rows = []
    for h in config.HORIZONS:
        sub = per_fold_df[per_fold_df["horizon"] == h]
        if sub.empty:
            continue
        summary_rows.append({
            "horizon": h, "n_folds": len(sub),
            "acc_mean": sub["accuracy"].mean(), "acc_std": sub["accuracy"].std(ddof=0),
            "mcc_mean": sub["mcc"].mean(), "mcc_std": sub["mcc"].std(ddof=0),
        })
    return per_fold_df, pd.DataFrame(summary_rows)


def main() -> None:
    df = load_combined()
    per_fold, summary = walk_forward(df)

    print("=== Walk-forward per-fold (LogReg/combined) ===")
    show = per_fold.copy()
    for c in ("accuracy", "mcc"):
        show[c] = show[c].map(lambda v: f"{v:+.4f}")
    print(show.to_string(index=False))

    print("\n=== Walk-forward summary: mean +/- std across folds, per horizon ===")
    for _, r in summary.iterrows():
        print(f"  h={int(r['horizon']):>2}  ({int(r['n_folds'])} folds):  "
              f"accuracy {r['acc_mean']:.3f} +/- {r['acc_std']:.3f}   "
              f"MCC {r['mcc_mean']:+.3f} +/- {r['mcc_std']:.3f}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.DATA_DIR / "walkforward_results.csv"
    per_fold.to_csv(out, index=False)
    print(f"\nWrote per-fold results -> {out}")


if __name__ == "__main__":
    main()
