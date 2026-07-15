"""Deeper, honesty-focused analyses that back up the headline horizon curve.

Three studies, all runnable via `python predictor/analysis.py`:

  1. Non-overlapping cross-check — re-run each horizon on non-overlapping windows (labels
     that don't share days at all), a stricter test that the overlapping-window result
     isn't an autocorrelation artefact.
  2. Conviction curve — calibrate probabilities and ask: if we only "trade" the most
     confident days, does accuracy rise? (Coverage vs accuracy trade-off.)
  3. Per-sector models — train a separate model per StockNet sector to see whether some
     sectors are more predictable than others (with a small-sample caveat).

These use the fast, well-behaved LogisticRegression/combined variant (no per-fold XGBoost
tuning) so the studies stay quick and reproducible. Train/dev are used for fitting and
calibration; test is scored once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from features.build_features import load_ticker_sectors  # noqa: E402
from predictor.evaluate import score  # noqa: E402
from predictor.train_predictor import embargoed_frame, load_combined, split_frame  # noqa: E402

COMBINED_COLS = config.PRICE_FEATURE_COLS + config.SENTIMENT_FEATURE_COLS


def _fit_logreg(train: pd.DataFrame, cols: list[str]):
    """Fit a scaler (train only) + balanced LogisticRegression. Returns (scaler, clf)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train[cols].to_numpy(dtype=float))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=config.RANDOM_SEED)
    clf.fit(scaler.transform(train[cols].to_numpy(dtype=float)), train["y"].to_numpy())
    return scaler, clf


def _proba(scaler, clf, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return clf.predict_proba(scaler.transform(frame[cols].to_numpy(dtype=float)))[:, 1]


# ---------------------------------------------------------------------------
# 1. Non-overlapping cross-check
# ---------------------------------------------------------------------------
def nonoverlap_subsample(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    """Keep every h-th row per ticker so consecutive kept rows are h trading days apart —
    their h-day label windows therefore never overlap. This removes the autocorrelation the
    embargo can only partly address, at the cost of ~1/h of the data."""
    f = frame.sort_values(["ticker", "date"])
    keep = f.groupby("ticker").cumcount() % h == 0
    return f[keep].reset_index(drop=True)


def nonoverlap_crosscheck(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in config.HORIZONS:
        frame = embargoed_frame(df, h)
        sub = nonoverlap_subsample(frame, h)
        train, test = split_frame(sub, "train"), split_frame(sub, "test")
        if len(train) < 50 or len(test) < 30:
            rows.append({"horizon": h, "n_train": len(train), "n_test": len(test),
                         "accuracy": float("nan"), "mcc": float("nan")})
            continue
        scaler, clf = _fit_logreg(train, COMBINED_COLS)
        preds = (_proba(scaler, clf, test, COMBINED_COLS) > 0.5).astype(int)
        m = score(test["y"].to_numpy(), preds)
        rows.append({"horizon": h, "n_train": len(train), "n_test": len(test),
                     "accuracy": m["accuracy"], "mcc": m["mcc"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Conviction curve (calibration + high-confidence subset)
# ---------------------------------------------------------------------------
def conviction_curve(df: pd.DataFrame, h: int, coverages=(1.0, 0.5, 0.25, 0.1)) -> pd.DataFrame:
    """Calibrate LogReg/combined probabilities on dev, then report accuracy when we act only
    on the most confident fraction of test days. If the model has any real edge, accuracy on
    the high-conviction subset should exceed accuracy on all days."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    frame = embargoed_frame(df, h)
    train, dev, test = (split_frame(frame, s) for s in ("train", "dev", "test"))
    scaler, clf = _fit_logreg(train, COMBINED_COLS)

    y_test = test["y"].to_numpy()
    raw_p = _proba(scaler, clf, test, COMBINED_COLS)

    # Calibrate on dev if dev is big enough; else fall back to raw probabilities.
    # sklearn >=1.6 calibrates an already-fitted model by wrapping it in FrozenEstimator
    # (the old cv="prefit" was removed).
    if len(dev) >= 100:
        from sklearn.frozen import FrozenEstimator

        X_dev = scaler.transform(dev[COMBINED_COLS].to_numpy(dtype=float))
        calib = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
        calib.fit(X_dev, dev["y"].to_numpy())
        X_test = scaler.transform(test[COMBINED_COLS].to_numpy(dtype=float))
        cal_p = calib.predict_proba(X_test)[:, 1]
        brier_raw = brier_score_loss(y_test, raw_p)
        brier_cal = brier_score_loss(y_test, cal_p)
        p = cal_p
    else:
        brier_raw = brier_score_loss(y_test, raw_p)
        brier_cal = float("nan")
        p = raw_p

    conf = np.abs(p - 0.5)  # distance from the decision boundary = confidence
    rows = []
    for cov in coverages:
        # Keep the top `cov` fraction most confident test days.
        k = max(1, int(round(cov * len(conf))))
        idx = np.argsort(conf)[::-1][:k]
        preds = (p[idx] > 0.5).astype(int)
        m = score(y_test[idx], preds)
        rows.append({"horizon": h, "coverage": cov, "n": k,
                     "accuracy": m["accuracy"], "mcc": m["mcc"]})
    out = pd.DataFrame(rows)
    out.attrs["brier_raw"] = brier_raw
    out.attrs["brier_cal"] = brier_cal
    return out


# ---------------------------------------------------------------------------
# 3. Per-sector models
# ---------------------------------------------------------------------------
def sector_analysis(df: pd.DataFrame, h: int, min_test: int = 60) -> pd.DataFrame:
    """Train a separate LogReg/combined model per sector and evaluate on that sector's test
    rows. Small sectors are noisy — reported with their sample sizes so the reader can weigh
    them accordingly."""
    sectors = load_ticker_sectors()
    frame = embargoed_frame(df, h)
    frame = frame.copy()
    frame["sector"] = frame["ticker"].map(sectors).fillna("Unknown")

    rows = []
    for sector, sdf in frame.groupby("sector"):
        train, test = split_frame(sdf, "train"), split_frame(sdf, "test")
        if len(train) < 200 or len(test) < min_test or test["y"].nunique() < 2:
            continue
        scaler, clf = _fit_logreg(train, COMBINED_COLS)
        preds = (_proba(scaler, clf, test, COMBINED_COLS) > 0.5).astype(int)
        m = score(test["y"].to_numpy(), preds)
        rows.append({"sector": sector, "n_train": len(train), "n_test": len(test),
                     "accuracy": m["accuracy"], "mcc": m["mcc"]})
    out = pd.DataFrame(rows).sort_values("mcc", ascending=False).reset_index(drop=True)
    return out


def _fmt(df: pd.DataFrame, floats=("accuracy", "mcc")) -> str:
    show = df.copy()
    for c in floats:
        if c in show.columns:
            show[c] = show[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
    return show.to_string(index=False)


def main() -> None:
    df = load_combined()

    print("=" * 70)
    print("1. NON-OVERLAPPING CROSS-CHECK (labels never share days)")
    print("=" * 70)
    print(_fmt(nonoverlap_crosscheck(df)))

    print("\n" + "=" * 70)
    print("2. CONVICTION CURVE (act only on the most confident test days)")
    print("=" * 70)
    for h in (5, 10):
        cc = conviction_curve(df, h)
        print(f"\nHorizon {h}d  (Brier raw={cc.attrs['brier_raw']:.4f}, "
              f"calibrated={cc.attrs['brier_cal']:.4f}):")
        print(_fmt(cc[["coverage", "n", "accuracy", "mcc"]]))

    print("\n" + "=" * 70)
    print("3. PER-SECTOR MODELS (horizon = 10d)")
    print("=" * 70)
    print(_fmt(sector_analysis(df, 10)))


if __name__ == "__main__":
    main()
