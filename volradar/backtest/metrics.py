"""Metric definitions, in one place so every table computes them identically.

The reporting rules this project follows:

* **Never report a score without its baseline.** R^2 is already relative to the mean, but
  volatility's real baseline is persistence, so we also report *skill vs persistence*:
  how much of persistence's remaining error the model removed.
* **Report spread, not just the mean.** A single window's R^2 can be luck; the
  walk-forward mean +/- std is what decides whether an edge is real.
* **Regime accuracy sits next to the majority baseline**, and MCC next to accuracy, so
  class imbalance cannot masquerade as skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, matthews_corrcoef, mean_absolute_error,
                             r2_score)


def qlike(y_log_vol, pred_log_vol) -> float:
    """Mean QLIKE loss — the volatility-native alternative to squared error.

    QLIKE(sigma^2, h) = log(h) + sigma^2/h. With log daily volatility inputs, h = exp(2p)
    and sigma^2 = exp(2y), giving 2p + exp(2(y - p)).

    Two reasons it belongs next to R^2. It is **scale-aware**: squared error on logs
    punishes a given miss identically at 10% and 60% volatility, while QLIKE penalises
    under-forecasting risk harder than over-forecasting it — the asymmetry a risk user
    actually faces. And it is **robust to a noisy proxy**: realized volatility is only an
    estimate of true volatility, and QLIKE is one of the few losses whose ranking of
    forecasts survives that noise without bias (Patton 2011). Lower is better.
    """
    y = np.asarray(y_log_vol, float)
    p = np.asarray(pred_log_vol, float)
    d = np.clip(2.0 * (y - p), -20, 20)          # guard the exponential against outliers
    return float(np.mean(2.0 * p + np.exp(d)))


def regression_metrics(y_true, y_pred, y_persist=None) -> dict:
    """R^2, RMSE, MAE, QLIKE, and skill relative to the persistence baseline."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qlike": qlike(y_true, y_pred),
    }
    if y_persist is not None:
        yp = np.asarray(y_persist, float)
        mse_p = float(np.mean((y_true - yp) ** 2))
        out["r2_persistence"] = float(r2_score(y_true, yp))
        # Fraction of the persistence baseline's squared error that the model removed.
        out["skill_vs_persistence"] = float(1.0 - np.mean((y_true - y_pred) ** 2) / mse_p) \
            if mse_p > 0 else np.nan
        # Same idea under the volatility-native loss: how much QLIKE the model removed.
        out["qlike_persistence"] = qlike(y_true, yp)
        out["qlike_gain"] = out["qlike_persistence"] - out["qlike"]
    return out


def direction_metrics(y_true, y_pred, y_persist) -> dict:
    """'Will volatility rise?' — derived from the regression, not a separate model.

    The forecast says vol rises if the predicted level exceeds the current level. Reporting
    it this way keeps one model behind both the number and the arrow on the dashboard.
    """
    rise_true = (np.asarray(y_true, float) > np.asarray(y_persist, float)).astype(int)
    rise_pred = (np.asarray(y_pred, float) > np.asarray(y_persist, float)).astype(int)
    base = float(max(rise_true.mean(), 1 - rise_true.mean()))
    return {
        "dir_accuracy": float(accuracy_score(rise_true, rise_pred)),
        "dir_baseline": base,
        "dir_mcc": float(matthews_corrcoef(rise_true, rise_pred)),
        "dir_lift": float(accuracy_score(rise_true, rise_pred) - base),
    }


def regime_metrics(true_regime, pred_regime) -> dict:
    """3-class calm/normal/stressed accuracy against the majority baseline."""
    t = np.asarray(true_regime, int)
    p = np.asarray(pred_regime, int)
    counts = np.bincount(t, minlength=3)
    return {
        "regime_accuracy": float(accuracy_score(t, p)),
        "regime_baseline": float(counts.max() / max(len(t), 1)),
        "regime_mcc": float(matthews_corrcoef(t, p)),
        # Stressed periods are the ones that matter; recall on class 2 alone.
        "stressed_recall": float(((p == 2) & (t == 2)).sum() / max((t == 2).sum(), 1)),
    }


def summarize_folds(folds: pd.DataFrame, by: list[str], value: str = "r2") -> pd.DataFrame:
    """Collapse per-fold scores to mean +/- std, the form a reader can judge."""
    g = folds.groupby(by)[value]
    out = g.agg(["count", "mean", "std", "min", "max"]).reset_index()
    out = out.rename(columns={"count": "n_folds", "mean": f"{value}_mean",
                              "std": f"{value}_std", "min": f"{value}_min",
                              "max": f"{value}_max"})
    # A mean that is not at least two standard deviations from zero is not an edge.
    out["significant"] = out[f"{value}_mean"] > 2 * out[f"{value}_std"].fillna(0)
    return out
