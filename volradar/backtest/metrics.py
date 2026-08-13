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
    return float(np.mean(qlike_obs(y_log_vol, pred_log_vol)))


def qlike_obs(y_log_vol, pred_log_vol) -> np.ndarray:
    """Per-observation QLIKE. The Model Confidence Set needs the loss series, not its mean."""
    y = np.asarray(y_log_vol, float)
    p = np.asarray(pred_log_vol, float)
    d = np.clip(2.0 * (y - p), -20, 20)          # guard the exponential against outliers
    return 2.0 * p + np.exp(d)


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


# ---------------------------------------------------------------------------
# Comparing two forecasters
# ---------------------------------------------------------------------------
ADOPT_T = 2.0          # the bar a challenger must clear, used by every selection step


def paired_test(cand_folds, base_folds) -> dict:
    """Compare a candidate to an incumbent fold by fold.

    Comparing two marginal means and their separate standard errors is the wrong test here,
    and it fails in a specific way that matters: an *unstable* incumbent has a large
    standard error, which inflates the bar a challenger must clear — so the noisier the
    thing being replaced, the harder it becomes to replace it. Exactly backwards.

    Every candidate is evaluated on the same folds, so the differences are paired and most
    of the shared year-to-year variation cancels. The test is on the mean of the per-fold
    difference. `t > ADOPT_T` with a positive delta is the project-wide adoption bar; this
    is the only implementation of it.
    """
    a = np.asarray(cand_folds, float)
    b = np.asarray(base_folds, float)
    n = min(len(a), len(b))
    if n < 3:
        return {"delta": np.nan, "delta_se": np.nan, "t": np.nan, "wins": 0, "n": n}
    d = a[:n] - b[:n]
    se = float(d.std(ddof=1) / np.sqrt(n))
    mean = float(d.mean())
    if se > 0:
        t = mean / se
    elif mean == 0:
        t = np.nan          # identical on every fold — the same forecaster twice over
    else:
        # A difference with no spread at all: the candidate wins (or loses) by the same
        # margin on every single fold. That is maximal evidence, not missing evidence, so it
        # must not fall through to NaN and be silently treated as "inconclusive".
        t = np.inf if mean > 0 else -np.inf
    return {
        "delta": mean,
        "delta_se": se,
        "t": float(t),
        "wins": int((d > 0).sum()),
        "n": n,
    }


def adopts(verdict: dict) -> bool:
    """Does this paired verdict clear the adoption bar?"""
    t = verdict.get("t", np.nan)
    # `not np.isnan` rather than `isfinite`: an infinite t means a difference with zero
    # spread across folds, which adopts.
    return bool(not np.isnan(t) and t > ADOPT_T and verdict.get("delta", 0) > 0)


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason 2011)
# ---------------------------------------------------------------------------
def _block_bootstrap_indices(n: int, n_boot: int, block: int, rng) -> np.ndarray:
    """Circular block bootstrap index matrix, shape (n_boot, n).

    Blocks rather than individual observations because volatility forecast errors are
    strongly autocorrelated; an i.i.d. bootstrap would understate their variance and make
    every model look more distinguishable than it is.
    """
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.10,
                         n_boot: int = 1000, block: int = 20,
                         seed: int = 42) -> pd.DataFrame:
    """The set of models that cannot be distinguished from the best one.

    Picking the single lowest average loss and calling it the winner overstates what the
    data supports: with a few hundred overlapping observations, several models are usually
    within noise of each other, and which one comes first is close to a coin flip. The MCS
    answers the question that can actually be answered — *which models can be ruled out?* —
    and returns everything else. It is the more conservative statement, and the standard
    one in this literature.

    Parameters
    ----------
    losses : (n_observations x n_models) per-observation loss, lower is better
    alpha  : models with an MCS p-value below this are eliminated

    Returns a frame with one row per model: mean loss, MCS p-value, and whether it survived.
    """
    names = list(losses.columns)
    # Complete cases only, and the reported mean loss is computed on the same rows the test
    # uses — a mean over a different row set than the comparison would be a different number
    # from the one the ranking is based on.
    complete = losses.dropna()
    L = complete.to_numpy(float)
    n = len(L)
    if n < 30 or len(names) < 2:
        return pd.DataFrame({"model": names, "mean_loss": losses.mean().to_numpy(),
                             "mcs_pvalue": np.nan, "in_mcs": True})

    rng = np.random.default_rng(seed)
    idx = _block_bootstrap_indices(n, n_boot, block, rng)

    # Bootstrap the per-model mean losses ONCE. Every statistic below is a linear function
    # of these means, so re-drawing inside the elimination loop would only add cost and
    # sampling noise. Chunked because L[idx] materialises n_boot x n x m floats.
    boot_L = np.empty((n_boot, L.shape[1]))
    for s in range(0, n_boot, 50):
        e = min(s + 50, n_boot)
        boot_L[s:e] = L[idx[s:e]].mean(axis=1)
    mean_L = L.mean(axis=0)

    alive = list(range(len(names)))
    pvalue = {}
    running = 0.0
    while len(alive) > 1:
        # d_i = model i's loss minus the average loss of the surviving set.
        dbar = mean_L[alive] - mean_L[alive].mean()
        boot = boot_L[:, alive] - boot_L[:, alive].mean(axis=1, keepdims=True)
        sd = np.sqrt(np.maximum(((boot - dbar) ** 2).mean(axis=0), 1e-18))
        t = dbar / sd
        t_boot = (boot - dbar) / sd
        # Range statistic: is the WORST surviving model worse than bootstrap noise allows?
        p = float((t_boot.max(axis=1) > t.max()).mean())
        running = max(running, p)                        # p-values are made monotone
        worst = alive[int(np.argmax(t))]
        pvalue[worst] = running
        if running >= alpha:
            # Nothing more can be eliminated: everything still alive is in the set.
            for i in alive:
                pvalue.setdefault(i, running)
            break
        alive.remove(worst)
    for i in alive:
        pvalue.setdefault(i, 1.0)

    out = pd.DataFrame({
        "model": names,
        "mean_loss": complete.mean().reindex(names).to_numpy(float),
        "n_obs": n,
        "mcs_pvalue": [pvalue[i] for i in range(len(names))],
    })
    out["in_mcs"] = out["mcs_pvalue"] >= alpha
    return out.sort_values("mean_loss").reset_index(drop=True)


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
