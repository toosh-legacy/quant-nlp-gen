"""Conformal prediction intervals — turning a point forecast into an honest range.

A single number implies a precision the model does not have. "Volatility will be 14.2%"
invites the reader to plan around 14.2%; "12.1% to 17.4%, and that range has covered the
outcome 90% of the time" is both more useful and more truthful.

**Split conformal prediction** gives that with no distributional assumption. Fit the model,
measure its absolute errors on a calibration slice the model never saw, and take the
appropriate quantile of those errors as the interval half-width. The resulting interval has
a finite-sample coverage guarantee under exchangeability — no normality, no correct model
specification, no asymptotics.

Two adaptations matter here:

* **The calibration slice is chronological**, taken from the end of the training period.
  Financial data is not exchangeable across time, so the guarantee is approximate; using
  the most recent pre-cutoff data makes the approximation as close as it can be. The honest
  test is the *realized* coverage on out-of-sample data, which is what `evaluate_coverage`
  reports.
* **Scaled (Mondrian) residuals.** Errors are larger when volatility is high, so a single
  fixed width would be too wide in calm markets and too narrow in stressed ones — exactly
  backwards. Residuals are normalised by a difficulty estimate before quantiling, so the
  interval breathes with conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config

# Nominal coverage of the shipped interval.
COVERAGE = 0.80
CALIB_DAYS = 504          # ~2 years of trailing calibration


def _difficulty(df: pd.DataFrame, h: int) -> np.ndarray:
    """A per-row proxy for how hard this day is to forecast.

    Uses the dispersion between the series' short and long realized-vol measures: when the
    volatility term structure is steep the level is moving fast and forecasts are less
    reliable. Kept strictly causal — it only reads features, never the target.
    """
    if "rv_slope_wm" in df.columns:
        d = 1.0 + df["rv_slope_wm"].abs().to_numpy(float)
    else:
        d = np.ones(len(df))
    return np.clip(d, 0.25, 4.0)


def calibrate(model, train_df: pd.DataFrame, h: int,
              coverage: float = COVERAGE) -> dict:
    """Compute the conformal half-width from the tail of the training period."""
    calib = train_df.sort_values("date").tail(CALIB_DAYS)
    if len(calib) < 50:
        return {"q": float("nan"), "coverage": coverage, "n_calib": len(calib)}

    pred = model.predict(calib)
    y = calib[config.fwd_col(h)].to_numpy(float)
    diff = _difficulty(calib, h)
    scores = np.abs(y - pred) / diff

    # The finite-sample-valid quantile level for n calibration points.
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return {"q": float(np.quantile(scores, level)), "coverage": coverage, "n_calib": n}


def interval(pred: np.ndarray, df: pd.DataFrame, cal: dict, h: int):
    """(lower, upper) log-vol bounds for each prediction."""
    if not np.isfinite(cal.get("q", np.nan)):
        return None, None
    half = cal["q"] * _difficulty(df, h)
    return np.asarray(pred, float) - half, np.asarray(pred, float) + half


def adaptive_intervals(oos: pd.DataFrame, h: int, coverage: float = COVERAGE,
                       gamma: float = 0.02, window: int = 750) -> pd.DataFrame:
    """Adaptive conformal inference (Gibbs & Candes 2021) — fixes coverage under drift.

    Split conformal assumes exchangeability. Financial series plainly are not exchangeable,
    and the consequence is measurable: the static interval here covered 73% of outcomes
    against a nominal 80%. Rather than advertise a guarantee the data does not honour, the
    effective miss rate is *steered* online.

    The rule is one line: after each outcome is revealed, raise the target coverage if the
    interval missed and lower it if it covered,

        alpha <- alpha + gamma * (alpha_target - missed)

    which drives the long-run realized miss rate to the target regardless of how the
    distribution shifts. Widths breathe: intervals widen after a run of misses (turbulence)
    and tighten again once forecasts start landing.

    Strict causality: the outcome for day t is only observable at t+h, so the update at
    day t may use residuals no fresher than t-h. That lag is enforced explicitly below —
    without it the interval would be quietly using tomorrow's error to size today's band.
    """
    df = oos.sort_values("date").reset_index(drop=True).copy()
    y = df[config.fwd_col(h)].to_numpy(float)
    pred = df["pred"].to_numpy(float)
    diff = _difficulty(df, h)
    scores = np.abs(y - pred) / diff          # revealed only h rows later

    n = len(df)
    alpha_t = 1.0 - coverage
    target = 1.0 - coverage
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)

    for t in range(n):
        # Residuals whose outcome window has already closed by day t.
        avail_end = t - h
        if avail_end < 50:
            continue
        start = max(0, avail_end - window)
        recent = scores[start:avail_end]
        level = float(np.clip(1.0 - alpha_t, 0.01, 0.999))
        q = float(np.quantile(recent, level))
        half = q * diff[t]
        lo[t], hi[t] = pred[t] - half, pred[t] + half

        # Update using the most recently revealed outcome, not today's.
        j = avail_end
        if np.isfinite(scores[j]):
            q_j = float(np.quantile(scores[max(0, j - window):j], level)) if j > 50 else q
            missed = float(scores[j] > q_j)
            alpha_t = float(np.clip(alpha_t + gamma * (target - missed), 0.001, 0.5))

    df["pred_lo_adaptive"], df["pred_hi_adaptive"] = lo, hi
    return df


def evaluate_coverage(oos: pd.DataFrame, h: int) -> dict:
    """Did the interval actually cover the outcome as often as it promised?

    This is the number that matters. A conformal guarantee assumes exchangeability, which
    financial time series violate, so the claim is only worth what the out-of-sample
    coverage says it is.
    """
    def _one(lo_col: str, hi_col: str) -> dict:
        if lo_col not in oos.columns:
            return {}
        y = oos[config.fwd_col(h)].to_numpy(float)
        lo = oos[lo_col].to_numpy(float)
        hi = oos[hi_col].to_numpy(float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.sum() == 0:
            return {}
        covered = (y[ok] >= lo[ok]) & (y[ok] <= hi[ok])
        # Width in annualized-vol points, so it reads in the dashboard's units.
        ann = lambda v: np.exp(v) * np.sqrt(config.TRADING_DAYS) * 100.0
        width = ann(hi[ok]) - ann(lo[ok])
        return {
            "realized_coverage": float(covered.mean()),
            "mean_width_vol_pts": float(np.mean(width)),
            "median_width_vol_pts": float(np.median(width)),
            "n": int(ok.sum()),
        }

    out = {"nominal_coverage": COVERAGE}
    for tag, (lc, hc) in {"static": ("pred_lo", "pred_hi"),
                          "adaptive": ("pred_lo_adaptive", "pred_hi_adaptive")}.items():
        for k, v in _one(lc, hc).items():
            out[f"{tag}_{k}"] = v
    return out
