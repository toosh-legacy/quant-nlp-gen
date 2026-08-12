"""Tests for the methodology layer: QLIKE, conformal intervals, pooled training.

The conformal test is the important one. Its interval width is derived from *realized
errors*, which are only observable h days after the forecast — so it is one edit away from
being the most subtle lookahead bug in the codebase. It gets an explicit causality test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volradar import config
from volradar.backtest import metrics as M
from volradar.models import conformal as C


# ---------------------------------------------------------------------------
# QLIKE
# ---------------------------------------------------------------------------
def test_qlike_is_minimized_by_a_perfect_forecast():
    y = np.log(np.array([0.008, 0.012, 0.02, 0.03]))
    perfect = M.qlike(y, y)
    for wrong in (y + 0.3, y - 0.3):
        assert M.qlike(y, wrong) > perfect


def test_qlike_hand_computed():
    """QLIKE(y, p) = mean(2p + exp(2(y - p))). At y == p it equals 2y + 1."""
    y = np.array([-4.0, -3.0])
    assert M.qlike(y, y) == pytest.approx(np.mean(2 * y + 1.0))


def test_qlike_punishes_under_forecasting_risk_harder():
    """The asymmetry is the point: missing high is safer than missing low."""
    y = np.array([np.log(0.02)] * 50)
    under = y - 0.5          # forecast too calm
    over = y + 0.5           # forecast too stormy
    assert M.qlike(y, under) > M.qlike(y, over)


def test_regression_metrics_expose_qlike_gain():
    y = np.array([-4.0, -3.5, -3.0, -4.2])
    persist = y + 0.4
    model = y + 0.05
    m = M.regression_metrics(y, model, persist)
    assert m["qlike"] < m["qlike_persistence"]
    assert m["qlike_gain"] == pytest.approx(m["qlike_persistence"] - m["qlike"])


# ---------------------------------------------------------------------------
# Conformal intervals
# ---------------------------------------------------------------------------
@pytest.fixture
def oos_frame():
    rng = np.random.default_rng(0)
    n = 900
    h = 10
    y = np.log(0.01) + np.cumsum(rng.normal(0, 0.02, n))
    return pd.DataFrame({
        "date": pd.bdate_range("2015-01-01", periods=n),
        config.fwd_col(h): y,
        "pred": y + rng.normal(0, 0.25, n),
        "rv_slope_wm": rng.normal(0, 0.1, n),
        "horizon": h,
    })


def test_adaptive_interval_never_uses_an_unrevealed_outcome(oos_frame):
    """Changing an outcome must not move any interval that closes before it is known.

    The interval at day t may only use residuals whose h-day window has already closed,
    i.e. up to t-h. Perturbing the target at row t must leave every interval at rows
    <= t + h untouched.
    """
    h = 10
    base = C.adaptive_intervals(oos_frame, h)

    tampered = oos_frame.copy()
    t = 500
    tampered.loc[t, config.fwd_col(h)] += 5.0        # a massive future surprise
    after = C.adaptive_intervals(tampered, h)

    upto = slice(0, t + h + 1)
    a = base["pred_lo_adaptive"].to_numpy()[upto]
    b = after["pred_lo_adaptive"].to_numpy()[upto]
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan]), \
        "an interval changed in response to an outcome not yet observable — lookahead"


def test_adaptive_interval_reacts_after_the_outcome_is_revealed(oos_frame):
    """Mirror test: once revealed, a burst of surprises SHOULD widen the next intervals.

    Checked on the window immediately after revelation, not the whole remainder — the
    controller is *supposed* to converge back to its target once forecasts start landing
    again, so a permanently inflated band would be the bug, not the fix.
    """
    h = 10
    base = C.adaptive_intervals(oos_frame, h)
    tampered = oos_frame.copy()
    tampered.loc[400:420, config.fwd_col(h)] += 3.0
    after = C.adaptive_intervals(tampered, h)

    width = lambda d: (d["pred_hi_adaptive"] - d["pred_lo_adaptive"]).to_numpy()
    near = slice(432, 520)          # outcomes for rows 400-420 are known from row 430
    wb, wa = width(base)[near], width(after)[near]
    ok = np.isfinite(wb) & np.isfinite(wa)
    assert np.nanmean(wa[ok]) > np.nanmean(wb[ok]), \
        "intervals ignored a revealed regime change"

    # ...and it should settle back down rather than stay permanently wide.
    tail = slice(850, 900)
    wb_t, wa_t = width(base)[tail], width(after)[tail]
    ok_t = np.isfinite(wb_t) & np.isfinite(wa_t)
    assert np.nanmean(wa_t[ok_t]) < 1.15 * np.nanmean(wb_t[ok_t]), \
        "controller never returned to target after the shock passed"


def test_intervals_are_ordered_and_centred(oos_frame):
    h = 10
    out = C.adaptive_intervals(oos_frame, h)
    ok = out["pred_lo_adaptive"].notna()
    lo = out.loc[ok, "pred_lo_adaptive"]
    hi = out.loc[ok, "pred_hi_adaptive"]
    pred = out.loc[ok, "pred"]
    assert (hi > lo).all()
    assert np.allclose((lo + hi) / 2, pred)


def test_coverage_evaluation_counts_correctly():
    h = 10
    df = pd.DataFrame({
        config.fwd_col(h): [0.0, 0.0, 0.0, 0.0],
        "pred_lo_adaptive": [-1.0, -1.0, 0.5, -1.0],     # third interval misses
        "pred_hi_adaptive": [1.0, 1.0, 1.5, 1.0],
    })
    cov = C.evaluate_coverage(df, h)
    assert cov["adaptive_realized_coverage"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Pooled training wiring
# ---------------------------------------------------------------------------
def test_pooled_panels_declare_their_tags():
    """A pooled-trained panel must carry the index/sector indicators in its feature list."""
    from volradar.features.assemble import all_features
    for panel, source in config.TRAINING_SOURCE.items():
        feats = all_features(panel)
        if source == "pooled":
            assert all(t in feats for t in config.POOL_TAGS), \
                f"{panel} trains pooled but is missing {config.POOL_TAGS}"
        else:
            assert not any(t in feats for t in config.POOL_TAGS)
