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


# ---------------------------------------------------------------------------
# The paired adoption test — the single bar every selection step in the project uses
# ---------------------------------------------------------------------------
def test_paired_test_needs_at_least_three_folds():
    """Two folds cannot support a t-statistic; it must decline rather than invent one."""
    v = M.paired_test([1.0, 2.0], [0.0, 1.0])
    assert np.isnan(v["t"])
    assert not M.adopts(v)


def test_paired_test_is_paired_not_marginal():
    """A consistent small edge must beat a noisy large one — that is the whole point.

    Both candidates average the same improvement over the incumbent. One does it every
    fold; the other swings wildly. An unpaired comparison of marginal means cannot tell
    them apart. The paired test must.
    """
    base = np.array([0.10, 0.40, 0.05, 0.30, 0.20, 0.35])
    steady = base + 0.02
    erratic = base + np.array([0.52, -0.48, 0.50, -0.46, 0.48, -0.44])
    assert M.paired_test(steady, base)["t"] > M.paired_test(erratic, base)["t"]


def test_paired_test_rejects_a_candidate_that_is_merely_lucky_once():
    base = np.zeros(8)
    cand = np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0])
    v = M.paired_test(cand, base)
    assert v["delta"] > 0, "the mean really is higher"
    assert not M.adopts(v), "but one lucky fold must not clear the adoption bar"


def test_adoption_bar_requires_a_positive_delta():
    """A significant result in the WRONG direction must never adopt."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.1, 12)
    worse = base - 0.5 + rng.normal(0, 0.01, 12)
    v = M.paired_test(worse, base)
    assert v["t"] < 0 and not M.adopts(v)


def test_a_difference_with_no_spread_adopts_rather_than_returning_nan():
    """Winning by the same margin on every fold is maximal evidence, not missing evidence.

    Dividing by a zero standard error is the obvious way to turn the strongest possible
    result into a NaN and silently file it as "inconclusive".
    """
    base = np.arange(8, dtype=float)
    v = M.paired_test(base + 0.5, base)
    assert v["t"] == np.inf and M.adopts(v)
    assert not M.adopts(M.paired_test(base - 0.5, base))


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------
def _loss_frame(means, n=600, sd=0.3, seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, sd, n)      # common shocks, as real forecast errors have
    return pd.DataFrame({name: shared + rng.normal(mu, sd * 0.3, n)
                         for name, mu in means.items()})


def test_mcs_eliminates_a_clearly_worse_model():
    out = M.model_confidence_set(_loss_frame({"good": 1.0, "awful": 3.0}), n_boot=300)
    kept = set(out[out["in_mcs"]]["model"])
    assert "good" in kept and "awful" not in kept


def test_mcs_keeps_models_it_cannot_separate():
    """Two models with the same loss must BOTH survive. The MCS's job is to decline to
    pick a winner when the data does not support one."""
    out = M.model_confidence_set(_loss_frame({"a": 1.0, "b": 1.0}), n_boot=300)
    assert out["in_mcs"].all()


def test_mcs_is_more_conservative_than_taking_the_minimum():
    """The set must never be empty, and must always contain the lowest-loss model."""
    out = M.model_confidence_set(
        _loss_frame({"a": 1.0, "b": 1.02, "c": 2.5}), n_boot=300)
    assert out["in_mcs"].any()
    assert bool(out.iloc[0]["in_mcs"]), "the best model must always be in the set"


def test_mcs_declines_on_too_few_observations():
    """With almost no data it must return everything, not a confident ranking."""
    out = M.model_confidence_set(_loss_frame({"a": 1.0, "b": 5.0}, n=10), n_boot=50)
    assert out["in_mcs"].all()
