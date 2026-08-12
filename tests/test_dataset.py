"""Splitting, embargo and regime-threshold correctness.

These are three-line functions whose bugs are silent: an off-by-one in the embargo does
not raise, it just inflates every score by a little. So each gets an explicit test with a
hand-checkable expectation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volradar import config
from volradar.models import dataset as D


@pytest.fixture
def frame():
    dates = pd.bdate_range("2020-01-01", periods=60)
    h = 5
    return pd.DataFrame({
        "date": dates,
        "f1": np.arange(60, dtype=float),
        config.fwd_col(h): np.linspace(-4, -2, 60),
        config.persist_col(h): np.linspace(-4, -2, 60),
        config.label_date_col(h): pd.Series(dates).shift(-h),
    })


def test_embargo_excludes_rows_whose_outcome_crosses_the_cutoff(frame):
    """A row before the cutoff whose label lands after it must be dropped."""
    h, cutoff = 5, pd.Timestamp("2020-02-14")
    train = D.embargoed_train(frame, h, cutoff)

    assert (train["date"] < cutoff).all(), "kept a row dated on/after the cutoff"
    assert (train[config.label_date_col(h)] < cutoff).all(), \
        "kept a row whose outcome window crosses the cutoff — this leaks the test period"

    # And it must actually have dropped something, or the test proves nothing.
    naive = frame[frame["date"] < cutoff]
    assert len(train) < len(naive), "embargo removed no rows; it is not doing its job"


def test_embargo_is_stricter_for_longer_horizons(frame):
    """A 21-day outcome window crosses the boundary from further back than a 5-day one."""
    cutoff = pd.Timestamp("2020-02-14")
    dates = frame["date"]
    for h in (5, 10):
        frame[config.label_date_col(h)] = dates.shift(-h)
        frame[config.fwd_col(h)] = 0.0
    n5 = len(D.embargoed_train(frame, 5, cutoff))
    n10 = len(D.embargoed_train(frame, 10, cutoff))
    assert n10 < n5


def test_time_slice_is_half_open(frame):
    """[start, end) — a row exactly on `end` belongs to the NEXT slice, never both."""
    s = D.time_slice(frame, "2020-01-15", "2020-02-01")
    assert s["date"].min() >= pd.Timestamp("2020-01-15")
    assert s["date"].max() < pd.Timestamp("2020-02-01")

    a = D.time_slice(frame, None, "2020-02-01")
    b = D.time_slice(frame, "2020-02-01", None)
    assert len(a) + len(b) == len(frame), "slices overlap or lose rows at the boundary"


def test_regime_thresholds_come_only_from_training_data():
    h = 10
    train = pd.DataFrame({config.fwd_col(h): np.linspace(0, 100, 1000)})
    lo, hi = D.regime_thresholds(train, h)
    q_lo, q_hi = config.REGIME_QUANTILES
    assert lo == pytest.approx(train[config.fwd_col(h)].quantile(q_lo))
    assert hi == pytest.approx(train[config.fwd_col(h)].quantile(q_hi))


def test_to_regime_boundaries():
    """Values exactly on a threshold fall in the lower regime, consistently."""
    thr = (10.0, 20.0)
    got = D.to_regime([5, 10, 10.001, 20, 20.001, 99], thr)
    assert list(got) == [0, 0, 1, 1, 2, 2]


def test_labelled_drops_rows_with_any_missing_feature():
    h = 5
    df = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=5),
        "a": [1.0, np.nan, 3.0, 4.0, 5.0],
        "b": [1.0, 2.0, 3.0, np.nan, 5.0],
        config.fwd_col(h): [1.0, 1.0, 1.0, 1.0, np.nan],
        config.persist_col(h): [1.0] * 5,
    })
    out = D.labelled(df, ["a", "b"], h)
    assert len(out) == 2
    assert out[["a", "b"]].notna().all().all()
