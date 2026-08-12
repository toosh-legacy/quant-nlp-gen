"""Metric correctness, against hand-computed values.

Every headline claim in the README is one of these functions applied to real data. If the
metric is wrong the claim is wrong, so they are checked against arithmetic done by hand
rather than against another library call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volradar.backtest import metrics as M


def test_r2_of_perfect_and_mean_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.regression_metrics(y, y)["r2"] == pytest.approx(1.0)
    # Predicting the mean everywhere is the definition of R^2 = 0.
    assert M.regression_metrics(y, np.full(4, y.mean()))["r2"] == pytest.approx(0.0)


def test_skill_vs_persistence_is_relative_error_reduction():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    persist = np.array([2.0, 2.0, 2.0, 2.0])     # MSE 4
    model = np.array([1.0, 1.0, 1.0, 1.0])       # MSE 1
    m = M.regression_metrics(y, model, persist)
    assert m["skill_vs_persistence"] == pytest.approx(0.75)   # 1 - 1/4

    # A model exactly as good as persistence scores zero skill, not zero R^2.
    assert M.regression_metrics(y, persist, persist)["skill_vs_persistence"] == \
        pytest.approx(0.0)


def test_rmse_and_mae_hand_computed():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 5.0])                # errors 1, 0, 2
    m = M.regression_metrics(y, p)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["mae"] == pytest.approx(1.0)


def test_direction_metrics_derive_from_levels():
    """Rise/fall is predicted level vs current level — not a separate classifier."""
    persist = np.array([0.0, 0.0, 0.0, 0.0])
    y = np.array([1.0, -1.0, 1.0, -1.0])         # truth: up, down, up, down
    pred = np.array([1.0, -1.0, -1.0, -1.0])     # got the third one wrong
    m = M.direction_metrics(y, pred, persist)
    assert m["dir_accuracy"] == pytest.approx(0.75)
    assert m["dir_baseline"] == pytest.approx(0.5)
    assert m["dir_lift"] == pytest.approx(0.25)


def test_direction_baseline_catches_a_constant_predictor():
    """A model that always says 'rise' must show zero MCC, whatever its accuracy."""
    persist = np.zeros(10)
    y = np.array([1.0] * 8 + [-1.0] * 2)         # 80% of days rise
    always_up = np.ones(10)
    m = M.direction_metrics(y, always_up, persist)
    assert m["dir_accuracy"] == pytest.approx(0.8)
    assert m["dir_baseline"] == pytest.approx(0.8)
    assert m["dir_lift"] == pytest.approx(0.0)
    assert m["dir_mcc"] == pytest.approx(0.0)    # accuracy 0.8 with no skill at all


def test_regime_metrics_report_majority_baseline():
    true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 0, 1, 1, 1, 2, 1])
    m = M.regime_metrics(true, pred)
    assert m["regime_accuracy"] == pytest.approx(6 / 8)
    assert m["regime_baseline"] == pytest.approx(4 / 8)
    assert m["stressed_recall"] == pytest.approx(0.5)


def test_summarize_folds_flags_unstable_edges():
    """mean must exceed 2 std to count as significant — a wobbly edge is not an edge."""
    folds = pd.DataFrame({
        "panel": ["m"] * 6, "horizon": [5, 5, 5, 10, 10, 10],
        "r2": [0.30, 0.32, 0.31, 0.30, -0.30, 0.60],
    })
    out = M.summarize_folds(folds, ["panel", "horizon"]).set_index("horizon")
    assert bool(out.loc[5, "significant"]) is True      # tight cluster
    assert bool(out.loc[10, "significant"]) is False    # same mean, huge spread
