"""Tests that the evaluation metrics are computed correctly. If the scoreboard is wrong,
every conclusion in the write-up is wrong — so we pin accuracy/F1/MCC against known values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from predictor.analysis import nonoverlap_subsample  # noqa: E402
from predictor.evaluate import score  # noqa: E402
from predictor.train_predictor import embargoed_frame  # noqa: E402


def test_perfect_prediction():
    y = [0, 1, 1, 0, 1]
    m = score(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["mcc"] == 1.0


def test_all_wrong_binary_gives_mcc_minus_one():
    y_true = [0, 1, 0, 1]
    y_pred = [1, 0, 1, 0]
    m = score(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert np.isclose(m["mcc"], -1.0)


def test_known_confusion_matrix_values():
    """Hand-computed case. y_true / y_pred below give:
        TP=2, TN=1, FP=1, FN=1  (positive class = 1)
        accuracy = 3/5 = 0.6
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
            = (2*1 - 1*1)/sqrt(3*3*2*2) = 1/6 ≈ 0.16667
    """
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]
    m = score(y_true, y_pred)
    assert np.isclose(m["accuracy"], 0.6)
    assert np.isclose(m["mcc"], 1.0 / 6.0, atol=1e-6)


def test_majority_class_predictor_has_zero_mcc():
    """Predicting a single class everywhere is uncorrelated with truth -> MCC 0, even when
    accuracy looks respectable. This is exactly why the project reports MCC alongside
    accuracy."""
    y_true = [1, 1, 1, 0, 0]  # 60% positive
    y_pred = [1, 1, 1, 1, 1]  # always predict up
    m = score(y_true, y_pred)
    assert np.isclose(m["accuracy"], 0.6)
    assert np.isclose(m["mcc"], 0.0)


def test_embargo_purges_train_dev_but_lets_test_labels_run_past_window():
    """The overlapping-window embargo: train/dev rows are kept only if their h-day outcome
    stays in the same split (so a training label never overlaps a later split). Test rows are
    kept whenever the label is defined — their outcome may run past the short test window
    using real later prices, which is not contamination since nothing trains on test."""
    h = config.DEFAULT_HORIZON
    mv, ls = config.movement_col(h), config.label_split_col(h)
    df = pd.DataFrame({
        "split": ["train", "train", "dev",  "dev",  "test", "test", "test"],
        ls:      ["train", "dev",   "dev",  "test", "test", np.nan, np.nan],
        mv:      [1.0,      0.0,     1.0,    0.0,    1.0,    0.0,    np.nan],
    })
    out = embargoed_frame(df, h)
    # kept: train->train, dev->dev, test->test, test->(past window, label defined).
    # dropped: train->dev, dev->test (boundary-crossers), and the test row with no label.
    assert out["split"].tolist() == ["train", "dev", "test", "test"]
    assert out["y"].tolist() == [1, 1, 1, 0]


def test_nonoverlap_subsample_keeps_every_hth_row_per_ticker():
    """The non-overlapping cross-check must keep rows h apart within each ticker, so their
    h-day label windows don't overlap — and it must not bleed across tickers."""
    h = 5
    dates = pd.date_range("2014-01-01", periods=12, freq="D")
    df = pd.DataFrame({
        "ticker": ["AAA"] * 12 + ["BBB"] * 12,
        "date": list(dates) + list(dates),
        "y": list(range(12)) + list(range(12)),
    })
    out = nonoverlap_subsample(df, h)
    # Per ticker we keep positions 0, 5, 10 -> the y values 0, 5, 10.
    for tkr in ("AAA", "BBB"):
        kept = out[out["ticker"] == tkr]["y"].tolist()
        assert kept == [0, 5, 10]
