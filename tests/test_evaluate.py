"""Tests that the evaluation metrics are computed correctly. If the scoreboard is wrong,
every conclusion in the write-up is wrong — so we pin accuracy/F1/MCC against known values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from predictor.evaluate import score  # noqa: E402


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
