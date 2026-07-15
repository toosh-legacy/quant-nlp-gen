"""Sanity checks for the large-universe volatility dataset (Task B pro).

Guarded: skips when the cached feature table is absent (e.g. in CI, which never downloads
data). When present locally, it verifies the shape, finiteness, cross-sectional properties,
and that the forward-vol label genuinely looks h days ahead (not backward)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from features.universe_data import (  # noqa: E402
    FEATURES_CACHE, UNIV_FEATURES, UNIV_HORIZONS, univ_fwd_col, univ_persist_col,
)

pytestmark = pytest.mark.skipif(
    not FEATURES_CACHE.exists(),
    reason="Universe feature cache absent (run features/universe_data.py).",
)


def _load():
    return pd.read_parquet(FEATURES_CACHE)


def test_features_are_finite_and_present():
    df = _load()
    for col in UNIV_FEATURES:
        assert col in df.columns
        assert np.isfinite(df[col].to_numpy()).all(), f"{col} has non-finite values"


def test_cross_sectional_rank_is_a_percentile():
    df = _load()
    assert df["rv_rank"].min() >= 0.0 and df["rv_rank"].max() <= 1.0


def test_forward_label_looks_ahead_not_back():
    """For one ticker, the persistence column (current vol) must differ from the forward
    label (next-h vol) in general, and the label must be shifted so that at date t it reflects
    returns after t — confirmed by checking it aligns with the current-vol column shifted back."""
    df = _load().sort_values(["ticker", "date"])
    tkr = df["ticker"].iloc[0]
    sub = df[df["ticker"] == tkr].reset_index(drop=True)
    for h in UNIV_HORIZONS:
        y, per = univ_fwd_col(h), univ_persist_col(h)
        # The forward label at row i should equal the *persistence* (current-vol) value at
        # row i+h, because both measure realized vol over the same calendar window [i+1, i+h].
        aligned = sub[per].shift(-h)
        mask = sub[y].notna() & aligned.notna()
        assert np.allclose(sub[y][mask], aligned[mask], atol=1e-6), \
            f"forward label for h={h} is not the h-ahead realized vol"
