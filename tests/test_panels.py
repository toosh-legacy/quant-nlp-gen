"""The panel contract: every candidate family must exist as a column on the frame it is
scored on.

This is the bug that motivated the test. `TRAINING_SOURCE["sector"] = "pooled"` made
`load_panel("sector")` return rows from `panel_pooled.parquet`, but the pooled builder only
ran `build_core` — so none of the factor families existed on it. The selection study and the
ablation both asked for columns that were not there. Nothing reported a missing family; the
studies simply could not run, and the published conclusion that "no external factor family
earned its place" had never been tested on the panel that actually ships.

A family that is absent looks exactly like a family that is useless. So the contract is
tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volradar import config
from volradar.features import assemble as A


def test_every_family_advertises_at_least_one_column():
    for panel in ("market", "sector"):
        for name, cols in A.feature_families(panel).items():
            assert cols, f"{panel}/{name} advertises no columns"


def test_candidate_columns_cover_every_non_core_family():
    advertised = {c for k, v in A.feature_families("sector").items() if k != "core"
                  for c in v}
    assert advertised == set(A._candidate_columns()), \
        "the rebuild check must look for exactly the columns a candidate can ask for"


def test_sector_only_families_are_declared():
    """`sector_rel` is undefined for a single stock, and the code has to know that."""
    assert A.SECTOR_ONLY_FAMILIES <= set(A.feature_families("sector"))
    assert "sector_rel" in A.SECTOR_ONLY_FAMILIES
    assert set(A.SECTOR_REL_FEATURES) == set(A.feature_families("sector")["sector_rel"])


def _fake_pooled(n=40):
    """A pooled frame in miniature: one index, two sectors, two single stocks."""
    rows = []
    for ticker, kind in [("SPY", "index"), ("XLK", "sector"), ("XLF", "sector"),
                         ("AAPL", "stock"), ("MSFT", "stock")]:
        f = pd.DataFrame({
            "date": pd.bdate_range("2015-01-01", periods=n),
            "ticker": ticker, "series_type": kind,
            "is_index": int(kind == "index"), "is_sector": int(kind == "sector"),
            "log_rv_m": np.linspace(-5, -4, n),
            "vix_log": np.linspace(2.5, 3.0, n),
        })
        # Exactly the pooled panel's shape: sector-relative columns exist only for sectors.
        for c in A.SECTOR_REL_FEATURES:
            f[c] = np.linspace(0, 1, n) if kind == "sector" else np.nan
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def test_training_frame_keeps_the_whole_cross_section_by_default(monkeypatch):
    monkeypatch.setattr(A, "build_pooled_panel", lambda *a, **k: _fake_pooled())
    monkeypatch.setitem(config.TRAINING_SOURCE, "sector", "pooled")
    out = A.training_frame("sector", ["log_rv_m", "vix_log"])
    assert out["ticker"].nunique() == 5, "pooling must use every series when it can"


def test_training_frame_narrows_when_a_sector_only_family_is_selected(monkeypatch):
    """The alternative is a silent 90% cut to the training set inside a dropna.

    With `sec_rv_rel` in the feature list, every stock and index row is NaN in a required
    column. `dataset.labelled` would drop them without a word, so the model would quietly
    train on a twentieth of the data it reports training on. Narrowing here makes that
    visible at the call site instead.
    """
    monkeypatch.setattr(A, "build_pooled_panel", lambda *a, **k: _fake_pooled())
    monkeypatch.setitem(config.TRAINING_SOURCE, "sector", "pooled")
    out = A.training_frame("sector", ["log_rv_m", "sec_rv_rel"])
    assert set(out["ticker"]) == {"XLK", "XLF"}
    assert out["sec_rv_rel"].notna().all(), "no row may survive with the family missing"


def test_training_frame_is_the_panel_itself_when_not_pooling(monkeypatch):
    monkeypatch.setitem(config.TRAINING_SOURCE, "market", "own")
    monkeypatch.setattr(A, "load_panel", lambda panel: pd.DataFrame({"ticker": ["SPY"]}))
    assert A.training_frame("market", ["log_rv_m"])["ticker"].tolist() == ["SPY"]
