"""The factor families: do they build what they promise, and only from the past?

`feature_families()` is a contract. Every name it advertises has to exist as a real column
after a build, or `dataset.labelled` silently drops every row and the family's verdict in
the selection study becomes "no data" wearing the mask of "no signal". That is precisely
the failure that made the sector panel's family selection meaningless once it switched to
pooled training, so it gets a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volradar import config
from volradar.features import factors
from volradar.features.core import build_core

VOL_INDICES = config.VIX_TICKERS


def _prices(tickers: list[str], n: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i, t in enumerate(tickers):
        vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        ret = rng.normal(0, 1, n) * vol
        close = 100 * np.exp(np.cumsum(ret))
        hi = close * (1 + np.abs(rng.normal(0, 0.005, n)))
        lo = close * (1 - np.abs(rng.normal(0, 0.005, n)))
        parts.append(pd.DataFrame({
            "date": pd.bdate_range("2015-01-01", periods=n),
            "ticker": t,
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": np.maximum(hi, close), "low": np.minimum(lo, close),
            "close": close, "volume": rng.integers(1e6, 5e6, n).astype(float),
        }))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def blocks():
    market_core = build_core(_prices(["SPY"], seed=0))
    sector_core = build_core(_prices(["XLK", "XLF", "XLE"], seed=2))
    return {
        "vix": factors.build_vix(_prices(VOL_INDICES, seed=3), market_core),
        "breadth": factors.build_breadth(_prices([f"S{i}" for i in range(12)], seed=4)),
        "macro": factors.build_macro(_prices(config.MACRO_TICKERS, seed=5)),
        "sector_xs": factors.build_sector_xs(sector_core),
        "_market_core": market_core,
    }


@pytest.mark.parametrize("family,columns", [
    ("vix", factors.VIX_FEATURES),
    ("vix", factors.VIX_SHORT_FEATURES),
    ("breadth", factors.BREADTH_FEATURES),
    ("macro", factors.MACRO_FEATURES),
    ("sector_xs", factors.SECTOR_XS_FEATURES),
])
def test_every_advertised_column_exists(blocks, family, columns):
    got = blocks[family]
    missing = [c for c in columns if c not in got.columns]
    assert not missing, f"{family} advertises {missing} but does not build them"


@pytest.mark.parametrize("family", ["vix", "breadth", "macro", "sector_xs"])
def test_no_column_is_entirely_missing(blocks, family):
    """An all-NaN column deletes every training row it touches, without an error."""
    got = blocks[family].drop(columns=["date"])
    empty = [c for c in got.columns if got[c].notna().sum() == 0]
    assert not empty, f"{family} produced all-NaN columns: {empty}"


def test_vix_block_covers_the_whole_complex(blocks):
    """The term structure needs four indices, not one; the spreads need three more."""
    cols = set(blocks["vix"].columns)
    assert {"vix_term", "vix_term_short", "vix_term_long", "vix_curve"} <= cols
    assert {"skew_log", "vvix_log", "vix_vol_pk"} <= cols
    assert {"rvx_vix_spread", "vxn_vix_spread", "ovx_log"} <= cols


def test_vix_curvature_is_the_difference_of_the_two_slopes(blocks):
    v = blocks["vix"].dropna(subset=["vix_curve", "vix_term_long", "vix_term_short"])
    assert len(v) > 100
    assert np.allclose(v["vix_curve"], v["vix_term_long"] - v["vix_term_short"], atol=1e-12)


def test_vix_features_do_not_use_the_future():
    """Perturb the last VIX bar; nothing on an earlier row may move."""
    market_core = build_core(_prices(["SPY"], seed=0))
    base = _prices(VOL_INDICES, seed=3)
    tampered = base.copy()
    last = tampered.index[tampered["ticker"] == "^VIX"][-1]
    tampered.loc[last, ["open", "high", "low", "close"]] *= 4.0

    cols = factors.VIX_FEATURES + factors.VIX_SHORT_FEATURES
    a = factors.build_vix(base, market_core).set_index("date")[cols]
    b = factors.build_vix(tampered, market_core).set_index("date")[cols]
    tamper_date = base.loc[last, "date"]
    a, b = a[a.index < tamper_date], b[b.index < tamper_date]

    both_nan = a.isna().to_numpy() & b.isna().to_numpy()
    x, y = a.to_numpy(float), b.to_numpy(float)
    assert np.allclose(x[~both_nan], y[~both_nan]), \
        "a VIX feature on a past row moved when a FUTURE bar changed — lookahead bias"


def test_variance_risk_premium_is_implied_minus_realized(blocks):
    """The sign convention matters: positive means implied sits ABOVE realized."""
    v = blocks["vix"].set_index("date")
    rv = blocks["_market_core"].set_index("date")["log_rv_m"]
    rv_ann = np.log(np.exp(rv) * np.sqrt(config.TRADING_DAYS) * 100.0 + config.EPS)
    expected = v["vix_log"] - rv_ann.reindex(v.index)
    ok = expected.notna() & v["vix_rv_gap"].notna()
    assert ok.sum() > 100
    assert np.allclose(v.loc[ok, "vix_rv_gap"], expected[ok], atol=1e-12)


def test_average_correlation_stays_in_bounds(blocks):
    """`avg_corr` is reconstructed from a variance identity, which can overshoot."""
    c = blocks["breadth"]["avg_corr"].dropna()
    assert len(c) > 100
    assert c.between(-1.0, 1.0).all()
