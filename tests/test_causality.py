"""The cardinal test: no feature may depend on the future.

Lookahead bias is the failure mode that makes a worthless model look brilliant, and it is
almost invisible by inspection once a pipeline has rolling windows, groupbys and shifts in
it. So we test it behaviourally: build features twice from data that differs only in the
LAST bar, and assert that every feature on every earlier row is bit-for-bit identical.

If any rolling window is accidentally centred, any shift has the wrong sign, or any
normalisation is computed over the whole sample, this test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volradar import config
from volradar.features.core import CORE_FEATURES, build_core


def _synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Two tickers of plausible OHLCV data with volatility clustering."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in ["AAA", "BBB"]:
        vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
        ret = rng.normal(0, 1, n) * vol
        close = 100 * np.exp(np.cumsum(ret))
        high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
        frames.append(pd.DataFrame({
            "date": pd.bdate_range("2015-01-01", periods=n),
            "ticker": t,
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": np.maximum(high, close), "low": np.minimum(low, close),
            "close": close,
            "volume": rng.integers(1e6, 5e6, n).astype(float),
        }))
    return pd.concat(frames, ignore_index=True)


def test_no_lookahead_in_features():
    """Perturbing the final bar must not change any feature on any earlier row."""
    base = _synthetic()
    tampered = base.copy()
    last = tampered.index[tampered["ticker"] == "AAA"][-1]
    tampered.loc[last, ["open", "high", "low", "close"]] *= 3.0
    tampered.loc[last, "volume"] *= 10.0

    f_base = build_core(base)
    f_tamp = build_core(tampered)

    # Compare every row except the tampered one itself.
    mask = ~((f_base["ticker"] == "AAA") & (f_base["date"] == base.loc[last, "date"]))
    a = f_base.loc[mask, CORE_FEATURES].to_numpy(float)
    b = f_tamp.loc[mask, CORE_FEATURES].to_numpy(float)

    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], equal_nan=False), \
        "a feature on a past row changed when a FUTURE bar was modified — lookahead bias"


def test_target_does_depend_on_future():
    """The mirror image: the target MUST move when the future changes.

    Without this, a bug that made the target constant would pass the test above and look
    like perfect causal hygiene.
    """
    base = _synthetic()
    tampered = base.copy()
    idx = tampered.index[tampered["ticker"] == "AAA"][-5:]
    tampered.loc[idx, "close"] *= 1.5

    h = 10
    f_base = build_core(base).set_index(["ticker", "date"])
    f_tamp = build_core(tampered).set_index(["ticker", "date"])
    col = config.fwd_col(h)
    diff = (f_base[col] - f_tamp[col]).abs()
    assert diff.max() > 1e-9, "forward target ignored a change in future prices"


def test_persistence_is_backward_looking():
    """The persistence baseline must equal a purely trailing realized vol."""
    df = build_core(_synthetic())
    h = 10
    g = df[df["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    ret = g["close"].pct_change()
    manual = np.log(np.sqrt((ret ** 2).rolling(h).mean()) + config.EPS)
    got = g[config.persist_col(h)]
    ok = manual.notna() & got.notna()
    assert np.allclose(manual[ok], got[ok], atol=1e-9)


def test_forward_target_matches_manual_shift():
    """The target for row t must be the realized vol of days t+1..t+h, and nothing else."""
    df = build_core(_synthetic())
    h = 5
    g = df[df["ticker"] == "BBB"].sort_values("date").reset_index(drop=True)
    ret = g["close"].pct_change()
    manual = np.log(np.sqrt((ret ** 2).rolling(h).mean()).shift(-h) + config.EPS)
    got = g[config.fwd_col(h)]
    ok = manual.notna() & got.notna()
    assert ok.sum() > 100
    assert np.allclose(manual[ok], got[ok], atol=1e-9)


@pytest.mark.parametrize("h", config.HORIZONS)
def test_label_date_is_h_bars_ahead(h):
    """The embargo relies on label_date being exactly h rows ahead within a ticker."""
    df = build_core(_synthetic())
    g = df[df["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    expected = g["date"].shift(-h)
    got = g[config.label_date_col(h)]
    ok = expected.notna() & got.notna()
    assert (expected[ok] == got[ok]).all()
