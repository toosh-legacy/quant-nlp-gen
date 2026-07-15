"""Tests for the TSLA single-stock deep dive: no-lookahead, intraday-RV
construction, forward-label geometry, threshold scaling, and metric wiring.

The construction/leakage tests use synthetic data so they always run (incl. CI,
which never downloads). The tests that read the built table are skipped when the
feature cache is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from predictor.evaluate import score  # noqa: E402
from tesla.fetch_prices import realized_vol_from_intraday  # noqa: E402
from tesla.features import (  # noqa: E402
    DIR_PRICE_FEATURES, VOL_INTRADAY_FEATURES, _price_and_vol_features,
    cc_fwd_col, cc_persist_col, iv_fwd_col, iv_persist_col,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _synthetic_daily(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(0, 0.03, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"date": dates, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def _synthetic_intraday(sessions: int = 3, bars: int = 7, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(sessions):
        day = pd.Timestamp("2024-03-04") + pd.Timedelta(days=s)
        price = 200.0
        for b in range(bars):
            price *= np.exp(rng.normal(0, 0.004))
            ts = day + pd.Timedelta(hours=10 + b)
            rows.append({"datetime": ts, "session": day.normalize(),
                         "open": price, "high": price * 1.002,
                         "low": price * 0.998, "close": price, "volume": 1000.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# No-lookahead
# ---------------------------------------------------------------------------
def test_daily_features_do_not_use_future_prices():
    """Perturbing a FUTURE daily bar must not change any feature on an earlier day."""
    daily = _synthetic_daily()
    base = _price_and_vol_features(daily)

    perturb_i = 150
    d2 = daily.copy()
    d2.loc[perturb_i, ["open", "high", "low", "close"]] += 20.0
    d2.loc[perturb_i, "volume"] *= 4
    perturbed = _price_and_vol_features(d2)

    earlier = slice(0, perturb_i - 1)  # rows that cannot see day perturb_i
    for col in DIR_PRICE_FEATURES:
        a, b = base[col].to_numpy()[earlier], perturbed[col].to_numpy()[earlier]
        assert np.allclose(a, b, equal_nan=True), f"Feature {col} leaked future data!"


# ---------------------------------------------------------------------------
# Intraday realized-vol construction
# ---------------------------------------------------------------------------
def test_intraday_realized_variance_is_sum_of_squared_bar_returns():
    intr = _synthetic_intraday(sessions=2, bars=6)
    rv = realized_vol_from_intraday(intr).set_index("date")
    for day, g in intr.groupby("session"):
        expected_var = (np.log(g["close"]).diff() ** 2).sum()  # within-session
        got = rv.loc[pd.Timestamp(day), "iv_rvar"]
        assert np.isclose(got, expected_var, atol=1e-12)
        assert np.isclose(rv.loc[pd.Timestamp(day), "iv_rv"], np.sqrt(expected_var))


def test_intraday_rv_drops_thin_sessions():
    """A session with <3 bars is dropped as unreliable."""
    intr = _synthetic_intraday(sessions=2, bars=6)
    thin = intr[intr["session"] == intr["session"].min()].head(2)  # only 2 bars
    intr2 = pd.concat([thin, intr[intr["session"] != intr["session"].min()]])
    rv = realized_vol_from_intraday(intr2)
    assert pd.Timestamp(thin["session"].iloc[0]) not in set(rv["date"])


# ---------------------------------------------------------------------------
# Threshold scaling (shared with the parent project)
# ---------------------------------------------------------------------------
def test_direction_thresholds_scale_with_sqrt_h():
    up1, down1 = config.horizon_thresholds(1)
    for h in config.TSLA_DIR_HORIZONS:
        up, down = config.horizon_thresholds(h)
        assert np.isclose(up, up1 * h ** 0.5)
        assert np.isclose(down, down1 * h ** 0.5)


# ---------------------------------------------------------------------------
# Metric wiring
# ---------------------------------------------------------------------------
def test_score_matches_hand_computed_mcc():
    y = np.array([1, 0, 1, 0, 1, 0])
    perfect = score(y, y)
    assert np.isclose(perfect["accuracy"], 1.0) and np.isclose(perfect["mcc"], 1.0)
    inverted = score(y, 1 - y)
    assert np.isclose(inverted["mcc"], -1.0)


# ---------------------------------------------------------------------------
# Built-table checks (skipped without the cache)
# ---------------------------------------------------------------------------
_HAS_CACHE = config.TSLA_FEATURES_CACHE.exists()
cache_only = pytest.mark.skipif(not _HAS_CACHE, reason="TSLA feature cache absent "
                                "(run tesla/features.py).")


@cache_only
def test_forward_logrv_label_is_the_h_ahead_persistence():
    """At row t, the forward log-RV label must equal the persistence (current log-RV)
    value at row t+h — both measure realized vol over the same window (t+1..t+h)."""
    df = pd.read_parquet(config.TSLA_FEATURES_CACHE).sort_values("date").reset_index(drop=True)
    for h in config.TSLA_VOL_HORIZONS:
        for fwd, per in [(cc_fwd_col(h), cc_persist_col(h)),
                         (iv_fwd_col(h), iv_persist_col(h))]:
            aligned = df[per].shift(-h)
            mask = df[fwd].notna() & aligned.notna()
            assert mask.sum() > 50
            assert np.allclose(df[fwd][mask], aligned[mask], atol=1e-6), \
                f"{fwd} is not the h-ahead realized vol"


@cache_only
def test_intraday_features_only_in_hourly_window_daily_features_full_history():
    df = pd.read_parquet(config.TSLA_FEATURES_CACHE)
    # Daily features present for (almost) all rows; intraday only for a recent subset.
    assert df["log_rv_d"].notna().mean() > 0.99
    frac_intraday = df["log_iv_rv_d"].notna().mean()
    assert 0.10 < frac_intraday < 0.60, frac_intraday
    # Where intraday features exist, the +intraday vol set must be fully finite.
    sub = df.dropna(subset=VOL_INTRADAY_FEATURES)
    assert np.isfinite(sub[VOL_INTRADAY_FEATURES].to_numpy()).all()
