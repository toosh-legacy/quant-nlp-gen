"""The most important tests in the project: they enforce that no future information ever
leaks into a feature row. A model that accidentally sees the future looks great in a
backtest and fails in reality — these tests are the guardrail against that class of bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from features.build_features import (  # noqa: E402
    _roll_sentiment_onto_trading_days,
    assign_split,
    compute_price_features,
    load_ticker_sectors,
)


def _synthetic_prices(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A deterministic fake price series with all columns compute_price_features needs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n)  # business days
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000, 10_000, size=n).astype(float),
        }
    )


def test_price_features_do_not_use_future_prices():
    """Perturbing a FUTURE price must not change any feature computed for an earlier day.
    This is the core no-lookahead property. The label (movement) is allowed to change,
    because the label is deliberately the next day's direction."""
    prices = _synthetic_prices()
    base = compute_price_features(prices)

    perturb_i = 40  # change a price late in the series
    prices2 = prices.copy()
    prices2.loc[perturb_i, ["adj_close", "close"]] += 25.0
    prices2.loc[perturb_i, "volume"] *= 3
    perturbed = compute_price_features(prices2)

    # Every feature for days strictly before the perturbation must be identical.
    earlier = slice(0, perturb_i - 1)  # rows up to perturb_i-2 cannot see day perturb_i
    for col in config.PRICE_FEATURE_COLS:
        a = base[col].to_numpy()[earlier]
        b = perturbed[col].to_numpy()[earlier]
        assert np.allclose(a, b, equal_nan=True), f"Feature {col} leaked future data!"

    # Sanity: the label for the day BEFORE the perturbation *should* change, proving the
    # label genuinely looks one step forward (and that our test can detect a change).
    assert not np.allclose(
        base["next_return_1"].to_numpy()[perturb_i - 1],
        perturbed["next_return_1"].to_numpy()[perturb_i - 1],
        equal_nan=True,
    )


def test_labels_use_h_days_ahead_with_scaled_thresholds():
    """For every horizon, movement_h==1 iff the h-day-ahead return clears the (sqrt(h)-scaled)
    up threshold; ==0 iff below the down threshold; NaN in between. This confirms the label
    genuinely looks h steps forward and uses the fair, horizon-scaled band."""
    prices = _synthetic_prices()
    feats = compute_price_features(prices)
    px = prices["adj_close"].to_numpy()
    for h in config.HORIZONS:
        up_thr, down_thr = config.horizon_thresholds(h)
        col = feats[config.movement_col(h)].to_numpy()
        for i in range(len(prices) - h):
            fwd = px[i + h] / px[i] - 1.0
            if fwd >= up_thr:
                assert col[i] == 1
            elif fwd <= down_thr:
                assert col[i] == 0
            else:
                assert np.isnan(col[i])


def test_rsi_and_macd_are_well_formed():
    """RSI stays in [0, 100]; the MACD histogram equals macd - signal exactly."""
    prices = _synthetic_prices(n=120)
    feats = compute_price_features(prices)
    rsi = feats["rsi_14"].dropna().to_numpy()
    assert (rsi >= 0).all() and (rsi <= 100).all()
    assert np.allclose(
        feats["macd_hist"].to_numpy(),
        (feats["macd"] - feats["macd_signal"]).to_numpy(),
        equal_nan=True,
    )


def test_rsi_macd_vol_do_not_leak_future():
    """RSI/MACD (EMA-based) and the multi-scale volatility features are all causal — perturbing
    a future close leaves earlier values untouched."""
    prices = _synthetic_prices(n=120)
    base = compute_price_features(prices)
    perturb_i = 90
    p2 = prices.copy()
    p2.loc[perturb_i, ["adj_close", "close"]] += 30.0
    perturbed = compute_price_features(p2)
    earlier = slice(0, perturb_i)  # indices 0..perturb_i-1 must be unchanged
    for col in ("rsi_14", "macd", "macd_signal", "macd_hist",
                "volatility_5d", "volatility_10d", "volatility_20d"):
        assert np.allclose(
            base[col].to_numpy()[earlier], perturbed[col].to_numpy()[earlier], equal_nan=True
        ), f"{col} leaked future data!"


def test_forward_volatility_covers_the_next_h_days():
    """The Task B label source fwd_vol_h(t) must equal the std of daily returns over days
    t+1..t+h (a forward window) — that's what makes it a *label*, not a feature."""
    prices = _synthetic_prices(n=80)
    feats = compute_price_features(prices)
    ret = prices["adj_close"].pct_change(1)
    for h in config.VOL_HORIZONS:
        col = feats[config.fwd_vol_col(h)].to_numpy()
        for i in range(len(prices) - h - 1):
            if np.isnan(col[i]):
                continue
            expected = ret.iloc[i + 1 : i + 1 + h].std()  # ddof=1, matches rolling().std()
            assert np.isclose(col[i], expected), f"fwd_vol_{h} at {i} != future-window std"


def test_sentiment_rolls_forward_never_backward():
    """A tweet must attach to the first trading day >= its date — never to an earlier day.
    A Saturday tweet belongs to Monday's features, and Monday's tweet must never appear in
    Friday's row."""
    trading = pd.DataFrame({"date": pd.to_datetime(["2014-01-03", "2014-01-06", "2014-01-07"])})
    # Friday 1/3, Monday 1/6, Tuesday 1/7. Put a tweet on Saturday 1/4 and Monday 1/6.
    sent = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2014-01-04", "2014-01-06"]),
            "sent_mean": [0.9, -0.9],
            "sent_tweet_count": [1, 1],
            "sent_bull_ratio": [1.0, 0.0],
        }
    )
    rolled = _roll_sentiment_onto_trading_days(trading, sent)
    by_date = rolled.set_index("date")["sent_mean"].to_dict()

    # Saturday 1/4 tweet -> Monday 1/6 (first trading day >= 1/4); Monday tweet also -> 1/6.
    assert pd.Timestamp("2014-01-06") in by_date
    # Friday 1/3 must have received nothing (no tweet dated <= 1/3 that rolls onto it).
    assert pd.Timestamp("2014-01-03") not in by_date


def test_assign_split_boundaries_are_chronological_and_half_open():
    dates = pd.to_datetime(
        [config.TRAIN_START, config.DEV_START, config.TEST_START,
         "2015-12-31", config.TEST_END]
    )
    splits = assign_split(pd.Series(dates)).tolist()
    assert splits[0] == "train"           # TRAIN_START is in train
    assert splits[1] == "dev"             # DEV_START begins dev (half-open)
    assert splits[2] == "test"            # TEST_START begins test
    assert splits[3] == "test"            # last day inside test
    assert splits[4] is np.nan or splits[4] != splits[4]  # TEST_END excluded (NaN)


@pytest.mark.skipif(
    not (config.STOCKNET_DIR / "StockTable").exists(),
    reason="StockNet dataset not cloned (StockTable absent).",
)
def test_sector_mapping_covers_tickers_across_multiple_sectors():
    """The per-sector analysis needs a ticker->sector map. Confirm it parses, strips the
    leading '$', and spans several sectors (StockNet spans 9)."""
    mapping = load_ticker_sectors()
    assert mapping.get("AAPL") == "Consumer Goods"
    assert "$" not in "".join(mapping.keys())
    assert len({v for v in mapping.values()}) >= 5
