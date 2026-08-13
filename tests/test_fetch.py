"""The download layer's two silent failure modes.

Neither of these crashes. A truncated series produces a panel with a mostly-empty feature
column and a confident forecast built on it; a mis-parsed CSV produces plausible numbers
for the wrong dates. Both were previously untested, and the coverage guard in particular is
cited in the README as having already caught a real incident (a Yahoo request that returned
13 rows for ^TNX instead of ~5,400).
"""

from __future__ import annotations

import pandas as pd
import pytest

from volradar.data import fetch


def _long(rows: dict[str, int]) -> pd.DataFrame:
    """A long price frame with a given number of rows per ticker."""
    parts = []
    for t, n in rows.items():
        parts.append(pd.DataFrame({
            "date": pd.bdate_range("2015-01-01", periods=n),
            "ticker": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        }))
    return pd.concat(parts, ignore_index=True)


# --- the coverage guard ---------------------------------------------------------------
def test_coverage_guard_flags_a_truncated_series():
    df = _long({"AAA": 1000, "BBB": 1000, "CCC": 13})
    assert fetch._check_coverage(df, ["AAA", "BBB", "CCC"]) == ["CCC"]


def test_coverage_guard_flags_a_series_that_is_missing_entirely():
    df = _long({"AAA": 1000})
    assert "BBB" in fetch._check_coverage(df, ["AAA", "BBB"])


def test_coverage_guard_tolerates_a_legitimately_shorter_history():
    """A series that genuinely starts later must not be flagged.

    This is not hypothetical: VIX9D begins in 2011 and XLC in 2018, so a guard that fired on
    any series shorter than the longest would cry wolf on every run and be ignored.
    """
    df = _long({"AAA": 1000, "BBB": 800})
    assert fetch._check_coverage(df, ["AAA", "BBB"]) == []


def test_coverage_ratio_is_the_documented_one():
    n = 1000
    just_under = int(n * fetch.MIN_COVERAGE_RATIO) - 1
    df = _long({"AAA": n, "BBB": just_under})
    assert fetch._check_coverage(df, ["AAA", "BBB"]) == ["BBB"]


# --- the Cboe parser ------------------------------------------------------------------
OHLC_CSV = """DATE,OPEN,HIGH,LOW,CLOSE
01/02/2020,13.780000,14.560000,12.420000,12.470000
01/03/2020,15.010000,16.200000,13.880000,14.020000
"""

CLOSE_ONLY_CSV = """DATE,VVIX
01/02/2020,92.190000
01/03/2020,99.780000
"""


def test_cboe_ohlc_layout_parses_to_the_pipeline_schema():
    out = fetch._parse_cboe_csv(OHLC_CSV, "^VIX")
    assert list(out.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert out["ticker"].unique().tolist() == ["^VIX"]
    assert out["close"].iloc[0] == pytest.approx(12.47)
    assert out["high"].iloc[1] == pytest.approx(16.20)


def test_cboe_dates_are_month_first_not_day_first():
    """01/02/2020 is 2 January in Cboe's files. Read day-first it becomes 1 February.

    A whole series shifted by a month would still look entirely plausible in every summary
    statistic, which is exactly why this is worth asserting rather than eyeballing.
    """
    out = fetch._parse_cboe_csv(OHLC_CSV, "^VIX")
    assert out["date"].iloc[0] == pd.Timestamp("2020-01-02")
    assert out["date"].iloc[1] == pd.Timestamp("2020-01-03")


def test_cboe_close_only_layout_is_widened_to_ohlc():
    """VVIX and SKEW publish one column. Downstream code must not have to know that."""
    out = fetch._parse_cboe_csv(CLOSE_ONLY_CSV, "^VVIX")
    assert len(out) == 2
    for field in ("open", "high", "low", "close"):
        assert out[field].iloc[0] == pytest.approx(92.19)


def test_cboe_parser_drops_rows_without_a_usable_price():
    csv = OHLC_CSV + "01/06/2020,,,,\n"
    out = fetch._parse_cboe_csv(csv, "^VIX")
    assert len(out) == 2, "a blank price row must be dropped, not carried as NaN"
