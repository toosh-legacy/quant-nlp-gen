"""Single source of truth for Volatility Radar.

Scope is deliberately narrow: forecast **realized volatility** for the market as a whole
and for each equity sector, at a fixed headline horizon. Everything the pipeline needs to
be reproducible — universes, tickers, horizons, split boundaries, cache paths — lives here
and nowhere else, so a reader can audit the no-lookahead discipline from one file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility & paths
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # downloaded price/macro series, never modified
BUILD_DIR = DATA_DIR / "build"       # derived feature panels
REPORT_DIR = DATA_DIR / "reports"    # backtest outputs consumed by the API/frontend

# ---------------------------------------------------------------------------
# Horizons
# ---------------------------------------------------------------------------
# Trading days ahead. All three are evaluated; only PRODUCT_HORIZONS are served.
HORIZONS = [5, 10, 21]
HEADLINE_HORIZON = 10

# Horizons the dashboard is allowed to show. 21 is evaluated and reported but deliberately
# NOT served: in walk-forward it beat the persistence baseline in only 12 of 15 years on the
# MARKET panel, with a -1.29 miss in an ultra-calm year, and the simpler HAR model was no
# more reliable. Every model tested extrapolates a volatility floor that very quiet years
# fall through.
#
# Worth recording precisely: the SECTOR panel at 21 days is stable (15/15 years, +0.269 +/-
# 0.135). It is withheld anyway, because the dashboard presents market and sector forecasts
# together at one horizon, and the market headline is the half that fails. Shipping a
# horizon where one of the two panels is unreliable would put a number on screen that the
# track record does not support.
PRODUCT_HORIZONS = [5, 10]
UNSTABLE_HORIZONS = [21]

# ---------------------------------------------------------------------------
# What we forecast
# ---------------------------------------------------------------------------
MARKET_TICKER = "SPY"

# The eleven GICS sectors via their SPDR ETFs. Two of them start late (XLRE spun out of
# financials in 2015, XLC out of tech/discretionary in 2018); they simply contribute a
# shorter history, which the panel handles by dropping rows with missing features rather
# than by dropping the sector.
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Cons. Discretionary",
    "XLP": "Cons. Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Comm. Services",
}

# ---------------------------------------------------------------------------
# Factor-family source series
# ---------------------------------------------------------------------------
# Family 1 — the implied-volatility complex. Implied volatility is the market's own
# forward-looking forecast, and it is the predictor the machine-learning volatility
# literature ranks second only to realized volatility itself.
#
# These come from Cboe's own free daily-history CSVs rather than Yahoo (see
# `data/fetch.py`): the full term structure is published there, with OHLC rather than close
# alone, and it is the maintained primary source rather than a scrape of one.
#
#   VIX / VIX9D / VIX3M / VIX6M  the term structure — 9-day, 30-day, 3-month, 6-month.
#                                Its *shape* is the stress signal: backwardation (short end
#                                above long) is the classic one, and the level alone cannot
#                                express it.
#   VVIX                         vol-of-vol: how uncertain the option market is about VIX.
#   SKEW                         option-implied tail asymmetry, orthogonal to the level.
#   VXN / RVX                    Nasdaq-100 and Russell-2000 implied vol; the spread to VIX
#                                says whether stress is concentrated or broad.
#   OVX                          crude-oil implied vol — the cross-asset shock channel.
#
# Yahoo tickers are kept as the naming convention so a fallback fetch is a drop-in.
VIX_TICKERS = ["^VIX", "^VIX9D", "^VIX3M", "^VIX6M", "^VVIX", "^SKEW",
               "^VXN", "^RVX", "^OVX"]

# Family 3 — macro / rates / credit. Slow-moving but regime-defining.
#   ^TNX  10-year Treasury yield        ^IRX  13-week T-bill yield  (slope = TNX - IRX)
#   HYG   high-yield credit ETF         LQD   investment-grade ETF  (ratio = credit stress)
#   UUP   dollar index ETF
MACRO_TICKERS = ["^TNX", "^IRX", "HYG", "LQD", "UUP"]

# ---------------------------------------------------------------------------
# Which factor families actually ship
# ---------------------------------------------------------------------------
# Decided by `volradar/backtest/select_features.py` on walk-forward folds strictly BEFORE
# TEST_START, so the choice never saw the reporting window.
#
# The result is a negative one, and it is the most interesting finding in the project: NONE
# of the ten candidate families reliably improves on the series' own price history. That now
# includes the own-history blocks (jump, semivar, range_ext — the standard HAR-J and HAR-RS
# specifications plus Rogers-Satchell/Yang-Zhang) as well as the external ones. Realized
# volatility turns out to be a remarkably complete summary of a market's own risk state, and
# the signals that look independent are largely reflections of it.
#
# **The rule that produced this changed, and the answer did not.** It was previously a
# one-standard-error rule, which no other part of this project used — `methods.py` and
# `settle.py` both require a paired fold-by-fold t-test at t > 2. Two bars in one repo means
# the answer depends on which file asks the question. There is now one bar everywhere
# (`metrics.paired_test`): paired against core-only on identical folds, t > 2, on skill AND
# QLIKE, at BOTH annual and semi-annual fold slicings.
#
# One finding is worth recording here because it is easy to lose: the VIX complex splits the
# two loss functions. It clears the bar on QLIKE (sector t = +2.72 annual, +2.62
# semi-annual) and fails on skill (−1.48, −1.33). Implied volatility improves the forecast
# where under-forecasting risk is punished and not where squared error is, which is a
# statement about *where* it helps rather than whether it works. It does not ship because
# the rule requires both, and the rule was fixed before the numbers were seen.
#
# The families remain implemented and are re-tested on every run. If a future window shows
# one clearly earning its place, adding it here is a one-line change.
SELECTED_FAMILIES: dict[str, list[str]] = {
    "market": [],
    "sector": [],
}

# ---------------------------------------------------------------------------
# Where each panel's model gets its TRAINING data
# ---------------------------------------------------------------------------
# "own"    — fit only on that panel's own history.
# "pooled" — fit on all 91 series (79 stocks + 11 sector ETFs + SPY) at once, then predict
#            the panel. Every feature is scale-free (log ratios, ranks, z-scores), and two
#            indicator columns let the model tell an index from a single name.
#
# Chosen by the paired bake-off in `volradar/backtest/methods.py` on pre-test folds:
#   sector : pooled wins, delta +0.011 skill, t = +2.15 over 12 folds (9/12 wins), and the
#            same sign and size at 6 folds (t = +1.93). ~471k training rows instead of 37k.
#   market : pooled LOSES (t = -1.11). Plausibly because the index is genuinely a different
#            object from its constituents — diversification makes its volatility dynamics
#            distinct — and 79 single stocks drown out the one series that matters.
POOL_TAGS = ["is_index", "is_sector"]
TRAINING_SOURCE: dict[str, str] = {
    "market": "own",
    "sector": "pooled",
}

# Family 2 — cross-sectional dispersion is computed from this stock universe (the same
# ~80 liquid large-caps used by the earlier study, whose 2005-2024 OHLCV cache we reuse).
BREADTH_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "INTC", "CSCO", "ORCL", "IBM",
    "ADBE", "CRM", "QCOM", "TXN", "AMD", "JPM", "BAC", "WFC", "C", "GS",
    "MS", "AXP", "BLK", "SCHW", "USB", "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "LLY", "BMY", "AMGN", "GILD", "CVS", "XOM", "CVX", "COP", "SLB", "EOG",
    "PSX", "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "HD",
    "LOW", "TGT", "DIS", "CMCSA", "VZ", "T", "NFLX", "BA", "CAT", "GE",
    "HON", "UPS", "UNP", "MMM", "LMT", "DE", "F", "GM", "NEE", "DUK",
    "SO", "D", "AEP", "LIN", "APD", "SHW", "FCX", "NEM", "DOW",
]

# ---------------------------------------------------------------------------
# Date range & chronological splits
# ---------------------------------------------------------------------------
HISTORY_START = "2005-01-01"

# Through the most recent close. Dynamic so `fetch --force` brings the dashboard up to
# date; every split boundary below is an absolute date, so extending the end of the sample
# can never move a train/test boundary and cannot retroactively change a reported metric.
HISTORY_END = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# The fixed out-of-sample split. Train on everything before TEST_START, report on what
# follows. Chosen before any modelling so it cannot be tuned against.
TEST_START = "2018-01-01"

# Hyperparameters are chosen on this held-out slice of the *training* period, never on
# the test period.
VAL_START = "2015-01-01"
VAL_END = "2018-01-01"

# Walk-forward: retrain at the start of each year on all prior data, score that year.
# Ends at the current year, which the fold-size guard drops until it has enough sessions.
WALK_YEARS = list(range(2012, pd.Timestamp.today().year + 1))

# ---------------------------------------------------------------------------
# Regime thresholds
# ---------------------------------------------------------------------------
# Calm / normal / stressed are defined by percentiles of the *training-period* realized
# vol distribution, computed once and frozen so the labels never drift with the test data.
REGIME_QUANTILES = (0.33, 0.80)
REGIME_NAMES = ["calm", "normal", "stressed"]

# ---------------------------------------------------------------------------
# Numerical
# ---------------------------------------------------------------------------
EPS = 1e-8
TRADING_DAYS = 252


def fwd_col(h: int) -> str:
    """Regression target: log realized volatility over the NEXT h trading days."""
    return f"y_logrv_{h}"


def persist_col(h: int) -> str:
    """Persistence baseline: log realized volatility over the LAST h trading days."""
    return f"persist_logrv_{h}"


def label_date_col(h: int) -> str:
    """Calendar date of the h-days-ahead observation, used to purge (embargo) rows whose
    outcome window straddles a train/test boundary."""
    return f"label_date_{h}"


def ensure_dirs() -> None:
    for d in (RAW_DIR, BUILD_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
