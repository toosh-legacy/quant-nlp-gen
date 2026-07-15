"""Central configuration — the single source of truth for paths, seeds, model/dataset
names, feature columns, and the chronological date split.

Everything else in the project imports from here so there is exactly one place to change
a path or a hyperparameter. Keeping this centralized is also what makes the "no lookahead
bias" discipline auditable: the train/dev/test date boundaries live here and nowhere else.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# A single seed used everywhere we sample, shuffle, or initialize (numpy, torch,
# sklearn, xgboost). Fixing it means a rerun reproduces the same numbers.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root = the directory this file lives in.
ROOT = Path(__file__).resolve().parent

# All downloaded data and generated artifacts live under data/ which is gitignored.
# We never commit raw datasets or trained model weights.
DATA_DIR = ROOT / "data"

# Where the cloned StockNet dataset repo lands.
STOCKNET_DIR = DATA_DIR / "stocknet-dataset"

# Where the fine-tuned sentiment model is saved after training.
SENTIMENT_MODEL_DIR = DATA_DIR / "sentiment_model"

# Cached intermediate feature tables (parquet) so the expensive steps run once.
SENTIMENT_FEATURES_PATH = DATA_DIR / "stocknet_daily_sentiment.parquet"
PRICE_FEATURES_PATH = DATA_DIR / "stocknet_price_features.parquet"
COMBINED_FEATURES_PATH = DATA_DIR / "stocknet_combined_features.parquet"

# Evaluation output.
RESULTS_PATH = DATA_DIR / "results_comparison.csv"

# ---------------------------------------------------------------------------
# Model / dataset identifiers
# ---------------------------------------------------------------------------
# Hugging Face dataset that teaches the sentiment model bullish/bearish/neutral language.
SENTIMENT_DATASET = "zeroshot/twitter-financial-news-sentiment"

# Base transformer we fine-tune. 66M params — genuinely CPU-fine-tunable.
BASE_SENTIMENT_MODEL = "distilbert-base-uncased"

# StockNet source repo (Xu & Cohen, ACL 2018).
STOCKNET_REPO_URL = "https://github.com/yumoxu/stocknet-dataset"

# ---------------------------------------------------------------------------
# Sentiment label mapping
# ---------------------------------------------------------------------------
# The twitter-financial-news-sentiment dataset uses integer labels. Per its dataset
# card the mapping is: 0 = Bearish, 1 = Bullish, 2 = Neutral. Confirmed at load time
# in step 1 rather than assumed blindly.
LABEL_NAMES = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAMES.items()}
NUM_LABELS = 3

# ---------------------------------------------------------------------------
# Sentiment training hyperparameters (reasonable CPU defaults, not hard requirements)
# ---------------------------------------------------------------------------
SENT_MAX_LENGTH = 128        # tweets are short
SENT_BATCH_SIZE = 16
SENT_EPOCHS = 3
SENT_LEARNING_RATE = 2e-5

# ---------------------------------------------------------------------------
# StockNet chronological date split (from the original paper / repo).
# ---------------------------------------------------------------------------
# The StockNet paper uses a fixed chronological split over 2014-01-01 .. 2016-01-01:
#   train: 2014-01-01 .. 2015-08-01
#   dev:   2015-08-01 .. 2015-10-01
#   test:  2015-10-01 .. 2016-01-01
# Boundaries are half-open [start, end). Using the paper's own split keeps results
# comparable to the published baseline and prevents lookahead bias. These are
# re-confirmed against the cloned repo's README in step 1.
TRAIN_START = "2014-01-01"
DEV_START = "2015-08-01"
TEST_START = "2015-10-01"
TEST_END = "2016-01-01"

# ---------------------------------------------------------------------------
# Prediction horizons — "how far ahead can we predict?"
# ---------------------------------------------------------------------------
# We predict the direction of the return from close[t] to close[t+h] for several
# horizons h (in trading days) and compare how predictability changes with horizon.
# 1 = next day, 5 = next week, 10 = two weeks, 20 = one month, 60 = one quarter.
HORIZONS = [1, 5, 10, 20, 60]
DEFAULT_HORIZON = 5  # the horizon the demo/tests default to


def movement_col(h: int) -> str:
    """Name of the up/down label column for horizon h (e.g. 'movement_5')."""
    return f"movement_{h}"


def label_date_col(h: int) -> str:
    """Name of the column holding the calendar date of the h-days-ahead close, used to
    embargo rows whose outcome window crosses a train/dev/test boundary."""
    return f"label_date_{h}"


def label_split_col(h: int) -> str:
    """Name of the column holding the split (train/dev/test) of the h-days-ahead close."""
    return f"label_split_{h}"


# Backward-compatible default target (the daily label), used by the demo and some tests.
TARGET_COL = movement_col(1)

# Two-class movement thresholds. The daily band is StockNet's (Xu & Cohen 2018):
# up if next-day return >= +0.55%, down if <= -0.50%, drop the ambiguous middle. For a
# longer horizon h, returns scale roughly with sqrt(h) (random-walk volatility), so we
# scale the band by sqrt(h). This keeps the fraction of dropped "ambiguous" rows about
# constant across horizons, which makes the horizon comparison FAIR: a higher accuracy at
# 10 days then reflects real signal, not just a looser bar on bigger moves.
_DAILY_UP = 0.0055
_DAILY_DOWN = -0.005


def horizon_thresholds(h: int) -> tuple[float, float]:
    """(up_threshold, down_threshold) for horizon h, scaled by sqrt(h) from the daily band."""
    scale = float(h) ** 0.5
    return _DAILY_UP * scale, _DAILY_DOWN * scale


# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------
# Price-based technical features (computed in features/build_features.py). Each is
# strictly a function of information available up to and including the row's own day.
PRICE_FEATURE_COLS = [
    "return_1d",       # yesterday->today return (momentum, 1 day)
    "return_5d",       # 5-day momentum
    "ma_5_ratio",      # close / 5-day moving average
    "ma_10_ratio",     # close / 10-day moving average
    "volatility_5d",   # rolling std of daily returns (5d)
    "volume_change",   # today's volume vs prior day
    "rsi_14",          # Wilder's 14-day Relative Strength Index (0-100)
    "macd",            # MACD line: EMA(12) - EMA(26) of close
    "macd_signal",     # 9-day EMA of the MACD line
    "macd_hist",       # MACD histogram: macd - macd_signal
]

# Sentiment-based features (computed in step 3 from the fine-tuned model's scores).
SENTIMENT_FEATURE_COLS = [
    "sent_mean",        # mean signed sentiment score across the day's tweets
    "sent_tweet_count", # how many tweets that ticker-day had (log-scaled at use)
    "sent_bull_ratio",  # bullish / (bullish + bearish) among the day's tweets
    "sent_mean_3d",     # trailing 3-trading-day mean of sent_mean (smooths single-day noise)
]

# Number of trading days in the trailing sentiment window (causal — includes day t).
SENT_TRAILING_WINDOW = 3

# ---------------------------------------------------------------------------
# Task B — volatility-regime prediction ("will next week be calm or turbulent?")
# ---------------------------------------------------------------------------
# Unlike direction (near-random), realized volatility is strongly autocorrelated
# (volatility clustering / the ARCH effect), so a high-vs-low-volatility classifier is
# genuinely predictable — honestly ~0.70-0.80 accuracy. We predict whether the realized
# volatility over the NEXT h trading days is above or below the training-set median.
VOL_HORIZONS = [5, 10, 20]
DEFAULT_VOL_HORIZON = 5

# Extra backward-looking volatility features (added to the price features for this task).
# These carry the past-volatility signal that makes the future regime predictable.
VOL_EXTRA_FEATURE_COLS = ["volatility_10d", "volatility_20d"]

# HAR-RV features (Corsi 2009, "Heterogeneous AutoRegressive" realized volatility). The
# gold-standard volatility predictors: realized volatility measured over a DAILY, WEEKLY,
# and MONTHLY look-back. Traders at different frequencies react to vol on different scales,
# so these three components together forecast future vol remarkably well — often beating
# fancier models. All are backward-looking (past squared returns).
HAR_FEATURE_COLS = ["har_rv_d", "har_rv_w", "har_rv_m"]

# VIX / implied-volatility features (downloaded from Yahoo Finance via yfinance). Implied
# volatility is the market's own FORWARD-LOOKING volatility forecast, and one of the
# strongest predictors of future realized volatility — so this is the "better data" lever
# for Task B. All values are known at the close of day t (no lookahead).
VIX_SYMBOLS = ["^VIX", "^VIX3M"]           # spot VIX + 3-month VIX (for the term structure)
VIX_FEATURE_COLS = [
    "vix_level",     # VIX close (overall market fear level)
    "vix_ret_5d",    # 5-day change in VIX (is fear rising or falling?)
    "vix_rel_20d",   # VIX vs its own 20-day average (elevated vs recent norm?)
    "vix_term",      # VIX3M / VIX term structure: >1 calm (contango), <1 stressed (backwardation)
]
VIX_CACHE_PATH = DATA_DIR / "vix_features.parquet"

# Feature sets for Task B. NOTE: VIX features are intentionally NOT included here — an
# ablation showed VIX (a single market-wide series, identical across all tickers on a day)
# added no cross-sectional signal to per-ticker vol prediction and was redundant with each
# stock's own HAR volatility. See README ("Did VIX help?") and predictor/volatility.py. The
# yfinance VIX loader is kept in features/build_features.py for that documented experiment.
VOL_PRICE_FEATURE_COLS = PRICE_FEATURE_COLS + VOL_EXTRA_FEATURE_COLS + HAR_FEATURE_COLS
VOL_COMBINED_FEATURE_COLS = VOL_PRICE_FEATURE_COLS + SENTIMENT_FEATURE_COLS


def fwd_vol_col(h: int) -> str:
    """Column holding the realized volatility over the NEXT h trading days (the forward
    side of the Task B label)."""
    return f"fwd_vol_{h}"


def current_vol_col(h: int) -> str:
    """Backward-looking realized volatility over the last h days — the reference level the
    Task B label compares against. Label = 1 iff next-h-day vol > current h-day vol
    ("will volatility rise?"). Comparing to the current level makes the two classes roughly
    balanced in every period, so accuracy is a fair, interpretable metric (unlike a fixed
    threshold, which is imbalanced when the test period is unusually calm or turbulent)."""
    return f"volatility_{h}d"


# ===========================================================================
# Tesla (TSLA) single-stock intraday-enriched deep dive  (the `tesla/` package)
# ===========================================================================
# A focused companion to the multi-stock study. Same honest methodology
# (chronological splits, overlapping-window embargo, baseline-vs-MCC/R^2), but
# on ONE deliberately-volatile, retail/news-driven name, enriched with genuine
# intraday data the multi-stock project couldn't afford. The single question:
# does deeper, single-stock, intraday data buy a stronger or more RELIABLE edge?
TSLA_TICKER = "TSLA"

# yfinance replaces Alpha Vantage as the price source: no API key, already a
# dependency. AV is kept only as an OPTIONAL, env-gated path (off by default) so
# the NEWS_SENTIMENT ambition is documented, not required. Read at the use site:
#   os.environ.get(ALPHAVANTAGE_ENV)  -> None means "use the free yfinance path".
ALPHAVANTAGE_ENV = "ALPHAVANTAGE_API_KEY"

# Daily history goes back far enough for a real walk-forward across regimes
# (incl. the 2020 COVID crash and the 2021-22 meme/rate cycle). Intraday history
# is capped by Yahoo's free tier — see the per-interval windows below.
TSLA_DAILY_START = "2015-01-01"

# Yahoo intraday retention (free tier), used by tesla/fetch_prices.py:
#   1h  -> ~730 days  (the HEADLINE intraday layer: ~7 bars/session realized vol)
#   5m  -> ~60 days   } recent high-res slices, used only to VALIDATE that the
#   1m  -> ~7 days    } hourly realized-vol tracks finer-resolution RV.
TSLA_INTRADAY_WINDOWS = {"1h": "730d", "5m": "60d", "1m": "7d"}
TSLA_HEADLINE_INTRADAY = "1h"   # the interval whose RV feeds the daily features

# Caches (gitignored under data/). Downloads + feature build run once.
TSLA_DAILY_CACHE = DATA_DIR / "tsla_daily.parquet"
TSLA_INTRADAY_CACHE = {iv: DATA_DIR / f"tsla_intraday_{iv}.parquet"
                       for iv in TSLA_INTRADAY_WINDOWS}
TSLA_FEATURES_CACHE = DATA_DIR / "tsla_features.parquet"

# Public Tesla news/tweets dataset (the validation baseline + sentiment source).
# Loader reads any CSV(s) dropped here and auto-detects date/text/sentiment cols.
KAGGLE_TSLA_DIR = DATA_DIR / "kaggle_tsla"
TSLA_DAILY_SENTIMENT_CACHE = DATA_DIR / "tsla_daily_sentiment.parquet"

# Results.
TSLA_DIRECTION_RESULTS = DATA_DIR / "tsla_direction_results.csv"
TSLA_VOLATILITY_RESULTS = DATA_DIR / "tsla_volatility_results.csv"
TSLA_COMPARISON_RESULTS = DATA_DIR / "tsla_comparison.csv"

# Horizons (trading days). Direction reuses the parent project's set; volatility
# reuses the universe run's (one week / two weeks / one month) for a clean
# apples-to-apples R^2 comparison against R^2 ~ 0.42.
TSLA_DIR_HORIZONS = [1, 5, 10, 20]
TSLA_VOL_HORIZONS = [5, 10, 21]
