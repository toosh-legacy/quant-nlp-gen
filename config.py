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
# 1 = next day, 5 = next week, 10 = next two weeks.
HORIZONS = [1, 5, 10]
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
]
