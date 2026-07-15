"""Steps 3-5 — Build the per-ticker-day feature table.

This one module owns everything that turns the raw StockNet dataset into the tidy
`(ticker, date, features..., movement)` table the predictor trains on:

  * loading StockNet prices and tweets,
  * scoring every tweet through the fine-tuned sentiment model and aggregating to a daily
    sentiment factor  (step 3),
  * computing standard technical price factors               (step 4),
  * joining the two and attaching the up/down label + the chronological split  (step 5).

THE ONE RULE THAT MATTERS HERE: no lookahead bias.
Every feature attached to trading day t is computed from information available by the
close of day t and never from any later day. The *label* is the only thing that looks
forward — it is the direction of the return from close[t] to the next trading day's
close[t+1], which is exactly the future we are trying to predict. Rolling windows are
therefore taken over data up to and including t, and tweets are rolled onto the next
trading session that could actually act on them. This discipline is what makes a
backtest-style result honest instead of accidentally cheating; it is enforced by
tests/test_build_features.py.

Run:
    python features/build_features.py                 # score tweets + build everything
    python features/build_features.py --tickers AAPL GOOG   # subset, faster
    python features/build_features.py --skip-sentiment      # price features only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# StockNet's two-class movement thresholds (from Xu & Cohen 2018). Days whose next-day
# return falls in the ambiguous middle band are dropped so the up/down task is well-posed
# and the numbers stay comparable to the published baseline.
UP_THRESHOLD = 0.0055     # next-day return >= +0.55%  -> up   (label 1)
DOWN_THRESHOLD = -0.005   # next-day return <= -0.50%  -> down (label 0)


# ---------------------------------------------------------------------------
# StockNet IO
# ---------------------------------------------------------------------------
def list_tickers() -> list[str]:
    """All 88 tickers, derived from the price CSV filenames."""
    price_dir = config.STOCKNET_DIR / "price" / "raw"
    return sorted(p.stem for p in price_dir.glob("*.csv"))


def load_prices(ticker: str) -> pd.DataFrame:
    """Load one ticker's raw daily OHLCV, ascending by date, restricted to the project's
    2014-2016 window plus a small lead-in so early rolling features aren't all-NaN."""
    path = config.STOCKNET_DIR / "price" / "raw" / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns=str.lower)
    df = df.rename(columns={"adj close": "adj_close"})
    df = df.sort_values("date").reset_index(drop=True)
    # Keep a ~40-trading-day lead-in before TRAIN_START so a 10-day MA is defined on day one.
    lead_in = pd.Timestamp(config.TRAIN_START) - pd.Timedelta(days=90)
    df = df[(df["date"] >= lead_in) & (df["date"] < pd.Timestamp(config.TEST_END))]
    return df.reset_index(drop=True)


def load_tweets_for_date(ticker: str, date_str: str) -> list[str]:
    """Return the list of tweet texts for a ticker on a calendar date (or [] if none).
    The preprocessed tweets are stored pre-tokenized as a list of tokens per line; we join
    them back into a space-separated string for the sentiment model."""
    path = config.STOCKNET_DIR / "tweet" / "preprocessed" / ticker / date_str
    if not path.exists():
        return []
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        tokens = obj.get("text", [])
        texts.append(" ".join(tokens) if isinstance(tokens, list) else str(tokens))
    return texts


# ---------------------------------------------------------------------------
# Step 4 — price (technical) features
# ---------------------------------------------------------------------------
def compute_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Standard technical factors + the next-day movement label, for one ticker.

    All feature columns use only data up to and including each row's own day. The label
    uses the *next* day's close (the thing we predict) and is NOT a feature.
    """
    df = prices.copy()
    px = df["adj_close"]

    # --- Features (backward-looking only) ---
    # Daily return: today's close vs yesterday's. Known at close of today.
    df["return_1d"] = px.pct_change(1)
    # 5-day momentum.
    df["return_5d"] = px.pct_change(5)
    # Close relative to its own trailing moving averages (>1 = above trend).
    df["ma_5_ratio"] = px / px.rolling(5).mean()
    df["ma_10_ratio"] = px / px.rolling(10).mean()
    # Rolling volatility of daily returns (risk regime).
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    # Volume change vs prior day.
    df["volume_change"] = df["volume"].pct_change(1)

    # --- Label: direction of the NEXT trading day's return (the future we forecast) ---
    # shift(-1) looks one row FORWARD on purpose — this is the target, not a feature.
    next_return = px.shift(-1) / px - 1.0
    df["next_return"] = next_return
    df[config.TARGET_COL] = np.where(
        next_return >= UP_THRESHOLD, 1,
        np.where(next_return <= DOWN_THRESHOLD, 0, np.nan),
    )

    return df


# ---------------------------------------------------------------------------
# Step 3 — sentiment scoring
# ---------------------------------------------------------------------------
def _load_sentiment_pipeline():
    """Load the fine-tuned sentiment model for batched CPU inference."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = config.SENTIMENT_MODEL_DIR
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned sentiment model not found at {model_dir}. "
            "Run sentiment_model/train_sentiment_model.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return tokenizer, model


def score_texts(tokenizer, model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return an (N, 3) array of softmax probabilities [P(bear), P(bull), P(neutral)]."""
    import torch

    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, truncation=True, max_length=config.SENT_MAX_LENGTH,
                padding=True, return_tensors="pt",
            )
            logits = model(**enc).logits
            probs_all.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(probs_all) if probs_all else np.empty((0, config.NUM_LABELS))


def build_sentiment_features(tickers: list[str]) -> pd.DataFrame:
    """Score every StockNet tweet in the date window and aggregate to one row per
    (ticker, calendar_date). Cached to parquet because CPU inference over all tweets is
    the slowest stage — we only ever want to pay for it once."""
    tokenizer, model = _load_sentiment_pipeline()

    bear_idx = config.NAME_TO_LABEL["Bearish"]
    bull_idx = config.NAME_TO_LABEL["Bullish"]

    # Tweet folders exist per calendar date; scan the project window generously.
    start = pd.Timestamp(config.TRAIN_START) - pd.Timedelta(days=5)
    end = pd.Timestamp(config.TEST_END)

    rows = []
    for ticker in tickers:
        tdir = config.STOCKNET_DIR / "tweet" / "preprocessed" / ticker
        if not tdir.exists():
            continue
        for day_path in sorted(tdir.iterdir()):
            date_str = day_path.name
            try:
                date = pd.Timestamp(date_str)
            except ValueError:
                continue
            if not (start <= date < end):
                continue
            texts = load_tweets_for_date(ticker, date_str)
            if not texts:
                continue
            probs = score_texts(tokenizer, model, texts)
            preds = probs.argmax(axis=1)
            # Signed daily sentiment per tweet = P(bullish) - P(bearish), then averaged.
            signed = probs[:, bull_idx] - probs[:, bear_idx]
            n_bull = int((preds == bull_idx).sum())
            n_bear = int((preds == bear_idx).sum())
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "sent_mean": float(signed.mean()),
                    "sent_tweet_count": int(len(texts)),
                    "sent_bull_ratio": (n_bull / (n_bull + n_bear)) if (n_bull + n_bear) else 0.5,
                }
            )
        print(f"  scored tweets for {ticker}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5 — combine, align tweets to trading days, split
# ---------------------------------------------------------------------------
def _roll_sentiment_onto_trading_days(
    price_df: pd.DataFrame, sent_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach each ticker's calendar-date sentiment to the next trading session that could
    act on it. A tweet posted Saturday is only actionable at Monday's close, so we map each
    tweet date forward to the first trading day >= that date. This keeps weekend chatter
    without ever letting a trading day see tweets from its own future.
    """
    trading_days = price_df["date"].sort_values().to_numpy()
    if len(sent_df) == 0:
        empty = pd.DataFrame(columns=["date", *config.SENTIMENT_FEATURE_COLS])
        empty["date"] = pd.to_datetime(empty["date"])
        return empty

    s = sent_df.sort_values("date").copy()
    # searchsorted with side='left' finds the first trading day >= the tweet date.
    idx = np.searchsorted(trading_days, s["date"].to_numpy(), side="left")
    valid = idx < len(trading_days)
    s = s[valid].copy()
    s["trading_day"] = trading_days[idx[valid]]

    # A trading day may collect several calendar dates (e.g. Sat+Sun+Mon -> Monday).
    # Aggregate them with a tweet-count-weighted mean, done via explicit weighted sums so
    # the result doesn't depend on pandas' version-specific groupby.apply behavior.
    s["_wm"] = s["sent_mean"] * s["sent_tweet_count"]
    s["_wb"] = s["sent_bull_ratio"] * s["sent_tweet_count"]
    grp = s.groupby("trading_day", as_index=False).agg(
        _wm_sum=("_wm", "sum"),
        _wb_sum=("_wb", "sum"),
        sent_tweet_count=("sent_tweet_count", "sum"),
    )
    tot = grp["sent_tweet_count"].replace(0, np.nan)
    grp["sent_mean"] = (grp["_wm_sum"] / tot).fillna(0.0)
    grp["sent_bull_ratio"] = (grp["_wb_sum"] / tot).fillna(0.5)
    rolled = grp.rename(columns={"trading_day": "date"})[
        ["date", *config.SENTIMENT_FEATURE_COLS]
    ]
    # Guarantee the merge key dtype matches the price frame's datetime64 'date'.
    rolled["date"] = pd.to_datetime(rolled["date"])
    return rolled


def assign_split(dates: pd.Series) -> pd.Series:
    """Label each row train/dev/test by the paper's fixed chronological boundaries."""
    out = pd.Series(index=dates.index, dtype="object")
    d = dates
    out[(d >= config.TRAIN_START) & (d < config.DEV_START)] = "train"
    out[(d >= config.DEV_START) & (d < config.TEST_START)] = "dev"
    out[(d >= config.TEST_START) & (d < config.TEST_END)] = "test"
    return out


def build_combined(tickers: list[str], sent_df: pd.DataFrame | None) -> pd.DataFrame:
    """Join price + sentiment features per ticker-day, attach label + split, drop the
    ambiguous-movement rows and rows without a valid split."""
    frames = []
    for ticker in tickers:
        try:
            prices = load_prices(ticker)
        except FileNotFoundError:
            continue
        pf = compute_price_features(prices)
        pf["ticker"] = ticker

        if sent_df is not None:
            tsent = sent_df[sent_df["ticker"] == ticker]
            rolled = _roll_sentiment_onto_trading_days(pf[["date"]].copy(), tsent)
            pf = pf.merge(rolled, on="date", how="left")
        frames.append(pf)

    df = pd.concat(frames, ignore_index=True)

    # Sentiment columns: a ticker-day with no tweets is a neutral/zero signal, not missing.
    for col, fill in [("sent_mean", 0.0), ("sent_tweet_count", 0), ("sent_bull_ratio", 0.5)]:
        if col in df.columns:
            df[col] = df[col].fillna(fill)
        else:
            df[col] = fill
    # Log-scale the raw count so a day with 200 tweets doesn't dwarf a day with 5.
    df["sent_tweet_count"] = np.log1p(df["sent_tweet_count"].astype(float))

    df["split"] = assign_split(df["date"])

    # A prior-day volume of 0 makes volume_change divide by zero -> +/-inf. Treat those
    # (and any other non-finite feature) as missing so the dropna below removes the row
    # rather than feeding an infinity into the scaler/model.
    df[config.PRICE_FEATURE_COLS] = df[config.PRICE_FEATURE_COLS].replace(
        [np.inf, -np.inf], np.nan
    )

    # Keep only well-posed rows: inside the split window, with a defined label and defined
    # price features.
    before = len(df)
    df = df.dropna(subset=[config.TARGET_COL, *config.PRICE_FEATURE_COLS])
    df = df[df["split"].notna()].copy()
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)
    print(f"Combined table: {len(df)} usable ticker-days (dropped {before - len(df)}).")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the StockNet feature table.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Subset of tickers (default: all 88).")
    parser.add_argument("--skip-sentiment", action="store_true", help="Price features only (no model needed).")
    parser.add_argument("--reuse-sentiment-cache", action="store_true",
                        help="Load cached sentiment parquet instead of re-scoring tweets.")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers = args.tickers or list_tickers()
    print(f"Building features for {len(tickers)} ticker(s).")

    sent_df = None
    if not args.skip_sentiment:
        if args.reuse_sentiment_cache and config.SENTIMENT_FEATURES_PATH.exists():
            print(f"Reusing cached sentiment features: {config.SENTIMENT_FEATURES_PATH}")
            sent_df = pd.read_parquet(config.SENTIMENT_FEATURES_PATH)
            sent_df = sent_df[sent_df["ticker"].isin(tickers)]
        else:
            print("Scoring StockNet tweets through the fine-tuned sentiment model...")
            sent_df = build_sentiment_features(tickers)
            sent_df.to_parquet(config.SENTIMENT_FEATURES_PATH, index=False)
            print(f"Cached sentiment features -> {config.SENTIMENT_FEATURES_PATH}")

    combined = build_combined(tickers, sent_df)
    combined.to_parquet(config.COMBINED_FEATURES_PATH, index=False)
    print(f"Wrote combined feature table -> {config.COMBINED_FEATURES_PATH}")
    print(combined["split"].value_counts())


if __name__ == "__main__":
    main()
