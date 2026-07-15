"""Score the public Tesla/Musk tweets with the project's fine-tuned DistilBERT and
aggregate to one causal sentiment row per TSLA trading day.

This reuses the exact scoring pipeline from the multi-stock project
(features/build_features.py) — same model, same signed score P(bull)-P(bear),
same trailing-window smoothing, same "roll a tweet onto the next actionable
trading session" rule — so the TSLA sentiment factor is measured identically to
the parent study. That methodological consistency is the point (it lets us sanity
-check our own pipeline on a fresh, independent text source).

Output columns are exactly config.SENTIMENT_FEATURE_COLS, so the downstream
feature/model code treats TSLA sentiment just like StockNet sentiment.

Run:
    python tesla/sentiment_features.py            # score + cache + cross-check report
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
# Reuse the parent project's scoring + calendar-alignment utilities verbatim.
from features.build_features import (  # noqa: E402
    _load_sentiment_pipeline, _roll_sentiment_onto_trading_days, score_texts,
)
from tesla.fetch_prices import fetch_daily  # noqa: E402
from tesla.kaggle_baseline import load_public_tweets  # noqa: E402


def _score_tweets_by_calendar_date(tweets: pd.DataFrame) -> pd.DataFrame:
    """Score every tweet and aggregate to (calendar date -> signed sentiment,
    tweet count, bull ratio), identical to build_sentiment_features but for a flat
    tweets frame instead of StockNet's per-date files."""
    tokenizer, model = _load_sentiment_pipeline()
    bear_idx = config.NAME_TO_LABEL["Bearish"]
    bull_idx = config.NAME_TO_LABEL["Bullish"]

    texts = tweets["text"].astype(str).tolist()
    print(f"Scoring {len(texts)} tweets through DistilBERT (CPU) ...")
    probs = score_texts(tokenizer, model, texts)      # (N, 3) softmax
    preds = probs.argmax(axis=1)
    signed = probs[:, bull_idx] - probs[:, bear_idx]  # per-tweet signed sentiment

    scored = pd.DataFrame({
        "date": tweets["date"].to_numpy(),
        "signed": signed,
        "is_bull": (preds == bull_idx).astype(int),
        "is_bear": (preds == bear_idx).astype(int),
        "_prob_bull": probs[:, bull_idx],
    })
    grp = scored.groupby("date")
    daily = pd.DataFrame({
        "sent_mean": grp["signed"].mean(),
        "sent_tweet_count": grp.size(),
        "_n_bull": grp["is_bull"].sum(),
        "_n_bear": grp["is_bear"].sum(),
    }).reset_index()
    denom = (daily["_n_bull"] + daily["_n_bear"]).replace(0, np.nan)
    daily["sent_bull_ratio"] = (daily["_n_bull"] / denom).fillna(0.5)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily[["date", "sent_mean", "sent_tweet_count", "sent_bull_ratio"]], scored


def build_daily_sentiment(force: bool = False) -> pd.DataFrame:
    """Per-TSLA-trading-day sentiment features (config.SENTIMENT_FEATURE_COLS),
    cached. Days with no tweets are a neutral/zero signal (fill 0 / 0.5), matching
    the parent project's convention exactly."""
    cache = config.TSLA_DAILY_SENTIMENT_CACHE
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    tweets = load_public_tweets()
    daily_cal, _ = _score_tweets_by_calendar_date(tweets)

    # Roll each calendar-date score onto the next actionable TSLA trading session,
    # then merge onto the full daily calendar so every trading day has a row.
    trading = fetch_daily()[["date"]].copy()
    rolled = _roll_sentiment_onto_trading_days(trading, daily_cal)
    df = trading.merge(rolled, on="date", how="left")

    for col, fill in [("sent_mean", 0.0), ("sent_tweet_count", 0), ("sent_bull_ratio", 0.5)]:
        df[col] = df[col].fillna(fill) if col in df.columns else fill
    df["sent_tweet_count"] = np.log1p(df["sent_tweet_count"].astype(float))  # log-scale
    df = df.sort_values("date").reset_index(drop=True)
    df["sent_mean_3d"] = df["sent_mean"].rolling(
        config.SENT_TRAILING_WINDOW, min_periods=1).mean()

    out = df[["date", *config.SENTIMENT_FEATURE_COLS]]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    print(f"Cached TSLA daily sentiment -> {cache} ({len(out)} trading days)")
    return out


def cross_check(tweets: pd.DataFrame, scored: pd.DataFrame) -> None:
    """Sanity-check our DistilBERT scores. If the dataset ships its own sentiment,
    correlate against it (the intended cross-check). Otherwise fall back to the
    score distribution + a few labeled examples so the numbers are still auditable."""
    print("\n=== Sentiment cross-check ===")
    if "ext_sentiment" in tweets.columns:
        merged = tweets.reset_index(drop=True).join(scored["signed"].reset_index(drop=True))
        c = merged[["ext_sentiment", "signed"]].dropna()
        if len(c) >= 20:
            print(f"  corr(our signed sentiment, dataset sentiment) = "
                  f"{c['ext_sentiment'].corr(c['signed']):+.3f}  (n={len(c)})")
            return
    frac_bull = float((scored["signed"] > 0.15).mean())
    frac_bear = float((scored["signed"] < -0.15).mean())
    print("  (no dataset-provided sentiment column — reporting distribution instead)")
    print(f"  scored {len(scored)} tweets: bullish={frac_bull:.1%}, bearish={frac_bear:.1%}, "
          f"neutral={1 - frac_bull - frac_bear:.1%}")
    ex = scored.assign(text=tweets["text"].to_numpy()).sort_values("signed")
    print("  most bearish:", repr(ex.iloc[0]["text"][:90]), f"({ex.iloc[0]['signed']:+.2f})")
    print("  most bullish:", repr(ex.iloc[-1]["text"][:90]), f"({ex.iloc[-1]['signed']:+.2f})")


def main() -> None:
    tweets = load_public_tweets()
    daily_cal, scored = _score_tweets_by_calendar_date(tweets)
    cross_check(tweets, scored)

    out = build_daily_sentiment(force=True)
    active = (out["sent_tweet_count"] > 0).sum()
    print(f"\nDaily sentiment table: {len(out)} trading days, {active} with tweets "
          f"({active / len(out):.0%}).")
    print(out[out["sent_tweet_count"] > 0].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
