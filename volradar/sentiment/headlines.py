"""News-tone overlay — and an honest account of why it is not a model input.

The plan was to score recent financial headlines with the fine-tuned DistilBERT classifier
in this package and test whether news tone adds anything to the volatility forecast.

**What we found instead.** Free headline sources do not provide the history that test
requires. Yahoo Finance exposes only the most recent handful of articles per ticker — days
of coverage, not years. A feature can only be validated by walk-forward if it exists across
the walk-forward period, so there is no honest way to measure whether news tone improves a
2012-2024 forecast using free data. Fabricating the history by back-filling from today's
articles would leak the future into the past, which is precisely the error the rest of this
project is built to prevent.

**So sentiment is not in the model.** It is exposed as a descriptive current-conditions
panel, clearly separated from the forecast, and labelled as unvalidated. Wiring an
unmeasured signal into a forecast that advertises its own track record would undermine the
one thing that makes the track record worth reading.

To validate it properly you would need a historical headline archive with timestamps
(commercial vendors sell these). `score_headlines` below is the piece that would slot in.

Run:
    python -m volradar.sentiment.headlines
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config

TONE_PATH = config.REPORT_DIR / "news_tone.json"

# Tickers to sample for a market-level tone reading: the index plus each sector ETF.
TONE_TICKERS = [config.MARKET_TICKER] + list(config.SECTOR_ETFS)


def fetch_headlines(tickers: list[str] | None = None) -> pd.DataFrame:
    """Whatever recent headlines the free source will give us. Often very few."""
    import yfinance as yf

    tickers = tickers or TONE_TICKERS
    rows = []
    for t in tickers:
        try:
            for item in (yf.Ticker(t).news or []):
                content = item.get("content", item)
                title = content.get("title") or item.get("title")
                pub = content.get("pubDate") or item.get("providerPublishTime")
                if not title:
                    continue
                rows.append({"ticker": t, "title": title, "published": pub})
        except Exception as exc:              # network/schema drift must not break the CLI
            print(f"  [warn] headlines for {t}: {exc}")
    return pd.DataFrame(rows)


def score_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """Label each headline bearish / bullish / neutral with the fine-tuned classifier.

    Returns the frame unscored (with a reason) if the optional sentiment extras or the
    trained model are not present, so this never becomes a hard dependency of the pipeline.
    """
    if df.empty:
        return df.assign(label=None, score=np.nan)

    model_dir = config.DATA_DIR / "sentiment_model"
    if not model_dir.exists():
        print("  [skip] no fine-tuned model at data/sentiment_model — "
              "run volradar/sentiment/train_sentiment_model.py first")
        return df.assign(label=None, score=np.nan)

    try:
        from transformers import pipeline
    except ImportError:
        print("  [skip] transformers not installed — pip install -r requirements-sentiment.txt")
        return df.assign(label=None, score=np.nan)

    clf = pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir),
                   truncation=True, max_length=128)
    preds = clf(df["title"].tolist())
    names = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    labels, scores = [], []
    for p in preds:
        raw = p["label"]
        idx = int(raw.split("_")[-1]) if raw.startswith("LABEL_") else None
        labels.append(names.get(idx, raw))
        scores.append(float(p["score"]))
    return df.assign(label=labels, score=scores)


def tone_summary(scored: pd.DataFrame) -> dict:
    """A descriptive tone reading, with its own coverage stated plainly."""
    if scored.empty or scored["label"].isna().all():
        return {"available": False,
                "reason": "no scored headlines (missing model, extras, or source data)"}

    n = len(scored)
    bull = int((scored["label"] == "Bullish").sum())
    bear = int((scored["label"] == "Bearish").sum())
    return {
        "available": True,
        "n_headlines": n,
        "n_tickers": int(scored["ticker"].nunique()),
        "bullish": bull, "bearish": bear,
        "neutral": int((scored["label"] == "Neutral").sum()),
        "net_tone": round((bull - bear) / n, 3),
        "validated_as_predictor": False,
        "caveat": ("Descriptive only. Free sources provide days of headline history, not "
                   "years, so this signal could not be walk-forward validated and is NOT "
                   "an input to the volatility forecast."),
    }


def main() -> None:
    import json
    config.ensure_dirs()
    print("Fetching recent headlines (free source — expect thin coverage)...")
    raw = fetch_headlines()
    print(f"  {len(raw)} headlines across {raw['ticker'].nunique() if len(raw) else 0} tickers")
    scored = score_headlines(raw)
    summary = tone_summary(scored)
    TONE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote -> {TONE_PATH}")


if __name__ == "__main__":
    main()
