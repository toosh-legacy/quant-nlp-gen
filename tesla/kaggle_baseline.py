"""The public-dataset side of the deep dive: load a public Tesla news/tweets
dataset that pairs text with the TSLA era, to (a) feed our own DistilBERT
sentiment pipeline and (b) serve as the "public Tesla-prediction project" we
sanity-check our methodology against.

We use Elon Musk's tweets (2015-2020) as the public text source — Musk's tweets
are the canonical, sustained retail/news driver of TSLA, which is exactly the
narrative this single-stock study is about. The default source is a no-auth
GitHub mirror of a well-known Kaggle "Elon Musk tweets" dataset; if it is
unavailable the loader prints exactly what to download and where to drop it.

The loader is deliberately SCHEMA-FLEXIBLE: drop any CSV(s) with a date column
and a text column into data/kaggle_tsla/ and it will find them, so swapping in a
different public dataset needs no code change.

Run:
    python tesla/kaggle_baseline.py            # fetch (if needed) + coverage report
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# No-auth mirror of a well-known Kaggle "Elon Musk tweets" dataset (id,user,text,
# date,retweets,favorites), 2015-07..2020-07. If this 404s, download any Tesla /
# Musk tweets CSV from Kaggle and drop it in data/kaggle_tsla/ (see fetch note).
DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/Umarfoo/Tesla_Stock_ETL_Project/"
    "main/Output%20Data/musk_tweets.csv"
)
DEFAULT_LOCAL_NAME = "musk_tweets.csv"

# Column-name hints for schema auto-detection (case-insensitive substring match).
_DATE_HINTS = ("date", "created", "timestamp", "time")
_TEXT_HINTS = ("text", "tweet", "headline", "title", "content", "body")
_SENT_HINTS = ("sentiment", "polarity", "label")

_FETCH_NOTE = (
    "No public tweets CSV found in {dir}.\n"
    "  Auto-fetch from the default mirror failed ({err}).\n"
    "  Fix: download a Tesla/Musk tweets dataset and drop the CSV there, e.g.\n"
    "    Kaggle: 'Elon Musk Tweets (2010-2021)' or\n"
    "            'omermetinn/tweets-about-the-top-companies-from-2015-to-2020',\n"
    "  then re-run. Any CSV with a date column + a text column works."
)


def _try_fetch(dest: Path) -> bool:
    """Best-effort no-auth download of the default source. Returns success."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching public tweets -> {dest} ...")
        urllib.request.urlretrieve(DEFAULT_SOURCE_URL, dest)  # noqa: S310 (trusted URL)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        print(f"  fetch failed: {e}")
        return False


def _pick_column(cols: list[str], hints: tuple[str, ...]) -> str | None:
    low = {c: c.lower() for c in cols}
    for h in hints:
        for c in cols:
            if h in low[c]:
                return c
    return None


def _clean_tweet(text: str) -> str:
    """Undo the b'...' byte-string wrapping some scrapes leave behind; collapse space."""
    s = str(text).strip()
    if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
        s = s[2:-1]
    return s.replace("\\n", " ").replace("\\t", " ").strip()


def load_public_tweets(fetch: bool = True) -> pd.DataFrame:
    """Return the public text dataset as tidy columns: `datetime` (tz-naive),
    `date` (normalized), `text`, and optional `ext_sentiment`. Concatenates every
    CSV in data/kaggle_tsla/, auto-detecting the date and text columns of each."""
    ddir = config.KAGGLE_TSLA_DIR
    csvs = sorted(ddir.glob("*.csv")) if ddir.exists() else []
    if not csvs and fetch:
        if _try_fetch(ddir / DEFAULT_LOCAL_NAME):
            csvs = sorted(ddir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(_FETCH_NOTE.format(dir=ddir, err="no local CSVs"))

    frames = []
    for path in csvs:
        raw = pd.read_csv(path)
        dcol = _pick_column(list(raw.columns), _DATE_HINTS)
        tcol = _pick_column(list(raw.columns), _TEXT_HINTS)
        if dcol is None or tcol is None:
            print(f"  skip {path.name}: no date/text column detected "
                  f"(cols={list(raw.columns)})")
            continue
        out = pd.DataFrame({
            "datetime": pd.to_datetime(raw[dcol], utc=True, errors="coerce"),
            "text": raw[tcol].map(_clean_tweet),
        })
        scol = _pick_column(list(raw.columns), _SENT_HINTS)
        if scol is not None and pd.api.types.is_numeric_dtype(raw[scol]):
            out["ext_sentiment"] = pd.to_numeric(raw[scol], errors="coerce")
        out = out.dropna(subset=["datetime"])
        out["datetime"] = out["datetime"].dt.tz_localize(None)
        out = out[out["text"].str.len() >= 3]
        frames.append(out)
        print(f"  loaded {path.name}: {len(out)} rows "
              f"(date='{dcol}', text='{tcol}'"
              f"{', sent=' + repr(scol) if 'ext_sentiment' in out else ''})")

    if not frames:
        raise FileNotFoundError(_FETCH_NOTE.format(dir=ddir, err="no usable schema"))
    df = pd.concat(frames, ignore_index=True).sort_values("datetime")
    df["date"] = df["datetime"].dt.normalize()
    return df.reset_index(drop=True)


def daily_tweet_volume(tweets: pd.DataFrame) -> pd.DataFrame:
    """Calendar-date tweet counts (a raw news-intensity series, useful for coverage
    reporting and as an optional feature)."""
    return (tweets.groupby("date").size().rename("tweet_count")
            .reset_index())


def main() -> None:
    tweets = load_public_tweets()
    print(f"\nPublic tweets: {len(tweets)} rows, "
          f"{tweets['date'].min().date()}..{tweets['date'].max().date()}")
    vol = daily_tweet_volume(tweets)
    print(f"Days with >=1 tweet: {len(vol)}   "
          f"median tweets/active-day: {int(vol['tweet_count'].median())}")
    by_year = tweets.groupby(tweets["date"].dt.year).size()
    print("Tweets per year:")
    for yr, n in by_year.items():
        print(f"  {yr}: {n}")
    if "ext_sentiment" in tweets:
        print(f"Dataset-provided sentiment present (mean={tweets['ext_sentiment'].mean():.3f}) "
              "— will be used as an independent cross-check.")
    else:
        print("No dataset-provided sentiment column — our DistilBERT scores stand alone "
              "(cross-check falls back to score distribution + examples).")


if __name__ == "__main__":
    main()
