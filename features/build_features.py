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
# and the numbers stay comparable to the published baseline. The daily band (Xu & Cohen
# 2018) is +0.55% / -0.50%; for longer horizons it is scaled by sqrt(h) — see
# config.horizon_thresholds(), which is the single source of truth for the thresholds.


# ---------------------------------------------------------------------------
# StockNet IO
# ---------------------------------------------------------------------------
def list_tickers() -> list[str]:
    """All 88 tickers, derived from the price CSV filenames."""
    price_dir = config.STOCKNET_DIR / "price" / "raw"
    return sorted(p.stem for p in price_dir.glob("*.csv"))


def load_ticker_sectors() -> dict[str, str]:
    """Map ticker -> sector from StockNet's StockTable (tab-separated: Sector, $Symbol,
    Company). Used for the per-sector predictability analysis. Symbols carry a leading '$'."""
    path = config.STOCKNET_DIR / "StockTable"
    mapping: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        sector = parts[0].strip()
        symbol = parts[1].strip().lstrip("$").upper()
        if symbol:
            mapping[symbol] = sector
    return mapping


def load_prices(ticker: str) -> pd.DataFrame:
    """Load one ticker's raw daily OHLCV, ascending by date.

    We keep a lead-in before TRAIN_START so the slowest indicator (MACD's 26-day EMA +
    9-day signal, and the 14-day RSI) is defined on the first training day. We deliberately
    do NOT cut the series at TEST_END: prices run to 2017, and the extra tail is needed so
    that labels for the *last* feature days (close[t+h] for long horizons like 60 days) are
    computable. Those post-TEST_END rows are used only as label targets — they are dropped
    as *feature* rows later because their split is undefined.
    """
    path = config.STOCKNET_DIR / "price" / "raw" / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns=str.lower)
    df = df.rename(columns={"adj close": "adj_close"})
    df = df.sort_values("date").reset_index(drop=True)
    lead_in = pd.Timestamp(config.TRAIN_START) - pd.Timedelta(days=150)
    df = df[df["date"] >= lead_in]
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
def _wilder_rsi(px: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100), computed causally over past closes only.

    RSI measures how one-sided recent moves have been: >70 = overbought, <30 = oversold.
    Uses only data up to and including each day (the exponential averages are trailing), so
    it introduces no lookahead.
    """
    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing = EMA with alpha = 1/period (min_periods so early values are NaN).
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When there were no losses in the window, RSI is 100 by definition.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def compute_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Technical factors + multi-horizon movement labels, for one ticker.

    All feature columns use only data up to and including each row's own day. The labels
    look FORWARD by h trading days (that's the future we forecast) and are NOT features.
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

    # RSI(14) — momentum oscillator, causal.
    df["rsi_14"] = _wilder_rsi(px, 14)

    # MACD(12, 26, 9) — trend/momentum. All three lines are trailing EMAs (adjust=False),
    # so nothing here peeks at future prices.
    ema_12 = px.ewm(span=12, adjust=False).mean()
    ema_26 = px.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Labels: direction of the return over the next h trading days, per horizon ---
    # px.shift(-h) looks h rows FORWARD on purpose — this is the target, not a feature.
    # The band is scaled by sqrt(h) so the drop-rate of ambiguous days is ~constant across
    # horizons (see config.horizon_thresholds), keeping the horizon comparison fair.
    for h in config.HORIZONS:
        up_thr, down_thr = config.horizon_thresholds(h)
        fwd_return = px.shift(-h) / px - 1.0
        if h == 1:
            df["next_return_1"] = fwd_return  # kept for readability / tests
        df[config.movement_col(h)] = np.where(
            fwd_return >= up_thr, 1,
            np.where(fwd_return <= down_thr, 0, np.nan),
        )
        # Calendar date of the h-days-ahead close, used later to embargo rows whose outcome
        # window crosses a train/dev/test boundary.
        df[config.label_date_col(h)] = df["date"].shift(-h)

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
    # The three base per-day aggregates produced by scoring; sent_mean_3d is derived later.
    base_cols = ["sent_mean", "sent_tweet_count", "sent_bull_ratio"]
    trading_days = price_df["date"].sort_values().to_numpy()
    if len(sent_df) == 0:
        empty = pd.DataFrame(columns=["date", *base_cols])
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
    rolled = grp.rename(columns={"trading_day": "date"})[["date", *base_cols]]
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
    """Join price + sentiment features per ticker-day, attach the multi-horizon labels and
    the split of both the feature day and each horizon's outcome day (for the embargo).

    We keep every row with finite features here; per-horizon label validity + the embargo
    (feature-day split == outcome-day split) are applied at evaluation time so the same
    table serves all horizons.
    """
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

    # Trailing multi-day sentiment: a causal rolling mean of the daily signed sentiment,
    # per ticker, over the last SENT_TRAILING_WINDOW trading days (includes day t, which is
    # allowed — day-t sentiment is known by day t's close). Smooths single-day tweet noise.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["sent_mean_3d"] = (
        df.groupby("ticker")["sent_mean"]
        .transform(lambda s: s.rolling(config.SENT_TRAILING_WINDOW, min_periods=1).mean())
    )

    # Split of the feature day, and split of each horizon's outcome day (for the embargo).
    df["split"] = assign_split(df["date"])
    for h in config.HORIZONS:
        df[config.label_split_col(h)] = assign_split(df[config.label_date_col(h)])

    # A prior-day volume of 0 makes volume_change divide by zero -> +/-inf. Treat those
    # (and any other non-finite feature) as missing so the dropna below removes the row
    # rather than feeding an infinity into the scaler/model.
    df[config.PRICE_FEATURE_COLS] = df[config.PRICE_FEATURE_COLS].replace(
        [np.inf, -np.inf], np.nan
    )

    # Keep rows that are inside a split window and have all price features defined. We do
    # NOT drop on the labels here — a row may be valid for one horizon and not another; that
    # per-horizon filtering (plus the embargo) happens at evaluation time.
    before = len(df)
    df = df.dropna(subset=config.PRICE_FEATURE_COLS)
    df = df[df["split"].notna()].copy()
    n_labeled = {h: int(df[config.movement_col(h)].notna().sum()) for h in config.HORIZONS}
    print(f"Combined table: {len(df)} rows with features (dropped {before - len(df)}).")
    print(f"  labeled rows per horizon (pre-embargo): {n_labeled}")
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
