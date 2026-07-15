"""Task B (pro) — build the best volatility dataset we can, for free.

The StockNet volatility result was capped by *data*: 88 tickers, 2 years, daily returns. This
module removes those limits:

  * Universe: ~80 liquid large-caps (below), downloaded daily from Yahoo Finance via yfinance.
  * History: 2005-2024 — ~10x the data, spanning the 2008 and 2020 volatility regimes, so a
    model learns real vol dynamics and walk-forward gives genuine multi-regime confidence
    intervals.
  * Target: LOG realized volatility. Volatility is log-normal and highly persistent in log
    space, which is exactly why the HAR literature reports much higher R^2 there than in raw
    vol space.
  * Features: best-practice, including the CROSS-SECTIONAL signal VIX couldn't provide — each
    stock's volatility RELATIVE to the universe that day (ratio + rank), plus multi-scale HAR,
    the leverage/downside-vol effect, and turnover.

Everything is causal (features use only past returns; the current level is the persistence
baseline; the forward realized vol is the label only). Cached to parquet so the download +
feature build run once.

Run directly to (re)build the cache:
    python features/universe_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# A diversified large-cap universe (tech, financials, healthcare, energy, staples,
# industrials, discretionary, utilities, materials, comms). Tickers that IPO'd after 2005
# simply contribute a shorter history — that's fine.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "INTC", "CSCO", "ORCL", "IBM",
    "ADBE", "CRM", "QCOM", "TXN", "AMD", "JPM", "BAC", "WFC", "C", "GS",
    "MS", "AXP", "BLK", "SCHW", "USB", "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "LLY", "BMY", "AMGN", "GILD", "CVS", "XOM", "CVX", "COP", "SLB", "EOG",
    "PSX", "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "HD",
    "LOW", "TGT", "DIS", "CMCSA", "VZ", "T", "NFLX", "BA", "CAT", "GE",
    "HON", "UPS", "UNP", "MMM", "LMT", "DE", "F", "GM", "NEE", "DUK",
    "SO", "D", "AEP", "LIN", "APD", "SHW", "FCX", "NEM", "DOW", "PXD",
]

UNIVERSE_START = "2005-01-01"
UNIVERSE_END = "2024-12-31"
EPS = 1e-8

RAW_CACHE = config.DATA_DIR / "universe_prices.parquet"
FEATURES_CACHE = config.DATA_DIR / "universe_vol_features.parquet"

# Prediction horizons for this task (trading days): one week, two weeks, one month.
UNIV_HORIZONS = [5, 10, 21]

# The model feature columns (all causal).
UNIV_FEATURES = [
    "log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q",   # multi-scale HAR realized vol (log)
    "ret_5", "ret_21",                                 # momentum
    "log_semivol_21",                                  # downside/leverage-effect vol
    "turnover_z",                                       # volume z-score
    "log_rv_rel", "rv_rank",                            # CROSS-SECTIONAL: vol vs the universe
]


def univ_fwd_col(h: int) -> str:
    return f"y_logrv_{h}"          # regression target: log next-h-day realized vol


def univ_persist_col(h: int) -> str:
    return f"persist_logrv_{h}"    # persistence baseline: log current h-day realized vol


def download_universe(force: bool = False) -> pd.DataFrame:
    """Download daily adjusted close + volume for the universe, cached as a long table
    (date, ticker, close, volume)."""
    if RAW_CACHE.exists() and not force:
        return pd.read_parquet(RAW_CACHE)

    import yfinance as yf

    print(f"Downloading {len(UNIVERSE)} tickers {UNIVERSE_START}..{UNIVERSE_END} (once) ...")
    data = yf.download(UNIVERSE, start=UNIVERSE_START, end=UNIVERSE_END,
                       auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError("Universe download failed / empty (Yahoo may be rate-limiting — retry).")
    close = data["Close"].stack().rename("close")
    volume = data["Volume"].stack().rename("volume")
    long = pd.concat([close, volume], axis=1).reset_index()
    long.columns = ["date", "ticker", "close", "volume"]
    long = long.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    long.to_parquet(RAW_CACHE, index=False)
    print(f"Cached raw universe prices -> {RAW_CACHE} ({len(long)} rows, "
          f"{long['ticker'].nunique()} tickers)")
    return long


def _roll(g: pd.core.groupby.SeriesGroupBy, fn):
    """Helper: apply a per-ticker rolling function that returns a Series aligned to df."""
    return fn(g)


def build_universe_features(force: bool = False) -> pd.DataFrame:
    """Compute causal volatility features + log-RV labels for every ticker-day."""
    if FEATURES_CACHE.exists() and not force:
        return pd.read_parquet(FEATURES_CACHE)

    df = download_universe().sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker")

    df["ret"] = g["close"].pct_change()
    r2 = df["ret"] ** 2
    df["_r2"] = r2

    def gr(col):
        return df.groupby("ticker")[col]

    # --- Backward realized vol at multiple scales (HAR components), in logs ---
    df["log_rv_d"] = np.log(df["ret"].abs() + EPS)
    df["log_rv_w"] = np.log(np.sqrt(gr("_r2").transform(lambda s: s.rolling(5).mean())) + EPS)
    df["log_rv_m"] = np.log(np.sqrt(gr("_r2").transform(lambda s: s.rolling(21).mean())) + EPS)
    df["log_rv_q"] = np.log(np.sqrt(gr("_r2").transform(lambda s: s.rolling(63).mean())) + EPS)

    # --- Momentum ---
    df["ret_5"] = gr("close").transform(lambda s: s.pct_change(5))
    df["ret_21"] = gr("close").transform(lambda s: s.pct_change(21))

    # --- Downside / leverage effect: realized vol of NEGATIVE-return days only ---
    df["_r2_down"] = np.where(df["ret"] < 0, r2, 0.0)
    df["log_semivol_21"] = np.log(
        np.sqrt(gr("_r2_down").transform(lambda s: s.rolling(21).mean())) + EPS)

    # --- Turnover: z-score of log volume vs its own 21-day history ---
    logv = np.log(df["volume"].replace(0, np.nan) + 1.0)
    df["_logv"] = logv
    mu = gr("_logv").transform(lambda s: s.rolling(21).mean())
    sd = gr("_logv").transform(lambda s: s.rolling(21).std())
    df["turnover_z"] = (logv - mu) / sd

    # --- Cross-sectional: this stock's 21d vol vs the universe that day ---
    df["_rv_m"] = np.sqrt(gr("_r2").transform(lambda s: s.rolling(21).mean()))
    mkt = df.groupby("date")["_rv_m"].transform("mean")
    df["log_rv_rel"] = np.log((df["_rv_m"] + EPS) / (mkt + EPS))
    df["rv_rank"] = df.groupby("date")["_rv_m"].rank(pct=True)

    # --- Labels: log realized vol over the NEXT h days; + persistence baseline ---
    for h in UNIV_HORIZONS:
        fwd_rv = np.sqrt(gr("_r2").transform(lambda s: s.rolling(h).mean()).shift(-h))
        df[univ_fwd_col(h)] = np.log(fwd_rv + EPS)
        cur_rv = np.sqrt(gr("_r2").transform(lambda s: s.rolling(h).mean()))
        df[univ_persist_col(h)] = np.log(cur_rv + EPS)
        # date of the h-days-ahead observation, for the embargo at fold boundaries.
        df[f"label_date_{h}"] = gr("date").transform(lambda s: s.shift(-h))

    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    # Keep rows with all features present (drops each ticker's early history).
    df = df.dropna(subset=UNIV_FEATURES).reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_CACHE, index=False)
    print(f"Cached universe vol features -> {FEATURES_CACHE} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    feats = build_universe_features(force=True)
    print(feats[["date", "ticker", *UNIV_FEATURES]].tail(3).to_string())
    for h in UNIV_HORIZONS:
        print(f"h={h}: labeled rows = {feats[univ_fwd_col(h)].notna().sum()}")
