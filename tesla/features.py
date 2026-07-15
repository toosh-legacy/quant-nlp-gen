"""Assemble the per-trading-day TSLA feature table used by both models.

Layers, all strictly causal (each feature uses only information available by the
close of its own day; the forward labels are the only thing that looks ahead):

  * technical      — the parent project's price factors (momentum, MAs, RSI, MACD, vol)
  * close-to-close HAR + range-based (Parkinson / Garman-Klass) volatility, in logs
  * INTRADAY       — the new lever: realized volatility built from ~2 years of hourly
                     bars (log_iv_rv_*), the intraday range, and the overnight gap
  * sentiment      — DistilBERT-scored Musk/Tesla tweets rolled onto trading days

Labels come in two families:
  * direction  — up/down of the close[t]->close[t+h] return, sqrt(h)-scaled band, per
                 h in config.TSLA_DIR_HORIZONS (same rule as the parent project)
  * volatility — forward log realized vol over the next h days, built TWICE: from
                 daily close-to-close returns (`cc`, full history, the apples-to-
                 apples target) and from intraday realized variance (`iv`, ~2yr, the
                 intraday-enriched target). Each with its persistence baseline.

Single-stock caveat, stated honestly: there is NO cross-sectional feature here (the
universe run's per-day rank/relative-vol, its single strongest extra signal, is
undefined for one ticker). Losing it is part of what we are measuring.

Run:
    python tesla/features.py            # build + cache + print coverage
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from features.build_features import _wilder_rsi  # noqa: E402  (reuse the exact RSI)
from tesla.fetch_prices import build_intraday_daily, fetch_daily  # noqa: E402
from tesla.sentiment_features import build_daily_sentiment  # noqa: E402

EPS = 1e-8

# ---- Feature groups (the model modules import these) ----
TECH_FEATURES = config.PRICE_FEATURE_COLS + ["volatility_10d", "volatility_20d"]
HAR_CC_FEATURES = ["log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q"]
RANGE_FEATURES = ["log_pk_d", "log_gk_d", "log_gk_w", "log_gk_m"]
EXTRA_VOL_FEATURES = ["ret_5", "ret_21", "log_semivol_21", "turnover_z"]
INTRADAY_FEATURES = ["log_iv_rv_d", "log_iv_rv_w", "log_iv_rv_m", "iv_range", "overnight_gap"]
SENT_FEATURES = config.SENTIMENT_FEATURE_COLS

# Direction feature variants (mirrors the parent's price_only / sentiment_only / combined).
DIR_PRICE_FEATURES = TECH_FEATURES + HAR_CC_FEATURES + RANGE_FEATURES
DIR_INTRADAY_FEATURES = DIR_PRICE_FEATURES + INTRADAY_FEATURES

# Volatility feature sets: a daily-only set (full history) and a +intraday set.
VOL_DAILY_FEATURES = HAR_CC_FEATURES + RANGE_FEATURES + EXTRA_VOL_FEATURES
VOL_INTRADAY_FEATURES = VOL_DAILY_FEATURES + INTRADAY_FEATURES


def cc_fwd_col(h: int) -> str:      # forward log realized vol, close-to-close target
    return f"y_logrv_cc_{h}"


def iv_fwd_col(h: int) -> str:      # forward log realized vol, intraday target
    return f"y_logrv_iv_{h}"


def cc_persist_col(h: int) -> str:  # persistence baseline for the cc target
    return f"persist_cc_{h}"


def iv_persist_col(h: int) -> str:  # persistence baseline for the iv target
    return f"persist_iv_{h}"


def _price_and_vol_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Technical + HAR + range-based + momentum/semivol/turnover features (all
    close-to-close / daily-OHLC), for the single TSLA series."""
    df = daily.sort_values("date").reset_index(drop=True).copy()
    px = df["close"]

    # --- Technical (identical formulas to features/build_features.compute_price_features) ---
    df["return_1d"] = px.pct_change(1)
    df["return_5d"] = px.pct_change(5)
    df["ma_5_ratio"] = px / px.rolling(5).mean()
    df["ma_10_ratio"] = px / px.rolling(10).mean()
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["volume_change"] = df["volume"].pct_change(1)
    df["rsi_14"] = _wilder_rsi(px, 14)
    ema_12 = px.ewm(span=12, adjust=False).mean()
    ema_26 = px.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Close-to-close HAR (log), matching features/universe_data.py ---
    ret = df["return_1d"]
    r2 = ret ** 2
    df["log_rv_d"] = np.log(ret.abs() + EPS)
    df["log_rv_w"] = np.log(np.sqrt(r2.rolling(5).mean()) + EPS)
    df["log_rv_m"] = np.log(np.sqrt(r2.rolling(21).mean()) + EPS)
    df["log_rv_q"] = np.log(np.sqrt(r2.rolling(63).mean()) + EPS)

    # --- Range-based daily vol (Parkinson / Garman-Klass), log ---
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    pk_var = (hl ** 2) / (4.0 * np.log(2.0))
    gk_var = (0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2).clip(lower=EPS)
    df["log_pk_d"] = np.log(np.sqrt(pk_var) + EPS)
    df["log_gk_d"] = np.log(np.sqrt(gk_var) + EPS)
    df["log_gk_w"] = np.log(np.sqrt(gk_var.rolling(5).mean()) + EPS)
    df["log_gk_m"] = np.log(np.sqrt(gk_var.rolling(21).mean()) + EPS)

    # --- Momentum / downside / turnover ---
    df["ret_5"] = px.pct_change(5)
    df["ret_21"] = px.pct_change(21)
    r2_down = np.where(ret < 0, r2, 0.0)
    df["log_semivol_21"] = np.log(np.sqrt(pd.Series(r2_down, index=df.index).rolling(21).mean()) + EPS)
    logv = np.log(df["volume"].replace(0, np.nan) + 1.0)
    df["turnover_z"] = (logv - logv.rolling(21).mean()) / logv.rolling(21).std()

    df["_r2_cc"] = r2  # kept for the cc labels below
    return df


def _add_intraday_features(df: pd.DataFrame, iv_daily: pd.DataFrame) -> pd.DataFrame:
    """Merge intraday-derived daily features + build the intraday HAR components.
    Only defined over the hourly window (~2 years); older rows keep NaN here."""
    df = df.merge(iv_daily, on="date", how="left")            # iv_rv, iv_rvar, iv_range, n_bars
    df["log_iv_rv_d"] = np.log(df["iv_rv"] + EPS)
    df["log_iv_rv_w"] = np.log(np.sqrt(df["iv_rvar"].rolling(5).mean()) + EPS)
    df["log_iv_rv_m"] = np.log(np.sqrt(df["iv_rvar"].rolling(21).mean()) + EPS)
    # Overnight gap: today's open vs yesterday's close (a real intraday-era signal;
    # for the vol model this captures gap risk the within-day RV misses).
    df["overnight_gap"] = np.log(df["open"] / df["close"].shift(1))
    return df


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Direction labels + both volatility-target families + embargo dates."""
    px = df["close"]

    # Direction (sqrt(h)-scaled band, same as config.horizon_thresholds).
    for h in config.TSLA_DIR_HORIZONS:
        up_thr, down_thr = config.horizon_thresholds(h)
        fwd_return = px.shift(-h) / px - 1.0
        df[config.movement_col(h)] = np.where(
            fwd_return >= up_thr, 1, np.where(fwd_return <= down_thr, 0, np.nan))
        df[config.label_date_col(h)] = df["date"].shift(-h)

    # Volatility targets: forward log realized vol, built from cc returns AND from
    # intraday realized variance. rolling(h).mean(var) then shift(-h) => next-h-day
    # realized variance, brought back to day t. persistence = current h-day level.
    r2_cc = df["_r2_cc"]
    iv_var = df["iv_rvar"]
    for h in config.TSLA_VOL_HORIZONS:
        fwd_cc = r2_cc.rolling(h).mean().shift(-h)
        df[cc_fwd_col(h)] = np.log(np.sqrt(fwd_cc) + EPS)
        df[cc_persist_col(h)] = np.log(np.sqrt(r2_cc.rolling(h).mean()) + EPS)

        fwd_iv = iv_var.rolling(h).mean().shift(-h)
        df[iv_fwd_col(h)] = np.log(np.sqrt(fwd_iv) + EPS)
        df[iv_persist_col(h)] = np.log(np.sqrt(iv_var.rolling(h).mean()) + EPS)
        # Shared embargo date (same feature/label geometry for both targets).
        if config.label_date_col(h) not in df.columns:
            df[config.label_date_col(h)] = df["date"].shift(-h)
    return df


def build_features(force: bool = False) -> pd.DataFrame:
    """Build (or load) the full TSLA feature+label table, cached to parquet."""
    cache = config.TSLA_FEATURES_CACHE
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    daily = fetch_daily()
    df = _price_and_vol_features(daily)
    df = _add_intraday_features(df, build_intraday_daily())
    df = df.merge(build_daily_sentiment(), on="date", how="left")
    # Sentiment: a day outside the tweet window is a neutral/zero signal, not missing.
    for col, fill in [("sent_mean", 0.0), ("sent_tweet_count", 0.0),
                      ("sent_bull_ratio", 0.5), ("sent_mean_3d", 0.0)]:
        df[col] = df[col].fillna(fill) if col in df.columns else fill
    df = _add_labels(df)

    # Non-finite guard on the daily feature set (e.g. volume_change /0 -> inf).
    daily_feats = DIR_PRICE_FEATURES + EXTRA_VOL_FEATURES
    df[daily_feats] = df[daily_feats].replace([np.inf, -np.inf], np.nan)
    # Keep rows with the DAILY features present (full history). Intraday features and
    # the iv-target are intentionally allowed to be NaN before the hourly window —
    # each model drops the rows it needs, so the daily model keeps 2015-2026.
    df = df.dropna(subset=daily_feats).reset_index(drop=True)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"Cached TSLA features -> {cache} ({len(df)} rows, "
          f"{df['date'].min().date()}..{df['date'].max().date()})")
    return df


def main() -> None:
    df = build_features(force=True)
    n_intraday = int(df["log_iv_rv_d"].notna().sum())
    print(f"\nRows: {len(df)}  |  with intraday features: {n_intraday} "
          f"({df.loc[df['log_iv_rv_d'].notna(), 'date'].min().date()}.."
          f"{df['date'].max().date()})")
    print(f"Direction feature groups: price={len(DIR_PRICE_FEATURES)}, "
          f"+intraday={len(DIR_INTRADAY_FEATURES)}, sentiment={len(SENT_FEATURES)}")
    print("Label coverage:")
    for h in config.TSLA_DIR_HORIZONS:
        print(f"  direction h={h:>2}: {int(df[config.movement_col(h)].notna().sum())} labeled")
    for h in config.TSLA_VOL_HORIZONS:
        cc = int(df[cc_fwd_col(h)].notna().sum())
        iv = int(df[iv_fwd_col(h)].notna().sum())
        print(f"  vol h={h:>2}: cc-target={cc}, iv-target={iv}")


if __name__ == "__main__":
    main()
