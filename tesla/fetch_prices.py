"""Fetch TSLA prices (daily + intraday) for free via yfinance, and turn the
intraday bars into a high-quality DAILY realized-volatility signal.

Why this matters: the parent project was capped by *data* — daily bars give one
noisy squared return per day, so its realized-vol proxy is crude. Intraday bars
let us measure a day's realized volatility from many within-day returns, which is
a far less noisy estimator (the whole point of the "realized volatility"
literature). This module is the "intraday info the multi-stock project couldn't
afford" lever, obtained with NO API key.

Yahoo's free tier limits intraday history, so we layer three intervals:
  * 1h over ~2 years  -> the HEADLINE realized-vol signal fed into daily features
    (~7 bars/session: a real RV estimate, far better than one daily squared return),
  * 5m over ~60 days and 1m over ~7 days -> recent high-res slices used only to
    VALIDATE that the hourly RV tracks finer-resolution RV (so we can honestly say
    what a paid 1-minute feed would and wouldn't add).

Everything is causal: intraday bars for session t summarize movement *within* day t
and become a feature known at that day's close. The forward realized vol (the
Task-B label) is built later, in tesla/features.py, and is never a feature.

Run:
    python tesla/fetch_prices.py            # download + cache everything, print checks
    python tesla/fetch_prices.py --force    # re-download
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

EPS = 1e-8
NY_TZ = "America/New_York"


# ---------------------------------------------------------------------------
# yfinance IO
# ---------------------------------------------------------------------------
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns single-ticker OHLCV either flat or as a (field, ticker)
    MultiIndex depending on version. Collapse to flat lower-case OHLCV columns."""
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the level that holds Open/High/Low/Close/Volume.
        lvl0 = set(df.columns.get_level_values(0))
        keep = 0 if {"Open", "Close"} & lvl0 else 1
        df = df.copy()
        df.columns = df.columns.get_level_values(keep)
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]]


def _download(interval: str, *, period: str | None = None,
              start: str | None = None, end: str | None = None,
              retries: int = 3) -> pd.DataFrame:
    """Download OHLCV via yfinance with a couple of polite retries (Yahoo rate-limits)."""
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(config.TSLA_TICKER, interval=interval, period=period,
                              start=start, end=end, auto_adjust=True, progress=False,
                              prepost=False)
            if raw is not None and not raw.empty:
                return _flatten(raw)
            last_err = RuntimeError("empty frame")
        except Exception as e:  # noqa: BLE001 — surface after retries
            last_err = e
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(
        f"yfinance download failed for TSLA interval={interval} "
        f"(period={period}, start={start}, end={end}): {last_err}. "
        "Yahoo may be rate-limiting — retry in a minute.")


# ---------------------------------------------------------------------------
# Daily
# ---------------------------------------------------------------------------
def fetch_daily(force: bool = False) -> pd.DataFrame:
    """Daily OHLCV (tz-naive `date`), cached. The backbone for HAR / range-based
    features and the close-to-close RV target used for the apples-to-apples
    comparison against the daily multi-stock run."""
    cache = config.TSLA_DAILY_CACHE
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    df = _download("1d", start=config.TSLA_DAILY_START).reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
    df = df.reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"Cached TSLA daily -> {cache} ({len(df)} rows, "
          f"{df['date'].min().date()}..{df['date'].max().date()})")
    return df


# ---------------------------------------------------------------------------
# Intraday
# ---------------------------------------------------------------------------
def fetch_intraday(interval: str, force: bool = False) -> pd.DataFrame:
    """Intraday OHLCV for one interval, cached. `datetime` is tz-aware
    (America/New_York); `session` is the trading date the bar belongs to."""
    cache = config.TSLA_INTRADAY_CACHE[interval]
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    period = config.TSLA_INTRADAY_WINDOWS[interval]
    df = _download(interval, period=period).reset_index()
    df = df.rename(columns={df.columns[0]: "datetime"})
    dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(NY_TZ)
    df["datetime"] = dt
    df["session"] = dt.dt.normalize().dt.tz_localize(None)
    # Regular trading hours only (defensive; prepost=False already trims most).
    minutes = dt.dt.hour * 60 + dt.dt.minute
    df = df[(minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)]
    df = df.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"Cached TSLA {interval} intraday -> {cache} ({len(df)} rows, "
          f"{df['session'].min().date()}..{df['session'].max().date()})")
    return df


def realized_vol_from_intraday(intraday: pd.DataFrame) -> pd.DataFrame:
    """Per-session realized volatility from within-day bar returns.

    Realized variance = sum of squared within-session log returns; realized vol =
    sqrt of that. This is a *daily* volatility number (same units as a daily
    |return|), but estimated from many intraday moves, so it is much less noisy
    than a single close-to-close squared return. Also returns the day's intraday
    log-range (high/low across the session) and the bar count (data-quality flag).
    """
    df = intraday.sort_values(["session", "datetime"]).copy()
    df["logret"] = np.log(df["close"]).groupby(df["session"]).diff()
    df["r2"] = df["logret"] ** 2
    g = df.groupby("session")
    out = pd.DataFrame({
        "iv_rvar": g["r2"].sum(),                                   # realized variance
        "iv_high": g["high"].max(),
        "iv_low": g["low"].min(),
        "n_bars": g["close"].count(),
    })
    out["iv_rv"] = np.sqrt(out["iv_rvar"])                          # realized vol (daily)
    out["iv_range"] = np.log(out["iv_high"] / out["iv_low"])       # intraday log-range
    out = out.reset_index().rename(columns={"session": "date"})
    out["date"] = pd.to_datetime(out["date"])
    # Drop half-day/thin sessions that would give an unreliable RV.
    out = out[out["n_bars"] >= 3].reset_index(drop=True)
    return out[["date", "iv_rv", "iv_rvar", "iv_range", "n_bars"]]


def build_intraday_daily(force: bool = False) -> pd.DataFrame:
    """The headline product: a per-session table of intraday-derived features
    (realized vol, intraday range) from the ~2-year hourly bars, plus the
    overnight gap from daily bars. Merged onto daily dates in tesla/features.py."""
    intr = fetch_intraday(config.TSLA_HEADLINE_INTRADAY, force=force)
    rv = realized_vol_from_intraday(intr)
    return rv


# ---------------------------------------------------------------------------
# Validation: does hourly RV track finer-resolution RV?
# ---------------------------------------------------------------------------
def _rv_series(interval: str, force: bool = False) -> pd.DataFrame:
    return realized_vol_from_intraday(fetch_intraday(interval, force=force))[
        ["date", "iv_rv"]].rename(columns={"iv_rv": f"rv_{interval}"})


def validate_rv_resolution(force: bool = False) -> dict[str, dict]:
    """Compare per-session realized vol across resolutions, PAIRWISE, so each
    comparison uses its full overlap (a triple inner-join collapses to the tiny
    1-minute window and is uninformative).

    High correlation => the hourly RV is a faithful (if downward-biased) stand-in,
    so the ~2-year hourly layer is a legitimate signal. The 1h-vs-5m pair spans
    ~60 days (many sessions) and is the credible check; 5m-vs-1m spans ~7 days.
    """
    rvs: dict[str, pd.DataFrame] = {}
    for iv in ("1h", "5m", "1m"):
        try:
            rvs[iv] = _rv_series(iv, force=force)
        except Exception as e:  # noqa: BLE001
            print(f"  (validation) {iv} unavailable: {e}")

    results: dict[str, dict] = {}
    for a, b in (("1h", "5m"), ("5m", "1m"), ("1h", "1m")):
        if a in rvs and b in rvs:
            m = rvs[a].merge(rvs[b], on="date", how="inner").dropna()
            if len(m) >= 3:
                corr = float(m[f"rv_{a}"].corr(m[f"rv_{b}"]))
                ratio = float((m[f"rv_{a}"] / m[f"rv_{b}"]).replace(
                    [np.inf, -np.inf], np.nan).dropna().mean())
                results[f"{a}_vs_{b}"] = {"n": len(m), "corr": corr, "mean_ratio": ratio}
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download everything")
    args = ap.parse_args()

    daily = fetch_daily(force=args.force)
    print(f"Daily: {len(daily)} rows, {daily['date'].min().date()}..{daily['date'].max().date()}")

    iv_daily = build_intraday_daily(force=args.force)
    print(f"Intraday->daily RV ({config.TSLA_HEADLINE_INTRADAY}): {len(iv_daily)} sessions, "
          f"{iv_daily['date'].min().date()}..{iv_daily['date'].max().date()}, "
          f"median bars/session={int(iv_daily['n_bars'].median())}")

    print("\n=== RV resolution check (pairwise, over each pair's full overlap) ===")
    print("  corr ~1 => coarser bars track finer RV; mean_ratio<1 => coarser under-measures")
    res = validate_rv_resolution(force=args.force)
    if res:
        for pair, r in res.items():
            print(f"  {pair:>9}:  n={r['n']:>3}  corr={r['corr']:+.3f}  "
                  f"mean_ratio={r['mean_ratio']:.3f}")
    else:
        print("  no overlapping intraday windows available.")


if __name__ == "__main__":
    main()
