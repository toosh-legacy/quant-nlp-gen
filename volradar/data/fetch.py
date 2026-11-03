"""Download every raw series the pipeline needs, and cache it.

One function per source family, each idempotent: if the parquet cache exists it is
returned untouched, so `build` and `train` never hit the network. Pass `force=True` (or
`--force` on the CLI) to refresh through the latest available close.

Prices and macro series come from Yahoo Finance via yfinance. The implied-volatility
complex comes from Cboe's own published daily-history CSVs, with Yahoo as a fallback. Both
are free and need no API key.
"""

from __future__ import annotations

import io
import urllib.request

import pandas as pd

from volradar import config


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for `tickers` and return a long (date, ticker, o/h/l/c/v) frame.

    yfinance returns a column-MultiIndex (field, ticker) for multi-ticker requests and a
    flat frame for a single ticker; normalising to long format here keeps every downstream
    feature builder identical regardless of how many series it asked for.
    """
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError(f"Download returned nothing for {tickers} "
                           "(Yahoo may be rate-limiting — retry in a minute).")

    fields = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(raw.columns, pd.MultiIndex):
        parts = {f.lower(): raw[f].stack() for f in fields if f in raw.columns.levels[0]}
        long = pd.concat(parts, axis=1).reset_index()
        long.columns = ["date", "ticker"] + list(parts.keys())
    else:
        long = raw[[f for f in fields if f in raw.columns]].copy()
        long.columns = [c.lower() for c in long.columns]
        long = long.reset_index().rename(columns={"Date": "date"})
        long["ticker"] = tickers[0] if isinstance(tickers, list) else tickers

    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    for c in keep:
        if c not in long.columns:
            long[c] = pd.NA
    return (long[keep]
            .dropna(subset=["close"])
            .sort_values(["ticker", "date"])
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Cboe volatility indices
# ---------------------------------------------------------------------------
# Cboe publishes each index's full daily history as a plain CSV, no key and no rate limit.
# It is worth preferring over Yahoo for three reasons: it is the *primary* source rather
# than a redistribution of it, it carries the whole term structure (Yahoo's coverage of
# VIX9D and VIX6M is patchy), and it gives OHLC rather than close alone — which is what
# makes a range-based vol-of-vol measure possible at all.
CBOE_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
CBOE_TIMEOUT = 30


def _parse_cboe_csv(text: str, ticker: str) -> pd.DataFrame:
    """One Cboe history CSV -> the long OHLCV shape the rest of the pipeline expects.

    Two layouts exist: `DATE,OPEN,HIGH,LOW,CLOSE` for the index families that trade
    intraday, and `DATE,<SYMBOL>` for the ones published as a single daily print (VVIX,
    SKEW). The second is widened to OHLC by repeating the close, so downstream code never
    has to ask which kind it is holding — a range estimator on a flat bar is simply zero.
    """
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    date_col = raw.columns[0]
    out = pd.DataFrame({"date": pd.to_datetime(raw[date_col], errors="coerce")})

    if {"open", "high", "low", "close"} <= set(raw.columns):
        for f in ("open", "high", "low", "close"):
            out[f] = pd.to_numeric(raw[f], errors="coerce")
    else:
        close = pd.to_numeric(raw[raw.columns[1]], errors="coerce")
        for f in ("open", "high", "low", "close"):
            out[f] = close

    out["ticker"] = ticker
    out["volume"] = pd.NA          # an index has no volume; kept so the schema matches
    return (out[["date", "ticker", "open", "high", "low", "close", "volume"]]
            .dropna(subset=["date", "close"])
            .sort_values("date"))


def _download_cboe(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download the Cboe volatility indices named by their Yahoo-style tickers."""
    parts = []
    for t in tickers:
        symbol = t.lstrip("^").upper()
        url = CBOE_CSV.format(symbol=symbol)
        with urllib.request.urlopen(url, timeout=CBOE_TIMEOUT) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        parts.append(_parse_cboe_csv(text, t))
    if not parts:
        raise RuntimeError("no Cboe symbols requested")
    long = pd.concat(parts, ignore_index=True)
    long = long[(long["date"] >= pd.Timestamp(start)) & (long["date"] < pd.Timestamp(end))]
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def _download_vol_indices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Cboe first, Yahoo second. A dead upstream should degrade the run, not end it."""
    try:
        return _download_cboe(tickers, start, end)
    except Exception as exc:
        print(f"  [warn] Cboe download failed ({exc}) — falling back to Yahoo")
        return _download(tickers, start, end)


# A series that returns far fewer rows than its peers has been silently truncated by the
# provider. This happens: a single Yahoo request once returned 13 rows for ^TNX instead of
# ~5,400, which nulled two macro features and silently dropped every macro row from the
# panel. Nothing raised. Guarding against it is not optional for a pipeline whose output is
# a forecast — a quietly-truncated input produces a confident, wrong number.
MIN_COVERAGE_RATIO = 0.5


def _check_coverage(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    """Return tickers whose history is missing or implausibly short versus the best series."""
    counts = df.groupby("ticker").size()
    best = counts.max() if len(counts) else 0
    bad = [t for t in tickers if t not in counts.index
           or counts[t] < best * MIN_COVERAGE_RATIO]
    return bad


def _cached(name: str, tickers: list[str], force: bool, downloader=None) -> pd.DataFrame:
    """Fetch-or-load helper shared by every source family, with a coverage guard.

    `downloader` lets a family come from somewhere other than Yahoo while still getting the
    cache, the coverage guard and the individual-retry behaviour — the guard is the part
    that must never be bypassed, whatever the source.
    """
    config.ensure_dirs()
    dl = downloader or _download
    path = config.RAW_DIR / f"{name}.parquet"
    if path.exists() and not force:
        cached = pd.read_parquet(path)
        missing = [t for t in tickers if t not in set(cached["ticker"].unique())]
        if not missing:
            return cached
        print(f"  {name}: cache is missing {missing} — refetching")
    print(f"  downloading {name}: {len(tickers)} series "
          f"{config.HISTORY_START}..{config.HISTORY_END}")
    df = dl(tickers, config.HISTORY_START, config.HISTORY_END)

    # Re-request anything that came back short, individually — batch requests are where
    # partial responses happen, and a single-ticker retry usually returns the full series.
    bad = _check_coverage(df, tickers)
    if bad:
        print(f"  [warn] short history for {bad} — retrying individually")
        for t in bad:
            try:
                one = dl([t], config.HISTORY_START, config.HISTORY_END)
                if len(one) > len(df[df["ticker"] == t]):
                    df = pd.concat([df[df["ticker"] != t], one], ignore_index=True)
                    print(f"  [ok] {t}: recovered {len(one):,} rows")
            except Exception as exc:
                print(f"  [warn] retry failed for {t}: {exc}")
        still_bad = _check_coverage(df, tickers)
        if still_bad:
            print(f"  [WARN] {still_bad} remain short. Features derived from them will be "
                  "mostly missing; check before trusting any result that uses them.")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(path, index=False)
    print(f"  cached -> {path.name} ({len(df):,} rows, {df['ticker'].nunique()} series)")
    return df


def fetch_market(force: bool = False) -> pd.DataFrame:
    """SPY daily OHLCV — the market-level forecast target."""
    return _cached("market", [config.MARKET_TICKER], force)


def fetch_sectors(force: bool = False) -> pd.DataFrame:
    """The eleven SPDR sector ETFs — the sector-level forecast targets."""
    return _cached("sectors", list(config.SECTOR_ETFS), force)


def fetch_vix(force: bool = False) -> pd.DataFrame:
    """The Cboe implied-volatility complex — term structure, vol-of-vol, skew, cross-asset.

    Index quotes have no volume. They do have OHLC, which the term-structure block uses for
    a range-based vol-of-vol measure that a close-only series cannot produce.
    """
    return _cached("vix", config.VIX_TICKERS, force, downloader=_download_vol_indices)


def fetch_macro(force: bool = False) -> pd.DataFrame:
    """Rates, credit and dollar proxies."""
    return _cached("macro", config.MACRO_TICKERS, force)


def fetch_breadth_universe(force: bool = False) -> pd.DataFrame:
    """The ~80-stock large-cap panel used for cross-sectional dispersion and breadth.

    Always downloaded rather than adopted from an older cache: the breadth features feed
    the *current* forecast, so a universe that stops short of the latest close would
    silently produce stale dispersion and correlation on today's row.
    """
    return _cached("breadth", config.BREADTH_UNIVERSE, force)


def fetch_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch every source family. Returns them keyed by name for the build step."""
    print("Fetching raw series...")
    out = {
        "market": fetch_market(force),
        "sectors": fetch_sectors(force),
        "vix": fetch_vix(force),
        "macro": fetch_macro(force),
        "breadth": fetch_breadth_universe(force),
    }
    for k, v in out.items():
        span = f"{v['date'].min().date()}..{v['date'].max().date()}"
        print(f"  {k:<9} {len(v):>8,} rows  {v['ticker'].nunique():>3} series  {span}")
    return out


if __name__ == "__main__":
    fetch_all()
