"""The Oxford-Man Realized Library — intraday-derived volatility measures, from one file.

**Why this exists.** Everything else in this project is built from daily OHLC bars, so its
volatility measures are estimated from one return (or one high-low range) per day. The
literature's strongest results — HARQ's measurement-error correction, and much of the
reported accuracy of HAR itself — assume realized variance computed from *intraday* returns,
which is a far more precise estimate of the same quantity. Whether that precision is worth
paying for is the biggest open question left in this project, and it cannot be answered with
daily data.

This library answers it. It provides 5-minute and 10-minute realized variance, bipower
variation, realized semivariance, realized kernels and median-truncated RV for 21 equity
indices including the S&P 500, computed from tick data.

**Provenance and its limits, stated plainly.** The Oxford-Man Institute shut the library
down and its domain no longer resolves. This loader reads an archived copy committed to a
public GitHub repository, which is a 2016 vintage: coverage runs **2000-01-03 to
2016-09-26**. That is enough to answer the scientific question but *not* enough to serve
live forecasts, so this data feeds a scoped study (`backtest/intraday_study.py`) and never
the production pipeline. The study therefore uses its own pre-registered split, because the
project's usual 2018 test boundary lies past the end of this data.

One file, cached once. No API keys, no rate limits.
"""

from __future__ import annotations

import pandas as pd

from volradar import config

# Archived copy of the Oxford-Man Realized Library (the original host is gone).
SOURCE_URL = ("https://raw.githubusercontent.com/yools56/Neural-Network-based-HAR-models/"
              "master/OxfordManRealizedVolatilityIndices(20160928).xlsx")

CACHE = config.RAW_DIR / "realized_library.parquet"

# The index whose dynamics match our market panel.
SPX = "S&P 500 (Live)"

# The measures worth carrying, mapped to short names. Each is an estimate of the same
# latent daily variance, computed differently:
#   rv5/rv10          - realized variance from 5- and 10-minute returns
#   rv5_ss            - 5-minute sampled with 1-minute subsampling (less noisy still)
#   bpv5              - bipower variation: jump-robust, estimates the continuous part
#   rsv5              - realized semivariance (downside component)
#   rk                - realized kernel: noise-robust, the most efficient estimator here
#   medrv             - median truncated RV, also jump-robust
MEASURES = {
    "Realized Variance (5-minute)": "rv5",
    "Realized Variance (5-minute using 1-minute subsamples)": "rv5_ss",
    "Realized Variance (10-minute)": "rv10",
    "Bipower Variation (5-minute)": "bpv5",
    "Realized Semivariance (5-minute)": "rsv5",
    "Realized Kernel": "rk",
    "Median Truncated Realized Variance": "medrv",
    "Closing Price": "close",
    "Opening Price": "open",
    "High-Low Range": "hl_range",
    "Return": "ret_oxf",
    "Close-to-Open Return": "co_ret",
    "Number of Transactions": "n_trades",
}


def _parse(xlsx_path) -> pd.DataFrame:
    """Wide (date x [index, measure]) -> long (date, symbol, measure...) frame."""
    raw = pd.read_excel(xlsx_path, header=[0, 1], index_col=0)

    # The index is YYYYMMDD integers with a stray 'DateID' label row from the export.
    idx = pd.to_numeric(pd.Series(raw.index), errors="coerce")
    raw = raw[idx.notna().to_numpy()]
    raw.index = pd.to_datetime(idx.dropna().astype(int).astype(str), format="%Y%m%d")

    # Several measure names in the export carry trailing spaces ("Bipower Variation
    # (5-minute) "), which silently drops those columns on an exact-match lookup. Normalise
    # before matching rather than encoding the whitespace into the lookup table.
    raw.columns = pd.MultiIndex.from_tuples(
        [(str(a).strip(), str(b).strip()) for a, b in raw.columns])
    lookup = {k.strip(): v for k, v in MEASURES.items()}

    frames = []
    for symbol in sorted({c[0] for c in raw.columns}):
        block = raw[symbol]
        keep = {src: dst for src, dst in lookup.items() if src in block.columns}
        if "Closing Price" not in keep:
            continue
        part = block[list(keep)].rename(columns=keep).apply(pd.to_numeric, errors="coerce")
        part.insert(0, "symbol", symbol)
        frames.append(part.reset_index().rename(columns={"index": "date"}))

    long = pd.concat(frames, ignore_index=True)
    return long.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_realized_library(force: bool = False) -> pd.DataFrame:
    """Download once, cache forever. Returns a long frame of intraday measures."""
    config.ensure_dirs()
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    print(f"  downloading the Oxford-Man Realized Library (one file, ~16MB)...")
    df = _parse(SOURCE_URL)
    df.to_parquet(CACHE, index=False)
    span = f"{df['date'].min().date()}..{df['date'].max().date()}"
    print(f"  cached -> {CACHE.name} ({len(df):,} rows, "
          f"{df['symbol'].nunique()} indices, {span})")
    return df


def load_symbol(symbol: str = SPX, force: bool = False) -> pd.DataFrame:
    """One index, with the OHLC columns named as the rest of the pipeline expects.

    High and Low are reconstructed from the library's high-low range around the close: the
    library reports the range rather than the levels, and the range is what the
    range-based estimators actually consume.
    """
    df = load_realized_library(force)
    out = df[df["symbol"] == symbol].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{symbol!r} not in the library; have {sorted(df['symbol'].unique())}")

    out["ticker"] = "SPX" if symbol == SPX else symbol
    # Reconstruct plausible High/Low levels that respect the reported range and bracket
    # the day's open and close, so build_core's range estimators behave as on real bars.
    hi_lo = out["hl_range"].abs().fillna(0.0)
    mid = out[["open", "close"]].mean(axis=1)
    out["high"] = mid * (1 + hi_lo / 2)
    out["low"] = mid * (1 - hi_lo / 2)
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    out["volume"] = out["n_trades"].ffill().bfill()
    return out


if __name__ == "__main__":
    lib = load_realized_library(force=True)
    print(lib.groupby("symbol").size().sort_values(ascending=False).head(5).to_string())
    spx = load_symbol()
    print(f"\nSPX: {len(spx):,} rows {spx['date'].min().date()}..{spx['date'].max().date()}")
    print(spx[["date", "close", "rv5", "bpv5", "rk", "rsv5"]].tail(3).to_string(index=False))
