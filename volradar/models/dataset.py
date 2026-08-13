"""Splitting, embargoing and target construction — the honesty layer.

Every number this project reports depends on these three functions being right, so they
live alone in one small module with dedicated tests.

The embargo matters more than it looks. A horizon-10 label spans days t+1..t+10, so a row
dated 2017-12-28 has an outcome that reaches into January 2018. Training on it while
testing on 2018 leaks the answer across the boundary. `embargoed_train` drops any row
whose outcome window crosses the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config


def labelled(df: pd.DataFrame, features: list[str], h: int) -> pd.DataFrame:
    """Rows with every feature present and the horizon-h target known."""
    need = list(features) + [config.fwd_col(h), config.persist_col(h)]
    return df.dropna(subset=need).reset_index(drop=True)


def embargoed_train(df: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows strictly before `cutoff` whose h-day outcome ALSO lands before it."""
    ld = config.label_date_col(h)
    return df[(df["date"] < cutoff) & df[ld].notna() & (df[ld] < cutoff)]


def time_slice(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] < pd.Timestamp(end)]
    return out


def xy(df: pd.DataFrame, features: list[str], h: int) -> tuple[np.ndarray, np.ndarray]:
    return df[features].to_numpy(float), df[config.fwd_col(h)].to_numpy(float)


def fold_edges(years: list[int], per_year: int = 1) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Walk-forward fold boundaries. `per_year=2` splits each year in half.

    Splitting folds more finely is a legitimate way to gain statistical power for a paired
    comparison: it adds independent evaluation points without touching the test period or
    changing what any model is trained on before its own cutoff. It is not a second bite at
    the apple — the same data is simply scored in smaller, still-disjoint blocks.

    Every selection step in the project reports at both granularities and adopts only when
    they agree, so a result that depends on how the folds were cut cannot be shipped.
    """
    out = []
    for y in years:
        for k in range(per_year):
            m0 = 1 + (12 // per_year) * k
            start = pd.Timestamp(f"{y}-{m0:02d}-01")
            out.append((start, start + pd.DateOffset(months=12 // per_year)))
    return out


def regime_thresholds(train_df: pd.DataFrame, h: int) -> tuple[float, float]:
    """Calm/normal/stressed cut points, taken from the TRAINING period only.

    Freezing them on train is what makes the regime labels honest: if they were computed
    on the full sample, a quiet decade would silently redefine what "stressed" means and
    the test-period labels would carry information from the future.
    """
    y = train_df[config.fwd_col(h)].dropna()
    lo, hi = config.REGIME_QUANTILES
    return float(y.quantile(lo)), float(y.quantile(hi))


def to_regime(values, thresholds: tuple[float, float]) -> np.ndarray:
    """Map log-RV values to 0=calm, 1=normal, 2=stressed."""
    lo, hi = thresholds
    v = np.asarray(values, dtype=float)
    return np.where(v <= lo, 0, np.where(v <= hi, 1, 2))
