"""The four candidate factor families, as date-level (one row per trading day) blocks.

Each family is built independently and exposed under a name in `FAMILIES`, so the
ablation study in `volradar/backtest/ablation.py` can add or remove a whole family and
measure exactly what it was worth. Nothing here is assumed to help — the earlier study
found the VIX added nothing to *per-stock* volatility, and the honest test of whether it
helps at the *market* level is the ablation table, not intuition.

Every column is known at the close of day t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config

EPS = config.EPS


def _wide(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot a long frame to date x ticker for the given field."""
    return df.pivot_table(index="date", columns="ticker", values=field, aggfunc="last").sort_index()


# ---------------------------------------------------------------------------
# Family: vix — the implied-volatility complex
# ---------------------------------------------------------------------------
# Two families, split by how far back the underlying index goes — because a feature costs
# training history, and that cost has to be priced separately from its information content.
#
# Every column here is dropped by `dataset.labelled` on any row where it is missing, so
# adding a series that begins in 2011 does not merely add a column: it deletes every
# training row before 2011, including the 2008 crisis, which is the most informative
# volatility episode in the sample. Bundling the 9-day index in with the rest would have
# made the whole complex look bad for a reason that has nothing to do with what the option
# market knows.
#
# `vix` is bounded by VIX3M (from 2009-09). `vix_short` is bounded by VIX9D (from 2011-01)
# and is offered on its own so the selection study can decide whether the near end of the
# curve is worth two more years of lost history.
VIX_FEATURES = [
    # Level and momentum.
    "vix_log", "vix_chg_5", "vix_rel_21",
    # Term structure: the classic 3-month slope and the far end.
    "vix_term", "vix_term_long",
    # Uncertainty about the level itself, and the shape of the tail.
    "vvix_log", "vix_vol_pk", "skew_log",
    # Variance risk premium.
    "vix_rv_gap",
    # Cross-asset: is stress concentrated in large caps, or everywhere?
    "rvx_vix_spread", "vxn_vix_spread", "ovx_log",
]

# The near end of the term structure. Separate family: costs history back to 2011-01.
VIX_SHORT_FEATURES = ["vix_term_short", "vix_curve", "vix_rv_gap_short"]


def _log_col(w: pd.DataFrame, ticker: str) -> pd.Series:
    """log of a volatility index, or an all-NaN column if the series was not fetched."""
    if ticker in w:
        return np.log(w[ticker] + EPS)
    return pd.Series(np.nan, index=w.index)


def build_vix(vix_df: pd.DataFrame, market_core: pd.DataFrame) -> pd.DataFrame:
    """The implied-volatility complex: level, term structure, vol-of-vol, skew, premium.

    Three ideas beyond the level, each with its own literature:

    **Shape, not just height.** VIX9D / VIX / VIX3M / VIX6M is a term structure, and its
    slope carries information the level does not. In calm markets the curve is upward
    sloping (contango); when the short end rises above the long end (backwardation) the
    market is pricing a shock it expects to pass. A single level cannot say which of those
    two a reading of 20 is.

    **The variance risk premium.** `vix_rv_gap` is implied minus realized, in logs. It is
    the most theoretically motivated column here: when implied sits far above realized, the
    market is already pricing trouble that has not shown up in returns yet. It is computed
    at two horizons because the premium is horizon-specific — the 9-day index against
    5-day realized, and the 30-day index against 21-day realized. A forecast at h=5 and one
    at h=10 are not asking the option market the same question.

    **Vol-of-vol from the range.** `vix_vol_pk` is a Parkinson estimator on VIX's own daily
    high-low range: how much the market's fear gauge moved *within* the day. This is only
    available because the Cboe source carries OHLC; the previous close-only feed could
    express VVIX but not this.
    """
    w = _wide(vix_df)
    hi, lo = _wide(vix_df, "high"), _wide(vix_df, "low")
    out = pd.DataFrame(index=w.index)

    vix_log = _log_col(w, "^VIX")
    out["vix_log"] = vix_log
    out["vix_chg_5"] = vix_log - vix_log.shift(5)
    out["vix_rel_21"] = vix_log - vix_log.rolling(21).mean()

    # --- term structure. Positive = contango = calm; negative = backwardation = stress ---
    v9, v3m, v6m = (_log_col(w, "^VIX9D"), _log_col(w, "^VIX3M"), _log_col(w, "^VIX6M"))
    out["vix_term"] = v3m - vix_log                 # the classic 3-month/spot slope
    out["vix_term_short"] = vix_log - v9            # 30-day vs 9-day: the near end
    out["vix_term_long"] = v6m - v3m                # 6-month vs 3-month: the far end
    out["vix_curve"] = out["vix_term_long"] - out["vix_term_short"]

    # --- uncertainty about the level, and tail shape -----------------------------------
    out["vvix_log"] = _log_col(w, "^VVIX")
    out["skew_log"] = _log_col(w, "^SKEW")
    if "^VIX" in hi and "^VIX" in lo:
        rng = np.log((hi["^VIX"] + EPS) / (lo["^VIX"] + EPS))
        out["vix_vol_pk"] = np.log(np.sqrt((rng ** 2) / (4.0 * np.log(2.0))) + EPS)
    else:
        out["vix_vol_pk"] = np.nan

    # --- variance risk premium ----------------------------------------------------------
    # VIX is an annualized percentage; convert the market's realized log daily vol to the
    # same units before differencing so the gap means what it says.
    def _ann(col: str) -> pd.Series:
        rv = market_core.set_index("date")[col]
        return np.log(np.exp(rv) * np.sqrt(config.TRADING_DAYS) * 100.0 + EPS).reindex(out.index)

    out["vix_rv_gap"] = (vix_log - _ann("log_rv_m")).astype(float)
    out["vix_rv_gap_short"] = (v9 - _ann("log_rv_w")).astype(float)

    # --- cross-asset --------------------------------------------------------------------
    out["rvx_vix_spread"] = _log_col(w, "^RVX") - vix_log      # small caps vs large caps
    out["vxn_vix_spread"] = _log_col(w, "^VXN") - vix_log      # Nasdaq vs the broad market
    out["ovx_log"] = _log_col(w, "^OVX")                       # crude oil
    return out.reset_index()


# ---------------------------------------------------------------------------
# Family: breadth — cross-sectional dispersion and correlation
# ---------------------------------------------------------------------------
BREADTH_FEATURES = ["disp_log", "disp_chg_5", "avg_corr", "corr_chg_21",
                    "breadth_down", "xs_vol_skew"]


def build_breadth(breadth_df: pd.DataFrame) -> pd.DataFrame:
    """What the cross-section of stocks is doing, which index-level vol cannot see.

    Average pairwise correlation is the key column: index volatility is roughly average
    single-stock vol times correlation, so a correlation spike mechanically precedes an
    index vol spike even when individual names look calm. `xs_vol_skew` asks whether the
    already-volatile tail is pulling away from the median name.
    """
    px = _wide(breadth_df)
    rets = px.pct_change()

    out = pd.DataFrame(index=rets.index)
    # Cross-sectional dispersion of daily returns across the universe.
    out["disp_log"] = np.log(rets.std(axis=1) + EPS)
    out["disp_chg_5"] = out["disp_log"] - out["disp_log"].shift(5)

    # Average pairwise correlation over a rolling 21-day window. Computed from the mean of
    # the correlation matrix's off-diagonal, done cheaply via the variance identity:
    #   var(mean return) = (1/n^2) * [ sum var_i + sum_{i!=j} cov_ij ]
    # so avg_corr = (var_portfolio*n^2 - sum var_i) / (sum_{i!=j} sd_i*sd_j).
    win = 21
    sd = rets.rolling(win).std()
    port = rets.mean(axis=1).rolling(win).std()
    n = rets.notna().sum(axis=1)
    sum_var = (sd ** 2).sum(axis=1)
    sum_sd = sd.sum(axis=1)
    denom = (sum_sd ** 2 - sum_var).replace(0, np.nan)
    out["avg_corr"] = ((port ** 2) * (n ** 2) - sum_var) / denom
    out["avg_corr"] = out["avg_corr"].clip(-1, 1)
    out["corr_chg_21"] = out["avg_corr"] - out["avg_corr"].shift(21)

    # Breadth: share of the universe down on the day, smoothed over a week.
    out["breadth_down"] = (rets < 0).sum(axis=1).div(n.replace(0, np.nan)).rolling(5).mean()

    # Skew of the cross-section of 21-day single-stock vols (log scale).
    rv21 = np.log((rets ** 2).rolling(21).mean().pow(0.5) + EPS)
    out["xs_vol_skew"] = rv21.quantile(0.9, axis=1) - rv21.median(axis=1)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Family: macro — rates, credit, dollar
# ---------------------------------------------------------------------------
MACRO_FEATURES = ["yield_10y", "curve_slope", "credit_ratio_z", "credit_chg_21",
                  "dollar_chg_21"]


def build_macro(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Slow-moving regime context. Credit is the one with the strongest prior: high-yield
    underperforming investment grade is the classic early warning that risk appetite is
    turning, and it moves before equity vol does."""
    w = _wide(macro_df)
    out = pd.DataFrame(index=w.index)
    out["yield_10y"] = w["^TNX"] if "^TNX" in w else np.nan
    if "^TNX" in w and "^IRX" in w:
        out["curve_slope"] = w["^TNX"] - w["^IRX"]     # inversion (<0) leads stress
    else:
        out["curve_slope"] = np.nan

    if "HYG" in w and "LQD" in w:
        ratio = np.log((w["HYG"] + EPS) / (w["LQD"] + EPS))
        out["credit_ratio_z"] = ((ratio - ratio.rolling(252).mean())
                                 / ratio.rolling(252).std())
        out["credit_chg_21"] = ratio - ratio.shift(21)
    else:
        out["credit_ratio_z"] = out["credit_chg_21"] = np.nan

    out["dollar_chg_21"] = (np.log(w["UUP"] + EPS) - np.log(w["UUP"].shift(21) + EPS)
                            if "UUP" in w else np.nan)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Family: sector — what the sector cross-section is doing
# ---------------------------------------------------------------------------
SECTOR_XS_FEATURES = ["sec_disp_log", "sec_vol_spread", "sec_avg_rv"]


def build_sector_xs(sector_core: pd.DataFrame) -> pd.DataFrame:
    """Sector-level dispersion, computed from the sector ETFs' own realized vols.

    Wide dispersion across sectors means the market is rotating rather than selling off
    wholesale; compression means everything is moving together, which is the shape a
    market-wide vol event takes.
    """
    rv = sector_core.pivot_table(index="date", columns="ticker",
                                 values="log_rv_m", aggfunc="last").sort_index()
    ret = sector_core.pivot_table(index="date", columns="ticker",
                                  values="ret", aggfunc="last").sort_index()
    out = pd.DataFrame(index=rv.index)
    out["sec_disp_log"] = np.log(ret.std(axis=1) + EPS)
    out["sec_vol_spread"] = rv.max(axis=1) - rv.min(axis=1)
    out["sec_avg_rv"] = rv.mean(axis=1)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Registry — used by the ablation study
# ---------------------------------------------------------------------------
FAMILIES: dict[str, list[str]] = {
    "core": [],            # filled in by assemble (the own-history block)
    "vix": VIX_FEATURES,
    "vix_short": VIX_SHORT_FEATURES,
    "breadth": BREADTH_FEATURES,
    "macro": MACRO_FEATURES,
    "sector_xs": SECTOR_XS_FEATURES,
}
