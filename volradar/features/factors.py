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
VIX_FEATURES = ["vix_log", "vix_chg_5", "vix_rel_21", "vix_term", "vvix_log", "vix_rv_gap"]


def build_vix(vix_df: pd.DataFrame, market_core: pd.DataFrame) -> pd.DataFrame:
    """Implied vol level, momentum, term structure, vol-of-vol, and the variance premium.

    `vix_rv_gap` is the variance risk premium in logs: implied minus realized. It is the
    single most theoretically-motivated column here — when implied sits far above realized
    the market is already pricing trouble that has not shown up in returns yet.
    """
    w = _wide(vix_df)
    out = pd.DataFrame(index=w.index)
    vix = w["^VIX"]
    out["vix_log"] = np.log(vix + EPS)
    out["vix_chg_5"] = out["vix_log"] - out["vix_log"].shift(5)
    out["vix_rel_21"] = out["vix_log"] - out["vix_log"].rolling(21).mean()
    # Term structure: >0 (contango) is calm, <0 (backwardation) is stress.
    out["vix_term"] = np.log(w["^VIX3M"] + EPS) - out["vix_log"] if "^VIX3M" in w else np.nan
    out["vvix_log"] = np.log(w["^VVIX"] + EPS) if "^VVIX" in w else np.nan

    # Variance risk premium: VIX is an annualized % ; convert the market's realized log
    # daily vol to the same units before differencing so the gap is meaningful.
    rv = market_core.set_index("date")["log_rv_m"]
    rv_ann = np.log(np.exp(rv) * np.sqrt(config.TRADING_DAYS) * 100.0 + EPS)
    out["vix_rv_gap"] = (out["vix_log"] - rv_ann.reindex(out.index)).astype(float)
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
    "breadth": BREADTH_FEATURES,
    "macro": MACRO_FEATURES,
    "sector_xs": SECTOR_XS_FEATURES,
}
