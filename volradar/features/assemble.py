"""Join the raw series into the two model-ready panels.

Two panels, two questions:

* **market** — one row per trading day, target = SPY's log realized vol over the next h
  days. Answers "is market-wide volatility about to rise or fall?"
* **sector** — one row per (day, sector ETF), target = that sector's log realized vol.
  Answers "which sectors are stable and which are fragile?"

Both are written to `data/build/` as parquet so training never re-reads raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config
from volradar.data import fetch
from volradar.features import factors
from volradar.features.core import CORE_FEATURES, build_core

# Sector-panel-only columns: where this sector sits within the sector cross-section that
# day. The earlier study found the cross-sectional block was what lifted R^2 from ~0.2 to
# ~0.4 — "which series is the volatile one" is a large part of the answer.
SECTOR_REL_FEATURES = ["sec_rv_rel", "sec_rv_rank", "sec_beta_63"]

MARKET_PANEL = config.BUILD_DIR / "panel_market.parquet"
SECTOR_PANEL = config.BUILD_DIR / "panel_sector.parquet"
POOLED_PANEL = config.BUILD_DIR / "panel_pooled.parquet"


def build_pooled_panel(force: bool = False) -> pd.DataFrame:
    """Every series — 79 single stocks, 11 sector ETFs and SPY — in one long panel.

    The motivation is a sample-size problem the two-panel design cannot fix. The market
    model has ~5,400 daily rows and roughly twenty features; that is thin enough that a
    gradient-boosted model is fitting noise as much as signal, which is exactly what the
    unstable per-year results suggested.

    But volatility dynamics are largely *shared* across equities: clustering, mean
    reversion, the leverage effect and the range/close-to-close relationship all behave
    alike whether the series is Apple, the energy sector or the index. So a model can be
    fitted on the pooled cross-section — roughly 470,000 rows — and applied to the market,
    provided every feature is scale-free. They are: each one is a log ratio, a log level of
    a return-based quantity, a rank, or a z-score, so a $400 index and a $30 stock produce
    comparable inputs.

    A `series_type` column is carried so the model can still distinguish an index from a
    single name, since diversification makes indices genuinely less volatile.
    """
    config.ensure_dirs()
    if POOLED_PANEL.exists() and not force:
        return pd.read_parquet(POOLED_PANEL)

    raw = fetch.fetch_all()
    parts = []
    for name, key in [("stock", "breadth"), ("sector", "sectors"), ("index", "market")]:
        core = build_core(raw[key])
        core["series_type"] = name
        parts.append(core)

    pooled = pd.concat(parts, ignore_index=True)
    # Numeric encoding for the model; ordered by breadth of diversification.
    pooled["is_index"] = (pooled["series_type"] == "index").astype(int)
    pooled["is_sector"] = (pooled["series_type"] == "sector").astype(int)
    pooled = pooled.sort_values(["ticker", "date"]).reset_index(drop=True)
    pooled.to_parquet(POOLED_PANEL, index=False)
    print(f"  pooled panel -> {POOLED_PANEL.name} ({len(pooled):,} rows, "
          f"{pooled['ticker'].nunique()} series)")
    return pooled


def _merge_date_blocks(base: pd.DataFrame, blocks: list[pd.DataFrame]) -> pd.DataFrame:
    """Left-join date-level factor blocks onto a panel keyed by date (+ ticker)."""
    out = base
    for b in blocks:
        out = out.merge(b, on="date", how="left")
    return out


def build_panels(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (market, sector) panels from cached raw data."""
    config.ensure_dirs()
    if MARKET_PANEL.exists() and SECTOR_PANEL.exists() and not force:
        return pd.read_parquet(MARKET_PANEL), pd.read_parquet(SECTOR_PANEL)

    raw = fetch.fetch_all()

    print("Building feature panels...")
    market_core = build_core(raw["market"])
    sector_core = build_core(raw["sectors"])

    # --- date-level factor families ---------------------------------------------------
    vix_block = factors.build_vix(raw["vix"], market_core)
    breadth_block = factors.build_breadth(raw["breadth"])
    macro_block = factors.build_macro(raw["macro"])
    sector_xs_block = factors.build_sector_xs(sector_core)
    blocks = [vix_block, breadth_block, macro_block, sector_xs_block]

    # --- market panel -----------------------------------------------------------------
    market = _merge_date_blocks(market_core.drop(columns=["ticker"]), blocks)
    market = market.sort_values("date").reset_index(drop=True)

    # --- sector panel: add each sector's position within the sector cross-section ------
    sec = sector_core.copy()
    rv = sec["log_rv_m"]
    sec["sec_rv_rel"] = rv - sec.groupby("date")["log_rv_m"].transform("mean")
    sec["sec_rv_rank"] = sec.groupby("date")["log_rv_m"].rank(pct=True)

    # Rolling 63-day beta of each sector to SPY: high-beta sectors inherit market vol.
    spy_ret = market_core.set_index("date")["ret"]
    sec["_mkt_ret"] = sec["date"].map(spy_ret)
    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["ret"].rolling(63).cov(g["_mkt_ret"])
        var = g["_mkt_ret"].rolling(63).var()
        return cov / var.replace(0, np.nan)
    sec["sec_beta_63"] = (sec.sort_values(["ticker", "date"])
                             .groupby("ticker", group_keys=False)
                             .apply(_beta, include_groups=False)
                             .reindex(sec.index))
    sec = sec.drop(columns=["_mkt_ret"])

    sector = _merge_date_blocks(sec, blocks)
    sector["sector_name"] = sector["ticker"].map(config.SECTOR_ETFS)
    sector = sector.sort_values(["ticker", "date"]).reset_index(drop=True)

    market.to_parquet(MARKET_PANEL, index=False)
    sector.to_parquet(SECTOR_PANEL, index=False)
    print(f"  market panel -> {MARKET_PANEL.name} ({len(market):,} rows)")
    print(f"  sector panel -> {SECTOR_PANEL.name} ({len(sector):,} rows, "
          f"{sector['ticker'].nunique()} sectors)")
    return market, sector


def feature_families(panel: str) -> dict[str, list[str]]:
    """The factor families available for a panel, for ablation and for training.

    'core' is the series' own price/volatility history and is never ablated away — it is
    the baseline every other family has to improve on.
    """
    fams = {
        "core": list(CORE_FEATURES),
        "vix": list(factors.VIX_FEATURES),
        "breadth": list(factors.BREADTH_FEATURES),
        "macro": list(factors.MACRO_FEATURES),
        "sector_xs": list(factors.SECTOR_XS_FEATURES),
    }
    if panel == "sector":
        fams["sector_rel"] = list(SECTOR_REL_FEATURES)
    return fams


def all_features(panel: str) -> list[str]:
    """The features the SHIPPED model uses: core plus whatever selection kept.

    `feature_families` still exposes every implemented family — the ablation and the
    selection study need them all — but training, serving and the headline evaluation go
    through here, so the reported numbers and the live forecast always use the same inputs.
    """
    fams = feature_families(panel)
    kept = config.SELECTED_FAMILIES.get(panel, [])
    feats = list(fams["core"]) + [c for k in kept for c in fams[k]]
    if config.TRAINING_SOURCE.get(panel) == "pooled":
        feats += config.POOL_TAGS
    return feats


def load_panel(panel: str) -> pd.DataFrame:
    """The frame a panel is EVALUATED and SERVED on.

    When a panel trains on the pooled cross-section, its evaluation rows are taken from the
    pooled frame too, so that training and serving see identical columns. Anything else
    invites the classic train/serve skew where a model is fitted on one feature layout and
    called with another.
    """
    if config.TRAINING_SOURCE.get(panel) == "pooled":
        pooled = build_pooled_panel()
        if panel == "market":
            out = pooled[pooled["ticker"] == config.MARKET_TICKER].copy()
        else:
            out = pooled[pooled["ticker"].isin(config.SECTOR_ETFS)].copy()
            out["sector_name"] = out["ticker"].map(config.SECTOR_ETFS)
        return out.reset_index(drop=True)

    path = MARKET_PANEL if panel == "market" else SECTOR_PANEL
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m volradar.cli build` first.")
    return pd.read_parquet(path)


def load_training_panel(panel: str) -> pd.DataFrame:
    """The frame a panel's model is FITTED on — the whole cross-section when pooled."""
    if config.TRAINING_SOURCE.get(panel) == "pooled":
        return build_pooled_panel()
    return load_panel(panel)


def fit_and_eval_frames(panel: str, features: list[str], h: int):
    """(training rows, evaluation rows) for a panel — the only place the two can differ.

    Returning both from one function keeps the distinction explicit at every call site.
    When a panel trains on its own history the two frames are the same object's rows, so
    nothing changes; when it trains on the pooled cross-section, the split is visible in
    the caller rather than hidden inside a loader.
    """
    from volradar.models import dataset as D

    eval_df = D.labelled(load_panel(panel), features, h)
    if config.TRAINING_SOURCE.get(panel) != "pooled":
        return eval_df, eval_df
    return D.labelled(load_training_panel(panel), features, h), eval_df


if __name__ == "__main__":
    build_panels(force=True)
