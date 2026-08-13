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
from volradar.features.core import (CORE_FEATURES, JUMP_FEATURES, RANGE_EXT_FEATURES,
                                    SEMIVAR_FEATURES, build_core)

# Sector-panel-only columns: where this sector sits within the sector cross-section that
# day. The earlier study found the cross-sectional block was what lifted R^2 from ~0.2 to
# ~0.4 — "which series is the volatile one" is a large part of the answer.
SECTOR_REL_FEATURES = ["sec_rv_rel", "sec_rv_rank", "sec_beta_63"]

MARKET_PANEL = config.BUILD_DIR / "panel_market.parquet"
SECTOR_PANEL = config.BUILD_DIR / "panel_sector.parquet"
POOLED_PANEL = config.BUILD_DIR / "panel_pooled.parquet"

# `sector_rel` is the one family that is not defined for every row: a single stock has no
# position within the *sector* cross-section. Anything listed here narrows the pooled
# training set to the sector ETFs when it is selected — see `training_frame`.
SECTOR_ONLY_FAMILIES = {"sector_rel"}


def _merge_date_blocks(base: pd.DataFrame, blocks: list[pd.DataFrame]) -> pd.DataFrame:
    """Left-join date-level factor blocks onto a panel keyed by date (+ ticker)."""
    out = base
    for b in blocks:
        out = out.merge(b, on="date", how="left")
    return out


# Building the raw series and the per-series core blocks is the expensive part, and both
# panel builders need the same pieces. Caching them for the life of the process keeps
# `build` from doing all of it twice.
_SOURCES: dict | None = None
_POOLED: pd.DataFrame | None = None


def _sources(force: bool = False) -> dict:
    """Raw series, per-family core blocks, and the date-level factor blocks — built once.

    The date-level blocks are the four factor families. They are keyed by date alone, so
    they merge onto *any* panel — market, sector or pooled — which is what lets the pooled
    panel carry the same candidate features as the other two.
    """
    global _SOURCES
    if _SOURCES is not None and not force:
        return _SOURCES

    raw = fetch.fetch_all()
    print("Building feature blocks...")
    market_core = build_core(raw["market"])
    sector_core = _with_sector_rel(build_core(raw["sectors"]), market_core)

    _SOURCES = {
        "raw": raw,
        "market_core": market_core,
        "sector_core": sector_core,
        "date_blocks": [
            factors.build_vix(raw["vix"], market_core),
            factors.build_breadth(raw["breadth"]),
            factors.build_macro(raw["macro"]),
            factors.build_sector_xs(sector_core),
        ],
    }
    return _SOURCES


def _with_sector_rel(sector_core: pd.DataFrame, market_core: pd.DataFrame) -> pd.DataFrame:
    """Where each sector sits within the sector cross-section on that day."""
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
    return sec.drop(columns=["_mkt_ret"])


def _candidate_columns(panel: str = "sector") -> list[str]:
    """Every column any factor family can ask for — what a panel must carry to be testable."""
    fams = feature_families(panel)
    return [c for k, v in fams.items() if k != "core" for c in v]


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

    The factor families are merged in here too. They are not part of the shipped feature
    set, but this frame is what `load_panel` returns for any panel that trains pooled, so a
    family missing here is a family the ablation and the selection study cannot test at all.
    """
    global _POOLED
    config.ensure_dirs()
    if _POOLED is not None and not force:
        return _POOLED
    if POOLED_PANEL.exists() and not force:
        cached = pd.read_parquet(POOLED_PANEL)
        missing = [c for c in _candidate_columns() if c not in cached.columns]
        if not missing:
            # Held in memory for the rest of the process. The selection and ablation studies
            # ask for this frame once per candidate feature set — dozens of times — and it is
            # a 140 MB parquet; re-reading it each time dominated their runtime.
            _POOLED = cached
            return cached
        print(f"  pooled panel is missing {len(missing)} candidate columns "
              f"({', '.join(missing[:4])}...) — rebuilding")

    s = _sources()
    stock_core = build_core(s["raw"]["breadth"])
    parts = []
    for name, core in [("stock", stock_core), ("sector", s["sector_core"]),
                       ("index", s["market_core"])]:
        core = core.copy()
        core["series_type"] = name
        parts.append(core)

    # Single stocks and the index have no `sector_rel` block, so concat leaves those three
    # columns NaN for them. That is correct and deliberate: `training_frame` narrows the
    # pooled rows when a sector-only family is in play, rather than letting a dropna quietly
    # delete 80% of the training set.
    pooled = pd.concat(parts, ignore_index=True)
    pooled = _merge_date_blocks(pooled, s["date_blocks"])

    # Numeric encoding for the model; ordered by breadth of diversification.
    pooled["is_index"] = (pooled["series_type"] == "index").astype(int)
    pooled["is_sector"] = (pooled["series_type"] == "sector").astype(int)
    pooled = pooled.sort_values(["ticker", "date"]).reset_index(drop=True)
    _POOLED = pooled
    pooled.to_parquet(POOLED_PANEL, index=False)
    print(f"  pooled panel -> {POOLED_PANEL.name} ({len(pooled):,} rows, "
          f"{pooled['ticker'].nunique()} series)")
    return pooled


def build_panels(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (market, sector) panels from cached raw data."""
    config.ensure_dirs()
    if MARKET_PANEL.exists() and SECTOR_PANEL.exists() and not force:
        return pd.read_parquet(MARKET_PANEL), pd.read_parquet(SECTOR_PANEL)

    s = _sources()
    blocks = s["date_blocks"]

    print("Building feature panels...")
    # --- market panel -----------------------------------------------------------------
    market = _merge_date_blocks(s["market_core"].drop(columns=["ticker"]), blocks)
    market = market.sort_values("date").reset_index(drop=True)

    # --- sector panel ------------------------------------------------------------------
    sector = _merge_date_blocks(s["sector_core"], blocks)
    sector["sector_name"] = sector["ticker"].map(config.SECTOR_ETFS)
    sector = sector.sort_values(["ticker", "date"]).reset_index(drop=True)

    market.to_parquet(MARKET_PANEL, index=False)
    sector.to_parquet(SECTOR_PANEL, index=False)
    print(f"  market panel -> {MARKET_PANEL.name} ({len(market):,} rows)")
    print(f"  sector panel -> {SECTOR_PANEL.name} ({len(sector):,} rows, "
          f"{sector['ticker'].nunique()} sectors)")

    # Keep the pooled panel in step: it is built from the same blocks and is what every
    # pooled-training panel is evaluated on.
    build_pooled_panel(force=force)
    return market, sector


def feature_families(panel: str) -> dict[str, list[str]]:
    """The factor families available for a panel, for ablation and for training.

    'core' is the series' own price/volatility history and is never ablated away — it is
    the baseline every other family has to improve on.
    """
    fams = {
        "core": list(CORE_FEATURES),
        # Own-history blocks. These were built into every panel from the start but were
        # only reachable through hand-written entries in the methods bake-off, so the
        # selection study could never choose them. They are standard specifications —
        # HAR-J and HAR-RS — and belong in the same competition as everything else.
        "jump": list(JUMP_FEATURES),
        "semivar": list(SEMIVAR_FEATURES),
        "range_ext": list(RANGE_EXT_FEATURES),
        # External series.
        "vix": list(factors.VIX_FEATURES),
        "vix_short": list(factors.VIX_SHORT_FEATURES),
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


def training_frame(panel: str, features: list[str] | None = None) -> pd.DataFrame:
    """The frame a panel's model is FITTED on, given the features it will use.

    The feature list matters because of `SECTOR_ONLY_FAMILIES`. `sec_rv_rel`, `sec_rv_rank`
    and `sec_beta_63` are undefined for a single stock, so pooling stocks in while one of
    them is selected would hand `dataset.labelled` a frame where 80% of the rows are NaN in
    a required column — and its `dropna` would delete them without a word. The training set
    would silently shrink from ~471k rows to ~37k and nothing downstream would notice.

    Narrowing here instead makes that consequence explicit and auditable.
    """
    if config.TRAINING_SOURCE.get(panel) != "pooled":
        return load_panel(panel)

    pooled = build_pooled_panel()
    feats = features if features is not None else all_features(panel)
    if any(f in SECTOR_REL_FEATURES for f in feats):
        return pooled[pooled["ticker"].isin(config.SECTOR_ETFS)].reset_index(drop=True)
    return pooled


def load_training_panel(panel: str) -> pd.DataFrame:
    """The frame a panel's model is FITTED on, for the SHIPPED feature set."""
    return training_frame(panel)


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
    return D.labelled(training_frame(panel, features), features, h), eval_df


if __name__ == "__main__":
    build_panels(force=True)
