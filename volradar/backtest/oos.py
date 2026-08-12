"""Generate a genuinely out-of-sample prediction series by walk-forward.

Every downstream claim — the event study, the decision simulation, the dashboard's
forecast-vs-realized chart — is built on this one function, so that none of them can
accidentally show a number produced by a model that had seen the data it is predicting.

For each year in `config.WALK_YEARS` a fresh model is fitted on all prior data (embargoed
at the boundary) and used to predict that year only. Concatenating the years gives a long
prediction series in which every point was produced by a model trained strictly before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config
from volradar.features.assemble import all_features, load_panel
from volradar.models import dataset as D
from volradar.models import estimators as E

OOS_MARKET = config.REPORT_DIR / "oos_market.parquet"
OOS_SECTOR = config.REPORT_DIR / "oos_sector.parquet"


def build_oos(panel: str, h: int | None = None, model_name: str = "gbm") -> pd.DataFrame:
    """Walk-forward OOS predictions for one panel and horizon."""
    h = h or config.HEADLINE_HORIZON
    df = load_panel(panel)
    features = all_features(panel)
    sub = D.labelled(df, features, h)

    fit = D.embargoed_train(sub, h, pd.Timestamp(config.VAL_START))
    val = D.time_slice(sub, config.VAL_START, config.VAL_END)
    params = (E.tune_gbm(fit, val, features, h, quiet=True)
              if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)

    keep = ["date", config.fwd_col(h), config.persist_col(h), "ret", "close"]
    if panel == "sector":
        keep += ["ticker", "sector_name"]

    pieces = []
    for year in config.WALK_YEARS:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = D.embargoed_train(sub, h, cutoff)
        test = D.time_slice(sub, cutoff, f"{year + 1}-01-01")
        if len(train) < 400 or len(test) < 60:
            continue
        fn = dict(E.LADDER)[model_name]
        model = (fn(train, features, h, params) if model_name == "gbm"
                 else fn(train, features, h))
        out = test[[c for c in keep if c in test.columns]].copy()
        out["pred"] = model.predict(test)
        # Regime thresholds are frozen on this fold's training data, so the labels a user
        # would have seen in real time are exactly the labels we score.
        lo, hi = D.regime_thresholds(train, h)
        out["thr_lo"], out["thr_hi"] = lo, hi
        out["pred_regime"] = D.to_regime(out["pred"], (lo, hi))
        out["true_regime"] = D.to_regime(out[config.fwd_col(h)], (lo, hi))
        out["year"] = year
        pieces.append(out)

    res = pd.concat(pieces, ignore_index=True).sort_values(
        ["date"] if panel == "market" else ["ticker", "date"]).reset_index(drop=True)
    res["horizon"] = h
    res["panel"] = panel
    return res


def annualize(log_rv) -> np.ndarray:
    """log daily realized vol -> annualized volatility in percent."""
    return np.exp(np.asarray(log_rv, float)) * np.sqrt(config.TRADING_DAYS) * 100.0


def load_or_build(panel: str, force: bool = False) -> pd.DataFrame:
    config.ensure_dirs()
    path = OOS_MARKET if panel == "market" else OOS_SECTOR
    if path.exists() and not force:
        return pd.read_parquet(path)
    df = build_oos(panel)
    df.to_parquet(path, index=False)
    print(f"  OOS predictions -> {path.name} ({len(df):,} rows, "
          f"{df['date'].min().date()}..{df['date'].max().date()})")
    return df


if __name__ == "__main__":
    for p in ("market", "sector"):
        load_or_build(p, force=True)
