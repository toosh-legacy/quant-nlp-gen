"""Fit the production models and write the forecast snapshot the dashboard reads.

Distinct from the backtest: here the models train on *all* available history so the live
forecast uses everything known, and the artifacts are saved with the metadata needed to
reproduce them (training window, feature list, parameters, panel, horizon, and the frozen
regime thresholds).

The snapshot is a single JSON file. The API serves it directly, so the web layer holds no
modelling logic and cannot quietly disagree with the backtest.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest.oos import annualize
from volradar.features.assemble import all_features, fit_and_eval_frames, load_panel
from volradar.models import dataset as D
from volradar.models import estimators as E
from volradar.models import conformal

SNAPSHOT_PATH = config.REPORT_DIR / "snapshot.json"


def _feature_contributions(model: E.Fitted, row: pd.DataFrame, background: pd.DataFrame,
                           top_n: int = 8) -> list[dict]:
    """Which inputs are pushing today's forecast away from a typical day?

    Uses the tree ensemble's own per-feature contributions (`pred_contribs`), which
    decompose a single prediction exactly into a bias term plus one term per feature.
    Values are reported in annualized-vol points so the dashboard can say "the VIX term
    structure is adding 1.4 points" rather than showing a unitless score.
    """
    try:
        import xgboost as xgb
        booster = model.estimator.get_booster()
        dm = xgb.DMatrix(row[model.columns].to_numpy(float), feature_names=model.columns)
        contribs = booster.predict(dm, pred_contribs=True)[0]
    except Exception:
        return []

    base_pred = float(np.mean(background["pred"])) if "pred" in background else 0.0
    out = []
    for name, c in zip(model.columns, contribs[:-1]):
        out.append({"feature": name, "contribution_log": float(c)})
    out.sort(key=lambda d: abs(d["contribution_log"]), reverse=True)

    # Convert each contribution to the annualized-vol change it represents at this level.
    total = float(contribs.sum())
    for d in out:
        d["contribution_vol_pts"] = float(
            annualize(total) - annualize(total - d["contribution_log"]))
        d.pop("contribution_log")
    return out[:top_n]


def _live_band(panel: str, pred_log: float) -> dict:
    """A calibrated uncertainty band for today's forecast.

    The half-width is taken from the most recent *adaptive* conformal interval in the
    out-of-sample series rather than recomputed from a nominal quantile. That matters: the
    static interval demonstrably under-covers on the market panel (73% against a nominal
    80%), because financial data is not exchangeable. The adaptive width is the one whose
    realized coverage was actually measured, so the band shown is the band that has been
    verified — and the measured figure is carried alongside it rather than the promise.
    """
    path = config.REPORT_DIR / f"oos_{panel}.parquet"
    cov_path = config.REPORT_DIR / f"conformal_{panel}.csv"
    if not path.exists():
        return {}
    oos = pd.read_parquet(path)
    if "pred_lo_adaptive" not in oos.columns:
        return {}
    last = oos.sort_values("date").dropna(subset=["pred_lo_adaptive"]).tail(1)
    if last.empty:
        return {}
    half = float(last["pred_hi_adaptive"].iloc[0] - last["pred_lo_adaptive"].iloc[0]) / 2.0

    out = {
        "forecast_lo_ann": round(float(annualize(pred_log - half)), 2),
        "forecast_hi_ann": round(float(annualize(pred_log + half)), 2),
        "nominal_coverage": conformal.COVERAGE,
    }
    if cov_path.exists():
        c = pd.read_csv(cov_path).iloc[0]
        out["measured_coverage"] = round(float(c.get("adaptive_realized_coverage", float("nan"))), 4)
    return out


def fit_production(panel: str, h: int) -> tuple[E.Fitted, dict]:
    """Train on all labelled history; return the model and its metadata."""
    features = all_features(panel)
    tr_sub, sub = fit_and_eval_frames(panel, features, h)

    fit = D.embargoed_train(tr_sub, h, pd.Timestamp(config.VAL_START))
    val = D.time_slice(sub, config.VAL_START, config.VAL_END)
    params = (E.tune_gbm(fit, val, features, h, quiet=True)
              if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)

    model = E.fit_gbm(tr_sub, features, h, params)
    # Regime bands describe THIS panel's own volatility distribution, so they come from the
    # evaluation frame even when the model was fitted on the pooled cross-section — a band
    # derived from single stocks would call a normal index day "calm".
    thr = D.regime_thresholds(sub, h)
    meta = {
        "panel": panel, "horizon": h, "model": "gbm", "params": params,
        "training_source": config.TRAINING_SOURCE.get(panel, "own"),
        "n_train": int(len(tr_sub)), "n_features": len(features), "features": features,
        "train_start": str(tr_sub["date"].min().date()),
        "train_end": str(tr_sub["date"].max().date()),
        "regime_thresholds": {"calm_max": thr[0], "normal_max": thr[1]},
        "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return model, meta


def build_snapshot() -> dict:
    """Current market + sector forecasts, plus the metrics that qualify them."""
    config.ensure_dirs()
    h = config.HEADLINE_HORIZON
    snapshot: dict = {"horizon": h, "generated_utc":
                      datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # --- market ---------------------------------------------------------------------
    mkt_model, mkt_meta = fit_production("market", h)
    mkt = load_panel("market")
    mkt_feat = all_features("market")
    latest_m = mkt.dropna(subset=mkt_feat).sort_values("date").tail(1)
    pred_m = float(mkt_model.predict(latest_m)[0])
    cur_m = float(latest_m[config.persist_col(h)].iloc[0])
    thr = (mkt_meta["regime_thresholds"]["calm_max"],
           mkt_meta["regime_thresholds"]["normal_max"])

    snapshot["market"] = {
        "as_of": str(latest_m["date"].iloc[0].date()),
        "current_vol_ann": round(float(annualize(cur_m)), 2),
        "forecast_vol_ann": round(float(annualize(pred_m)), 2),
        "change_pct": round(float(np.exp(pred_m - cur_m) - 1) * 100, 1),
        "direction": "rising" if pred_m > cur_m else "falling",
        "regime": config.REGIME_NAMES[int(D.to_regime([pred_m], thr)[0])],
        "current_regime": config.REGIME_NAMES[int(D.to_regime([cur_m], thr)[0])],
        "regime_bands_ann": {
            "calm_max": round(float(annualize(thr[0])), 2),
            "normal_max": round(float(annualize(thr[1])), 2),
        },
        **_live_band("market", pred_m),
        "drivers": _feature_contributions(mkt_model, latest_m, mkt),
        "meta": {k: v for k, v in mkt_meta.items() if k != "features"},
    }

    # --- sectors --------------------------------------------------------------------
    sec_model, sec_meta = fit_production("sector", h)
    sec = load_panel("sector")
    sec_feat = all_features("sector")
    ok = sec.dropna(subset=sec_feat)
    latest_date = ok["date"].max()
    latest_s = ok[ok["date"] == latest_date].copy()
    latest_s["pred"] = sec_model.predict(latest_s)

    sthr = (sec_meta["regime_thresholds"]["calm_max"],
            sec_meta["regime_thresholds"]["normal_max"])
    sectors = []
    for _, r in latest_s.iterrows():
        cur = float(r[config.persist_col(h)])
        pr = float(r["pred"])
        sectors.append({
            "ticker": r["ticker"],
            "sector": r.get("sector_name", r["ticker"]),
            "current_vol_ann": round(float(annualize(cur)), 2),
            "forecast_vol_ann": round(float(annualize(pr)), 2),
            "change_pct": round(float(np.exp(pr - cur) - 1) * 100, 1),
            "direction": "rising" if pr > cur else "falling",
            "regime": config.REGIME_NAMES[int(D.to_regime([pr], sthr)[0])],
        })
    sectors.sort(key=lambda d: d["forecast_vol_ann"])
    for i, s in enumerate(sectors, 1):
        s["stability_rank"] = i          # 1 = most stable forecast vol
    snapshot["sectors"] = {
        "as_of": str(latest_date.date()),
        "ranking": sectors,
        "meta": {k: v for k, v in sec_meta.items() if k != "features"},
    }
    return snapshot


def main() -> None:
    config.ensure_dirs()
    print("Fitting production models...")
    snap = build_snapshot()
    SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    m = snap["market"]
    print(f"\n  market as of {m['as_of']}: {m['current_vol_ann']}% -> "
          f"{m['forecast_vol_ann']}% ({m['change_pct']:+.1f}%, {m['regime']})")
    print("  sector stability ranking (forecast annualized vol):")
    for s in snap["sectors"]["ranking"]:
        print(f"    {s['stability_rank']:>2}. {s['sector']:<22} "
              f"{s['forecast_vol_ann']:>6.1f}%  ({s['change_pct']:+5.1f}%, {s['regime']})")
    print(f"\nWrote -> {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
