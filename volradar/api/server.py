"""FastAPI service. Reads artifacts, computes nothing.

Deliberate design constraint: every number this API returns was produced by the pipeline
and written to `data/reports/`. The web layer holds no modelling logic, so the dashboard
physically cannot show a figure that disagrees with the backtest. If a report file is
missing the endpoint says so rather than improvising.

Run:
    python -m volradar.cli serve
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from volradar import config
from volradar.backtest.oos import annualize

app = FastAPI(title="Volatility Radar", version="1.0",
              description="Market and sector volatility forecasts with their own track record.")

# The dashboard is served from the same origin in production; CORS is open only for the
# Vite dev server on localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

FRONTEND_DIST = config.ROOT / "frontend" / "dist"


def _report(name: str) -> pd.DataFrame:
    path = config.REPORT_DIR / name
    if not path.exists():
        raise HTTPException(503, f"{name} not generated yet — run `python -m volradar.cli all`.")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _clean(obj):
    """JSON-safe: NaN/Inf are not valid JSON and silently break the frontend."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.replace({np.nan: None}).to_dict(orient="records"))


@lru_cache(maxsize=1)
def _snapshot() -> dict:
    path = config.REPORT_DIR / "snapshot.json"
    if not path.exists():
        raise HTTPException(503, "snapshot.json missing — run `python -m volradar.cli train`.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    have = {p.name: p.exists() for p in [
        config.REPORT_DIR / "snapshot.json",
        config.REPORT_DIR / "pooled_oos.csv",
        config.REPORT_DIR / "per_sector.csv",
        config.REPORT_DIR / "events.csv",
        config.REPORT_DIR / "decision.csv",
        config.REPORT_DIR / "ablation.csv",
    ]}
    return {"status": "ok" if all(have.values()) else "incomplete", "artifacts": have}


@app.get("/api/forecast")
def forecast():
    """Current market forecast: level, direction, regime, and what is driving it."""
    return _clean(_snapshot()["market"])


@app.get("/api/sectors")
def sectors():
    """Sector stability ranking, joined to each sector's measured track record.

    The per-sector accuracy is attached on purpose: a ranking without it invites the reader
    to trust every row equally, and the model does not forecast every sector equally well.
    """
    snap = _clean(_snapshot()["sectors"])
    try:
        track = _report("per_sector.csv").set_index("ticker")
        for row in snap["ranking"]:
            t = row["ticker"]
            if t in track.index:
                r = track.loc[t]
                row["track_record"] = _clean({
                    "dir_accuracy": r.get("dir_accuracy"),
                    "skill_vs_persistence": r.get("skill_vs_persistence"),
                    "n_folds": r.get("n_folds"),
                })
    except HTTPException:
        pass
    return snap


@app.get("/api/history")
def history(panel: str = "market", ticker: str | None = None, limit: int = 1500):
    """Out-of-sample forecast vs what actually happened — the honesty chart.

    Values are annualized volatility in percent, the unit the rest of the dashboard uses.
    """
    path = config.REPORT_DIR / f"oos_{panel}.parquet"
    if not path.exists():
        raise HTTPException(503, f"oos_{panel}.parquet missing — run the backtest.")
    df = pd.read_parquet(path)
    if ticker:
        df = df[df["ticker"] == ticker]
    if df.empty:
        raise HTTPException(404, f"no out-of-sample rows for {panel}/{ticker}")

    h = int(df["horizon"].iloc[0])
    df = df.sort_values("date").tail(limit)
    return _clean({
        "panel": panel, "ticker": ticker, "horizon": h,
        "series": [{"date": str(d.date()),
                    "forecast": round(float(f), 2),
                    "realized": round(float(a), 2),
                    "current": round(float(c), 2),
                    "regime": int(r)}
                   for d, f, a, c, r in zip(
                       df["date"], annualize(df["pred"]),
                       annualize(df[config.fwd_col(h)]),
                       annualize(df[config.persist_col(h)]), df["pred_regime"])],
    })


@app.get("/api/performance")
def performance():
    """Everything that qualifies the forecast: accuracy, stability, events, utility."""
    out: dict = {"horizons_served": config.PRODUCT_HORIZONS,
                 "horizons_withheld": config.UNSTABLE_HORIZONS}
    for key, fname in [("pooled", "pooled_oos.csv"), ("per_sector", "per_sector.csv"),
                       ("events", "events.csv"), ("events_sweep", "events_sweep.csv"),
                       ("decision", "decision.csv"), ("ablation", "ablation.csv"),
                       ("decision_significance", "decision_significance.csv"),
                       ("family_selection", "family_selection.csv"),
                       ("fixed_split", "fixed_split.csv")]:
        try:
            out[key] = _records(_report(fname))
        except HTTPException:
            out[key] = None
    try:
        folds = _report("walkforward_folds.csv")
        stab = (folds.groupby(["panel", "horizon"])["skill_vs_persistence"]
                .agg(mean="mean", std="std", min="min",
                     years_positive=lambda s: int((s > 0).sum()), n="count")
                .reset_index())
        out["stability"] = _records(stab)
        out["folds"] = _records(folds)
    except HTTPException:
        out["stability"] = out["folds"] = None
    return out


@app.get("/api/equity")
def equity():
    """Equity curves from the volatility-targeting simulation."""
    df = _report("decision_curves.parquet")
    df = df.iloc[::5]        # weekly sampling keeps the payload small; shape is unchanged
    return _clean({"series": [
        {"date": str(pd.Timestamp(d).date()),
         "buy_and_hold": round(float(b), 4),
         "persistence_sized": round(float(p), 4),
         "model_sized": round(float(m), 4)}
        for d, b, p, m in zip(df["date"], df["buy_and_hold_equity"],
                              df["persistence_sized_equity"], df["model_sized_equity"])]})


# --- static frontend (only mounted once the dashboard has been built) ------------------
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIST / "index.html")
