"""Build the news radar: market + per-sector readings, the archive, and its validation status.

Every reading is computed from headlines published in the last WINDOW_HOURS, recency-
weighted (24h half-life) and breadth-weighted (a story run by several channels counts more).
Per group (the market, and each sector):

    net_tone   weighted mean of tone x confidence, in [-1, +1]
    intensity  weighted mean shock-word score (see semantics.SHOCK_WORDS)
    pressure   0-100 = 50 * (1 - net_tone) / 2  +  50 * (1 - exp(-intensity))
               half from how negative the coverage is, half from how loud it is.
               0 is quiet and positive; 100 is maximally negative and alarmed.
    attention  the sector's share of all sector-tagged coverage

`vs_typical` is a z-score of today's pressure against the archived history of the same
group, published only once there are enough days for it to mean something.

The forecast is never touched. Where news and forecast disagree sharply, the sector is
flagged as a *divergence* for a human to look at — with the caveat that nothing yet shows
the news side of that disagreement is the one to believe.

Run:
    python -m volradar.cli news
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from volradar import config
from volradar.news import scoring, semantics, sources

RADAR_PATH = config.REPORT_DIR / "news_radar.json"
ARCHIVE_PATH = config.NEWS_DIR / "headlines.parquet"
DAILY_PATH = config.NEWS_DIR / "daily.parquet"

WINDOW_HOURS = 72
HALF_LIFE_H = 24.0
MIN_DAYS_FOR_Z = 10
DIVERGENCE_RANK_GAP = 5

CAVEAT = ("Descriptive only. Free news feeds carry days of history, not years, so news "
          "tone has not been walk-forward validated and is NOT an input to the volatility "
          "forecast. Every headline is archived so that it can be tested later under the "
          "same paired-test rule as every other candidate feature.")


# --------------------------------------------------------------------------- archive
def _listify(v) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    return [x for x in str(v).split("|") if x]


def _to_archive(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["id", "published", "fetched", "source", "channel", "title", "link", "label",
              "confidence", "tone", "shock"]].copy()
    for col in ("channels", "ticker_hints", "sectors", "themes"):
        out[col] = df[col].map(lambda v: "|".join(_listify(v)))
    return out


def update_archive(scored: pd.DataFrame, scorer: str) -> pd.DataFrame:
    """Append new stories; a story already archived keeps its first sighting but gains any
    new channels that have since carried it."""
    config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
    new = _to_archive(scored).assign(scorer=scorer)
    if ARCHIVE_PATH.exists():
        old = pd.read_parquet(ARCHIVE_PATH)
        merged = pd.concat([old, new], ignore_index=True)
        chans = merged.groupby("id")["channels"].agg(
            lambda s: "|".join(sorted({c for v in s for c in _listify(v)})))
        merged = merged.drop_duplicates("id", keep="first").set_index("id")
        merged["channels"] = chans
        new = merged.reset_index()
    new = new.sort_values("published")
    new.to_parquet(ARCHIVE_PATH, index=False)
    return new


# --------------------------------------------------------------------------- reading
def _weights(df: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    age_h = (now - df["published"]).dt.total_seconds().clip(lower=0) / 3600
    breadth = df["channels"].map(lambda v: len(_listify(v)) or 1)
    return 0.5 ** (age_h / HALF_LIFE_H) * (1 + 0.25 * (breadth - 1))


def _theme_counts(df: pd.DataFrame, w: pd.Series) -> list[dict]:
    acc: dict[str, list[float]] = {}
    for themes, wi in zip(df["themes"], w):
        for t in _listify(themes):
            a = acc.setdefault(t, [0.0, 0])
            a[0] += wi
            a[1] += 1
    rows = [{"theme": k, "weight": round(v[0], 3), "n": v[1]} for k, v in acc.items()]
    return sorted(rows, key=lambda r: r["weight"], reverse=True)


def _headline(r) -> dict:
    return {"title": r["title"], "link": r["link"], "channels": _listify(r["channels"]),
            "published": r["published"].isoformat(), "label": r["label"],
            "confidence": round(float(r["confidence"]), 3), "tone": int(r["tone"]),
            "shock": int(r["shock"]), "sectors": _listify(r["sectors"]),
            "themes": _listify(r["themes"])}


def reading(df: pd.DataFrame, now: pd.Timestamp, top_n: int = 5) -> dict:
    """One group's news reading. Empty coverage reads as 'no data', never as calm."""
    if df.empty:
        return {"n": 0, "net_tone": None, "intensity": None, "pressure": None,
                "bearish": 0, "bullish": 0, "neutral": 0, "themes": [], "top": []}
    w = _weights(df, now)
    sw = float(w.sum())
    net_tone = float((w * df["tone"] * df["confidence"]).sum() / sw)
    intensity = float((w * df["shock"]).sum() / sw)
    pressure = 50 * (1 - net_tone) / 2 + 50 * (1 - math.exp(-intensity))
    # "Top" = most consequential, not most recent: loud, negative, widely carried, fresh.
    salience = w * (1 + df["shock"]) * (1 + 0.5 * (df["tone"] < 0))
    top = df.loc[salience.sort_values(ascending=False).index[:top_n]]
    return {
        "n": int(len(df)),
        "net_tone": round(net_tone, 3),
        "intensity": round(intensity, 3),
        "pressure": round(pressure, 1),
        "bearish": int((df["label"] == "Bearish").sum()),
        "bullish": int((df["label"] == "Bullish").sum()),
        "neutral": int((df["label"] == "Neutral").sum()),
        "themes": _theme_counts(df, w)[:8],
        "top": [_headline(r) for _, r in top.iterrows()],
    }


def _vs_typical(daily: pd.DataFrame, group: str, today: str, value: float | None):
    if value is None or daily.empty:
        return None
    hist = daily[(daily["group"] == group) & (daily["date"] < today)]["pressure"].dropna()
    if len(hist) < MIN_DAYS_FOR_Z or hist.std() == 0:
        return None
    return round(float((value - hist.mean()) / hist.std()), 2)


def _update_daily(rows: list[dict]) -> pd.DataFrame:
    """One row per (date, group); a later run on the same day replaces the earlier one."""
    new = pd.DataFrame(rows)
    if DAILY_PATH.exists():
        old = pd.read_parquet(DAILY_PATH)
        new = pd.concat([old, new]).drop_duplicates(["date", "group"], keep="last")
    new = new.sort_values(["date", "group"])
    new.to_parquet(DAILY_PATH, index=False)
    return new


def _timeline(df: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    if df.empty:
        return []
    start = (now - pd.Timedelta(hours=WINDOW_HOURS)).floor("6h")
    b = df.assign(bucket=df["published"].dt.floor("6h"))
    g = b.groupby(["bucket", "label"]).size().unstack(fill_value=0)
    idx = pd.date_range(start, now.floor("6h"), freq="6h", tz="UTC")
    g = g.reindex(idx, fill_value=0)
    return [{"t": t.isoformat(), "bearish": int(r.get("Bearish", 0)),
             "neutral": int(r.get("Neutral", 0)), "bullish": int(r.get("Bullish", 0))}
            for t, r in g.iterrows()]


def _forecast_sectors() -> dict[str, dict]:
    snap = config.REPORT_DIR / "snapshot.json"
    if not snap.exists():
        return {}
    s = json.loads(snap.read_text(encoding="utf-8"))
    return {r["ticker"]: r for r in s.get("sectors", {}).get("ranking", [])}


def _divergences(sectors: list[dict]) -> None:
    """Flag sectors whose news-pressure rank and forecast-volatility rank disagree by at
    least DIVERGENCE_RANK_GAP places (rank 1 = most pressure / most volatile)."""
    ok = [s for s in sectors
          if s["pressure"] is not None and s.get("forecast_vol_ann") is not None]
    by_p = sorted(ok, key=lambda s: -s["pressure"])
    by_v = sorted(ok, key=lambda s: -s["forecast_vol_ann"])
    rp = {s["ticker"]: i + 1 for i, s in enumerate(by_p)}
    rv = {s["ticker"]: i + 1 for i, s in enumerate(by_v)}
    for s in sectors:
        s["divergence"] = None
        if s["ticker"] not in rp:
            continue
        s["news_rank"], s["vol_rank"] = rp[s["ticker"]], rv[s["ticker"]]
        gap = rv[s["ticker"]] - rp[s["ticker"]]
        if gap >= DIVERGENCE_RANK_GAP:
            s["divergence"] = "news_hotter"     # loud news, calm forecast
        elif gap <= -DIVERGENCE_RANK_GAP:
            s["divergence"] = "news_calmer"     # quiet news, volatile forecast


# --------------------------------------------------------------------------- validation
def validation_status(daily: pd.DataFrame | None = None) -> dict:
    """How close the archive is to supporting an honest test, and — once it is — a first
    descriptive look: rank correlation between the day's market news pressure and the
    out-of-sample forecast's subsequent error. This is a screen, not the adoption rule;
    adoption still requires metrics.paired_test inside the walk-forward harness."""
    if daily is None:
        daily = pd.read_parquet(DAILY_PATH) if DAILY_PATH.exists() else pd.DataFrame()
    m = daily[daily["group"] == "market"] if not daily.empty else daily
    n_days = int(m["date"].nunique()) if not m.empty else 0
    need = config.NEWS_MIN_DAYS_FOR_TEST
    out = {"archived_days": n_days, "days_needed": need,
           "first_day": None if m.empty else str(m["date"].min()),
           "status": "accruing" if n_days < need else "testable"}
    if n_days < need:
        out["message"] = (f"{n_days} of {need} archived days. A validation is attempted "
                          f"once the archive spans {need} days.")
        return out

    oos_path = config.REPORT_DIR / "oos_market.parquet"
    if not oos_path.exists():
        out["message"] = "oos_market.parquet missing — run `python -m volradar.cli oos`."
        return out
    oos = pd.read_parquet(oos_path)
    h = int(oos["horizon"].iloc[0])
    oos = oos.assign(date=pd.to_datetime(oos["date"]).dt.strftime("%Y-%m-%d"),
                     resid=oos[config.fwd_col(h)] - oos["pred"])[["date", "resid"]]
    j = m.merge(oos.dropna(), on="date")
    out["overlap_days"] = int(len(j))
    if len(j) < need:
        out["message"] = (f"Only {len(j)} archived days have a scored out-of-sample "
                          f"outcome yet; re-run `oos --force` as labels mature.")
        return out
    rho = float(j["pressure"].rank().corr(j["resid"].rank()))
    t = rho * math.sqrt((len(j) - 2) / max(1e-12, 1 - rho ** 2))
    out.update({"spearman_pressure_vs_residual": round(rho, 3), "t": round(t, 2),
                "message": "Screen only — a positive rho means the forecast tended to "
                           "under-shoot after high-pressure days. Not yet an adoption test."})
    return out


# --------------------------------------------------------------------------- main
def build(force_lexicon: bool = False) -> dict:
    config.ensure_dirs()
    now = pd.Timestamp(datetime.now(timezone.utc))
    print("Fetching headlines from news channels...")
    raw, coverage = sources.fetch_headlines()
    ok = [c for c in coverage if not c["error"]]
    print(f"  {len(raw)} unique stories from {len(ok)}/{len(coverage)} feeds")
    for c in coverage:
        if c["error"]:
            print(f"  [warn] {c['feed']}: {c['error']}")

    if not raw.empty:
        tags = [semantics.annotate(t, s, h) for t, s, h in
                zip(raw["title"], raw["summary"], raw["ticker_hints"])]
        raw = raw.assign(sectors=[t["sectors"] for t in tags],
                         themes=[t["themes"] for t in tags],
                         shock=[t["shock"] for t in tags])
        # Only score stories the archive has not seen — scores are frozen at first sight,
        # so a later model update cannot silently rewrite the archived history.
        seen = set(pd.read_parquet(ARCHIVE_PATH, columns=["id"])["id"]) \
            if ARCHIVE_PATH.exists() else set()
        fresh = raw[~raw["id"].isin(seen)]
        scored, scorer = scoring.score(fresh, force_lexicon=force_lexicon)
        print(f"  scored {len(scored)} new stories with {scorer}")
        archive = update_archive(scored, scorer) if len(scored) else \
            pd.read_parquet(ARCHIVE_PATH)
    else:
        scorer = "none"
        archive = pd.read_parquet(ARCHIVE_PATH) if ARCHIVE_PATH.exists() else pd.DataFrame()

    if archive.empty:
        out = {"available": False, "generated_utc": now.isoformat(),
               "reason": "no headlines could be fetched and the archive is empty",
               "coverage": coverage, "caveat": CAVEAT}
        RADAR_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    archive["published"] = pd.to_datetime(archive["published"], utc=True)
    win = archive[archive["published"] >= now - pd.Timedelta(hours=WINDOW_HOURS)].copy()
    if "scorer" in win and win["scorer"].nunique() > 1:
        scorer = " + ".join(sorted(win["scorer"].unique()))
    elif "scorer" in win and len(win):
        scorer = win["scorer"].iloc[0]

    today = now.strftime("%Y-%m-%d")
    market = reading(win, now)
    fc = _forecast_sectors()
    tagged = win["sectors"].map(_listify)
    total_sector_w = float(_weights(win[tagged.map(bool)], now).sum()) or 1.0
    sectors = []
    for etf, name in config.SECTOR_ETFS.items():
        sub = win[tagged.map(lambda s, e=etf: e in s)]
        r = reading(sub, now, top_n=3)
        r.update({"ticker": etf, "sector": name,
                  "attention": round(float(_weights(sub, now).sum()) / total_sector_w, 3)
                  if len(sub) else 0.0})
        f = fc.get(etf, {})
        r.update({"forecast_vol_ann": f.get("forecast_vol_ann"), "regime": f.get("regime"),
                  "direction": f.get("direction")})
        sectors.append(r)
    _divergences(sectors)

    daily = _update_daily(
        [{"date": today, "group": "market", "n": market["n"], "pressure": market["pressure"],
          "net_tone": market["net_tone"], "intensity": market["intensity"]}]
        + [{"date": today, "group": s["ticker"], "n": s["n"], "pressure": s["pressure"],
            "net_tone": s["net_tone"], "intensity": s["intensity"]} for s in sectors])
    market["vs_typical"] = _vs_typical(daily, "market", today, market["pressure"])
    for s in sectors:
        s["vs_typical"] = _vs_typical(daily, s["ticker"], today, s["pressure"])

    chan = (win["channels"].map(_listify).explode().value_counts())
    # Every story in the window (capped), so filtering the feed by sector on the dashboard
    # agrees with the per-sector story counts in the same payload.
    latest = win.sort_values("published", ascending=False).head(400)

    out = {
        "available": True,
        "generated_utc": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scorer": scorer,
        "n_headlines": int(len(win)),
        "n_archived": int(len(archive)),
        "channels": [{"channel": c, "n": int(n)} for c, n in chan.items()],
        "coverage": coverage,
        "market": market,
        "sectors": sectors,
        "timeline": _timeline(win, now),
        "headlines": [_headline(r) for _, r in latest.iterrows()],
        "validation": validation_status(daily),
        "method": {
            "pressure": "50*(1-net_tone)/2 + 50*(1-exp(-intensity))",
            "weighting": f"recency half-life {HALF_LIFE_H:.0f}h x (1 + 0.25 per extra channel)",
            "divergence": f"news-pressure rank vs forecast-vol rank differ by >= "
                          f"{DIVERGENCE_RANK_GAP} of 11",
        },
        "validated_as_predictor": False,
        "caveat": CAVEAT,
    }
    RADAR_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"  market pressure {market['pressure']} · tone {market['net_tone']} · "
          f"{len(win)} stories in {WINDOW_HOURS}h window")
    print(f"Wrote -> {RADAR_PATH}")
    return out


def main(force_lexicon: bool = False) -> None:
    build(force_lexicon=force_lexicon)


if __name__ == "__main__":
    main()
