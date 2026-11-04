"""News semantics layer: parsing, tagging, scoring, aggregation, archive. All offline."""

from __future__ import annotations

import pandas as pd
import pytest

from volradar import config
from volradar.news import radar, scoring, semantics, sources

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Stocks plunge as Fed signals rate hike</title><link>http://a</link>
<description>&lt;p&gt;Banks &amp;amp; tech fall&lt;/p&gt;</description>
<pubDate>Tue, 03 Nov 2026 14:00:00 GMT</pubDate></item>
<item><title></title><link>http://skip</link></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Nvidia beats estimates</title><link href="http://b"/>
<updated>2026-11-03T10:00:00Z</updated></entry></feed>"""


def test_parse_rss_and_atom():
    r = sources.parse_feed(RSS)
    assert len(r) == 1, "items without a title are dropped"
    assert r[0]["title"].startswith("Stocks plunge")
    assert "<p>" not in r[0]["summary"]
    assert r[0]["published"] == pd.Timestamp("2026-11-03 14:00", tz="UTC")
    a = sources.parse_feed(ATOM)
    assert a[0]["link"] == "http://b" and a[0]["published"].hour == 10


def test_headline_id_dedups_syndicated_copies():
    a = sources.headline_id("Oil jumps on OPEC cut")
    b = sources.headline_id("Oil jumps on OPEC cut - Reuters")
    c = sources.headline_id("oil JUMPS on OPEC cut!")
    assert a == b == c
    assert a != sources.headline_id("Oil falls on OPEC cut")


@pytest.mark.parametrize("text,etf", [
    ("Nvidia shares rally on data center demand", "XLK"),
    ("OPEC+ agrees to extend crude output cuts", "XLE"),
    ("JPMorgan earnings top estimates", "XLF"),
    ("FDA approves new obesity drug", "XLV"),
    ("Mortgage rates climb, home sales slow", "XLRE"),
    ("Boeing cuts delivery forecast", "XLI"),
])
def test_sector_tagging(text, etf):
    assert etf in semantics.tag_sectors(text)


def test_tickers_are_case_sensitive():
    # 'cat', 'de', 'ge', 't' as ordinary words must not tag a sector
    assert semantics.tag_sectors("the cat sat at the table and we went to see it") == []
    assert "XLI" in semantics.tag_sectors("CAT shares fall")


def test_feed_hint_tags_sector_and_unknown_hint_ignored():
    assert semantics.tag_sectors("Weekly ETF flows", hints=["XLU", "SPY"]) == ["XLU"]


def test_themes_and_shock():
    t = semantics.annotate("Stocks plunge in worst day since 2020 as tariffs hit")
    assert {"trade"} <= set(t["themes"])
    assert t["shock"] >= 4                       # plunge (2) + worst day (2)
    assert semantics.annotate("Stocks edge higher")["shock"] == 0
    # magnitude, not direction: a surge is loud too
    assert semantics.shock_score("Shares soar") > 0


def test_lexicon_scorer():
    df = pd.DataFrame({"title": ["Apple beats and stock rallies", "Bank shares plunge on losses",
                                 "Company holds annual meeting"]})
    out, name = scoring.score(df, force_lexicon=True)
    assert name.startswith("lexicon")
    assert out["label"].tolist() == ["Bullish", "Bearish", "Neutral"]
    assert out["tone"].tolist() == [1, -1, 0]


def _frame(rows, now):
    df = pd.DataFrame(rows)
    df["published"] = now - pd.to_timedelta(df.pop("age_h"), unit="h")
    for c, d in [("channels", "CNBC"), ("sectors", ""), ("themes", ""), ("link", "")]:
        if c not in df:
            df[c] = d
    return df


def test_reading_empty_is_no_data_not_calm():
    r = radar.reading(pd.DataFrame(), pd.Timestamp.now(tz="UTC"))
    assert r["n"] == 0 and r["pressure"] is None


def test_pressure_bounds_and_ordering():
    now = pd.Timestamp("2026-11-03 12:00", tz="UTC")
    calm = _frame([{"title": "x", "label": "Bullish", "confidence": 1.0, "tone": 1,
                    "shock": 0, "age_h": 1}], now)
    loud = _frame([{"title": "y", "label": "Bearish", "confidence": 1.0, "tone": -1,
                    "shock": 6, "age_h": 1}], now)
    pc, pl = radar.reading(calm, now)["pressure"], radar.reading(loud, now)["pressure"]
    assert pc == 0.0
    assert 99 < pl <= 100
    assert pc < pl


def test_recency_weighting_favours_fresh_news():
    now = pd.Timestamp("2026-11-03 12:00", tz="UTC")
    df = _frame([{"title": "old", "label": "Bearish", "confidence": 1.0, "tone": -1,
                  "shock": 0, "age_h": 48},
                 {"title": "new", "label": "Bullish", "confidence": 1.0, "tone": 1,
                  "shock": 0, "age_h": 0}], now)
    assert radar.reading(df, now)["net_tone"] > 0.5      # 48h = two half-lives


def test_divergence_flags_rank_disagreement():
    secs = [{"ticker": f"S{i}", "pressure": float(i), "forecast_vol_ann": float(i)}
            for i in range(11)]
    secs[0]["pressure"] = 100.0     # quietest forecast, loudest news
    radar._divergences(secs)
    assert secs[0]["divergence"] == "news_hotter"
    assert all(s["divergence"] != "news_hotter" for s in secs[1:])


def test_archive_keeps_first_sighting_and_merges_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NEWS_DIR", tmp_path)
    monkeypatch.setattr(radar, "ARCHIVE_PATH", tmp_path / "h.parquet")
    now = pd.Timestamp("2026-11-03 12:00", tz="UTC")
    base = {"id": "abc", "fetched": now, "source": "s", "title": "t", "link": "",
            "label": "Bearish", "confidence": 0.9, "tone": -1, "shock": 1,
            "ticker_hints": [], "sectors": ["XLK"], "themes": []}
    radar.update_archive(pd.DataFrame([{**base, "published": now, "channel": "CNBC",
                                        "channels": ["CNBC"]}]), "lexicon")
    out = radar.update_archive(pd.DataFrame([{**base, "published": now + pd.Timedelta("1h"),
                                              "label": "Bullish", "channel": "BBC",
                                              "channels": ["BBC"]}]), "lexicon")
    assert len(out) == 1
    assert out["label"].iloc[0] == "Bearish", "first score is frozen"
    assert out["channels"].iloc[0] == "BBC|CNBC"


def test_validation_reports_accruing_until_enough_days():
    daily = pd.DataFrame({"date": ["2026-11-01", "2026-11-02"], "group": "market",
                          "pressure": [30.0, 40.0]})
    v = radar.validation_status(daily)
    assert v["status"] == "accruing" and v["archived_days"] == 2
    assert v["days_needed"] == config.NEWS_MIN_DAYS_FOR_TEST


def test_news_endpoint_contract():
    from fastapi.testclient import TestClient
    from volradar.api.server import app
    r = TestClient(app).get("/api/news")
    if not (config.REPORT_DIR / "news_radar.json").exists():
        assert r.status_code == 503 and "news" in r.json()["detail"]
        return
    body = r.json()
    assert body["validated_as_predictor"] is False
    if body.get("available"):
        assert len(body["sectors"]) == len(config.SECTOR_ETFS)
        assert 0 <= body["market"]["pressure"] <= 100
