"""Free headline sources: RSS feeds from financial news channels.

Standard library only (urllib + ElementTree), fetched in parallel. A feed that is down,
slow or has changed schema is reported and skipped — one broken channel must never end a
run. Every headline keeps the channel it came from, so coverage is visible on the dashboard
rather than assumed.
"""

from __future__ import annotations

import hashlib
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import pandas as pd

from volradar import config

USER_AGENT = "Mozilla/5.0 (compatible; volradar/1.0; research dashboard)"
TIMEOUT_S = 12


@dataclass(frozen=True)
class Feed:
    name: str          # stable id
    channel: str       # display name of the news channel
    url: str
    ticker: str | None = None   # set when the feed is about one ETF — a strong sector hint


def _yahoo(t: str) -> str:
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"


BROAD_FEEDS = [
    Feed("cnbc_top", "CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    Feed("cnbc_markets", "CNBC", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    Feed("cnbc_finance", "CNBC", "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
    Feed("cnbc_earnings", "CNBC", "https://www.cnbc.com/id/15839135/device/rss/rss.html"),
    Feed("marketwatch", "MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    Feed("nasdaq", "Nasdaq", "https://www.nasdaq.com/feed/rssoutbound?category=Markets"),
    Feed("seekingalpha", "Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    Feed("ft_markets", "Financial Times", "https://www.ft.com/markets?format=rss"),
    Feed("bbc_business", "BBC", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    Feed("investing", "Investing.com", "https://www.investing.com/rss/news_25.rss"),
    Feed("fed", "Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
    Feed("gnews_markets", "Google News",
         "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"),
    Feed("yahoo_spy", "Yahoo Finance", _yahoo(config.MARKET_TICKER), config.MARKET_TICKER),
]

SECTOR_FEEDS = [Feed(f"yahoo_{t.lower()}", "Yahoo Finance", _yahoo(t), t)
                for t in config.SECTOR_ETFS]

ALL_FEEDS = BROAD_FEEDS + SECTOR_FEEDS

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s))).strip()


def headline_id(title: str) -> str:
    """Same story, same id — across channels and across runs. Google News appends
    ' - Publisher'; that suffix is stripped so a syndicated story is counted once."""
    norm = re.sub(r"\s+-\s+[^-]{2,40}$", "", title.lower())
    norm = re.sub(r"[^a-z0-9 ]", "", norm)
    return hashlib.sha1(_WS.sub(" ", norm).strip().encode()).hexdigest()[:16]


def _parse_time(s: str | None) -> pd.Timestamp | None:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return pd.Timestamp(dt).tz_convert("UTC")
    except (TypeError, ValueError):
        pass
    try:
        ts = pd.Timestamp(s)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def parse_feed(xml_bytes: bytes) -> list[dict]:
    """RSS 2.0 <item> or Atom <entry>, namespace-agnostic."""
    root = ET.fromstring(xml_bytes)
    out = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        fields = {c.tag.rsplit("}", 1)[-1]: c for c in el}
        title = clean_text(fields["title"].text if "title" in fields else None)
        if not title:
            continue
        link_el = fields.get("link")
        link = (link_el.text or link_el.get("href") or "") if link_el is not None else ""
        summary = fields.get("description", fields.get("summary"))
        when = next((fields[k].text for k in ("pubDate", "published", "updated", "date")
                     if k in fields), None)
        out.append({"title": title, "link": link.strip(),
                    "summary": clean_text(summary.text if summary is not None else "")[:400],
                    "published": _parse_time(when)})
    return out


def _fetch_one(feed: Feed) -> tuple[Feed, list[dict], str | None]:
    try:
        req = urllib.request.Request(feed.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return feed, parse_feed(r.read()), None
    except Exception as exc:              # network/schema drift must not break the run
        return feed, [], f"{type(exc).__name__}: {str(exc)[:80]}"


def fetch_headlines(feeds: list[Feed] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """All headlines from every reachable feed, de-duplicated by story.

    Returns (headlines, coverage) where coverage lists every feed with its item count or
    the error that stopped it, so the dashboard can say exactly where the reading came from.
    """
    feeds = feeds or ALL_FEEDS
    now = pd.Timestamp(datetime.now(timezone.utc))
    rows, coverage = [], []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for feed, items, err in pool.map(_fetch_one, feeds):
            coverage.append({"feed": feed.name, "channel": feed.channel,
                             "items": len(items), "error": err})
            for it in items:
                rows.append({**it, "source": feed.name, "channel": feed.channel,
                             "ticker_hint": feed.ticker, "fetched": now})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, coverage

    df["id"] = df["title"].map(headline_id)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    # A missing timestamp is treated as "seen now", never back-dated.
    df["published"] = df["published"].fillna(df["fetched"]).clip(upper=now)
    # Keep the earliest sighting of each story, but remember every channel that ran it —
    # a story carried by five channels is louder than one carried by one.
    df = df.sort_values("published")
    channels: dict[str, set] = {}
    hints: dict[str, set] = {}
    for i, c, t in zip(df["id"], df["channel"], df["ticker_hint"]):
        channels.setdefault(i, set()).add(c)
        if isinstance(t, str):
            hints.setdefault(i, set()).add(t)
    df = df.drop_duplicates("id", keep="first").reset_index(drop=True)
    df["channels"] = [sorted(channels[i]) for i in df["id"]]
    df["ticker_hints"] = [sorted(hints.get(i, ())) for i in df["id"]]
    return df, coverage
