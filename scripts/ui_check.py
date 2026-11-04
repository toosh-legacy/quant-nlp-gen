"""End-to-end dashboard check with Playwright, against a running server.

    python -m volradar.cli serve            # in one terminal
    python scripts/ui_check.py              # in another; or pass a URL

Drives every page in system Chrome (no browser download needed): the overview gauge,
radar and table, navigation into every sector page by mouse and keyboard, the sector board,
the news filters, track record and method, plus phone-width layout and console errors.
Screenshots land in data/ui_shots/. Exits non-zero on any failure.

Setup: pip install playwright   (uses channel="chrome"; or `playwright install chromium`
and set UI_CHECK_CHANNEL= to use the bundled browser)
"""

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/").rstrip("/") + "/"
OUT = Path(__file__).resolve().parent.parent / "data" / "ui_shots"
OUT.mkdir(parents=True, exist_ok=True)
CHANNEL = os.environ.get("UI_CHECK_CHANNEL", "chrome") or None
problems = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        problems.append(msg)


def settle(page, ms=700):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


with sync_playwright() as p:
    browser = p.chromium.launch(channel=CHANNEL, headless=True)
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("response", lambda r: r.status >= 400 and "/api/" in r.url
            and errors.append(f"HTTP {r.status} {r.url}"))

    print("overview")
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector(".headline .big")
    settle(page)
    check(page.locator(".gauge line[stroke='#fff']").count() == 1, "gauge needle rendered")
    check(page.locator(".recent svg path").count() == 2, "market card shows its recent record")
    blips = page.locator(".radar .blip")
    check(blips.count() == 11, f"11 radar blips (got {blips.count()})")
    check(page.locator("table tbody tr").count() == 11, "sector table has 11 rows")
    check(page.locator(".feed-item").count() >= 1, "news pulse shows headlines")
    check(page.locator(".recharts-line").count() >= 2, "market history chart rendered")
    page.screenshot(path=str(OUT / "1_overview.png"), full_page=True)

    blips.nth(0).hover()
    page.wait_for_timeout(250)
    check(page.locator(".radar-tip").is_visible(), "radar tooltip on hover")
    blips.nth(3).click()
    settle(page)
    check(re.search(r"#sectors/[A-Z]+$", page.url) is not None, f"radar click opens a sector ({page.url})")
    check(page.locator(".facts").count() == 1, "sector detail page rendered")

    page.goto(URL + "#dashboard")
    settle(page)
    page.locator("table tbody tr").nth(0).focus()
    page.keyboard.press("Enter")
    settle(page)
    check("#sectors/" in page.url, "keyboard Enter on a table row opens the sector")

    print("sector board")
    page.goto(URL + "#sectors")
    settle(page, 1500)
    check(page.locator(".group").count() == 3, "three super-sector groups")
    check(page.locator(".srow").count() == 11, "11 sector rows on the board")
    check(page.locator("svg.trend path").count() == 22, "every row draws forecast + realized trend")
    page.screenshot(path=str(OUT / "2_sectors.png"), full_page=True)
    first = page.locator(".srow").nth(0)
    tk = first.get_attribute("href").split("/")[-1]
    first.click()
    settle(page, 1200)
    check(page.url.endswith(f"#sectors/{tk}"), f"board row opens {tk}")

    print("sector detail")
    check(page.locator(".facts > div").count() == 6, "six facts")
    check(page.locator(".peer").count() == 11 and page.locator(".peer.on").count() == 1,
          "peer bars with this sector highlighted")
    check(page.locator(".recharts-line").count() >= 2, "sector history chart rendered")
    check(page.locator(".pager a").count() == 2, "prev/next links")
    name = page.evaluate("document.querySelector('.page-title').firstChild.textContent.trim()")
    check(page.title().startswith(name), f"document title names the sector ({page.title()})")
    page.screenshot(path=str(OUT / "3_sector_detail.png"), full_page=True)
    before = page.url
    page.keyboard.press("ArrowRight")
    settle(page, 900)
    check(page.url != before and "#sectors/" in page.url, "ArrowRight moves to the next sector")
    page.go_back()
    settle(page, 600)
    check(page.url == before, "browser back returns to the previous sector")
    page.goto(URL + "#sectors/ZZZ")
    settle(page)
    check(page.get_by_text("No sector called").count() == 1, "unknown sector shows a recovery message")

    print("news")
    page.goto(URL + "#news")
    settle(page)
    n_all = page.locator(".feed .feed-item").count()
    page.get_by_role("button", name="Bearish", exact=True).click()
    page.wait_for_timeout(200)
    n_bear = page.locator(".feed .feed-item").count()
    check(0 < n_bear < n_all, f"tone filter narrows feed ({n_all} -> {n_bear})")
    check(page.locator(".feed .tone.Bullish").count() == 0, "bearish filter excludes bullish")
    page.get_by_role("button", name="All tones").click()
    page.locator("table tbody tr").nth(0).click()
    page.wait_for_timeout(200)
    n_sec = page.locator(".feed .feed-item").count()
    check(0 < n_sec <= n_all, f"sector filter narrows feed ({n_all} -> {n_sec})")
    check(page.locator(".recharts-bar-rectangle").count() > 0, "news bar charts rendered")
    page.screenshot(path=str(OUT / "4_news.png"), full_page=True)
    page.locator(".feed a.tag.sector").first.click()
    settle(page)
    check("#sectors/" in page.url, "a headline's sector tag opens that sector")

    print("track record / method")
    page.goto(URL + "#record")
    settle(page, 1200)
    check(page.locator(".card").count() > 5, "track record cards rendered")
    page.goto(URL + "#method")
    settle(page)
    check(page.get_by_text("News semantics overlay").count() == 1, "method has news section")

    print("mobile")
    m = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True,
                            has_touch=True).new_page()
    m.on("pageerror", lambda e: errors.append(str(e)))
    for tab in ["", "#sectors", "#sectors/XLK", "#news", "#record", "#method"]:
        m.goto(URL + tab, wait_until="networkidle")
        m.wait_for_timeout(900)
        sw = m.evaluate("document.documentElement.scrollWidth")
        check(sw <= 392, f"mobile {tab or '#overview'} has no horizontal scroll ({sw}px)")
    m.goto(URL + "#sectors", wait_until="networkidle")
    m.wait_for_timeout(1200)
    m.screenshot(path=str(OUT / "5_mobile_sectors.png"), full_page=True)

    check(not errors, f"no console, page or API errors {errors[:3]}")
    browser.close()

print("\nRESULT:", "PASS" if not problems else f"{len(problems)} problem(s)")
sys.exit(1 if problems else 0)
