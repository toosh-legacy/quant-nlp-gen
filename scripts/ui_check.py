"""End-to-end dashboard check with Playwright, against a running server.

    python -m volradar.cli serve            # in one terminal
    python scripts/ui_check.py              # in another; or pass a URL

Drives every tab in system Chrome (no browser download needed), exercises the radar
hover/click, the sector drill-in and the news filters, checks there is no horizontal page
scroll at phone width and no console errors, and writes screenshots to data/ui_shots/.
Exits non-zero on any failure.

Setup: pip install playwright   (uses channel="chrome"; or `playwright install chromium`
and set UI_CHECK_CHANNEL= to use the bundled browser)
"""

import os
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
OUT = Path(__file__).resolve().parent.parent / "data" / "ui_shots"
OUT.mkdir(parents=True, exist_ok=True)
CHANNEL = os.environ.get("UI_CHECK_CHANNEL", "chrome") or None
problems = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        problems.append(msg)


with sync_playwright() as p:
    browser = p.chromium.launch(channel=CHANNEL, headless=True)
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("pageerror", lambda e: errors.append(str(e)))

    print("overview")
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector(".headline .big")
    page.wait_for_timeout(1200)
    check(page.locator(".gauge line[stroke='#fff']").count() == 1, "gauge needle rendered")
    blips = page.locator(".radar .blip")
    check(blips.count() == 11, f"11 radar blips (got {blips.count()})")
    check(page.locator("table tbody tr").count() == 11, "sector table has 11 rows")
    check(page.locator(".feed-item").count() >= 1, "news pulse shows headlines")
    page.screenshot(path=str(OUT / "1_overview.png"), full_page=True)

    blips.nth(0).hover()
    page.wait_for_timeout(300)
    check(page.locator(".radar-tip").is_visible(), "radar tooltip on hover")
    page.screenshot(path=str(OUT / "2_radar_hover.png"),
                    clip={"x": 700, "y": 130, "width": 660, "height": 620})

    title_before = page.locator(".card h2").filter(has_text="forecast vs realized").inner_text()
    page.locator("table tbody tr").nth(0).click()
    page.wait_for_timeout(1500)
    title_after = page.locator(".card h2").filter(has_text="forecast vs realized").inner_text()
    check(title_before != title_after, f"sector drill-in: '{title_before}' -> '{title_after}'")
    check(page.locator(".recharts-line").count() >= 2, "history chart lines rendered")

    print("news")
    page.locator(".tab", has_text="News radar").click()
    page.wait_for_timeout(1000)
    check("#news" in page.url, "tab in URL hash")
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
    page.screenshot(path=str(OUT / "3_news.png"), full_page=True)

    print("track record / method")
    page.locator(".tab", has_text="Track record").click()
    page.wait_for_timeout(1500)
    check(page.locator(".card").count() > 5, "track record cards rendered")
    page.screenshot(path=str(OUT / "4_record.png"), full_page=True)
    page.locator(".tab", has_text="Method").click()
    page.wait_for_timeout(600)
    check(page.get_by_text("News semantics overlay").count() == 1, "method has news section")
    page.screenshot(path=str(OUT / "5_method.png"), full_page=True)

    print("mobile")
    m = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1,
                            is_mobile=True, has_touch=True).new_page()
    m.on("pageerror", lambda e: errors.append(str(e)))
    for tab in ["", "#news", "#record", "#method"]:
        m.goto(URL + tab, wait_until="networkidle")
        m.wait_for_timeout(900)
        sw = m.evaluate("document.documentElement.scrollWidth")
        check(sw <= 392, f"mobile {tab or '#overview'} no horizontal page scroll (scrollWidth {sw})")
    m.goto(URL, wait_until="networkidle")
    m.wait_for_timeout(900)
    m.screenshot(path=str(OUT / "6_mobile.png"), full_page=True)

    check(not errors, f"no console/page errors {errors[:3]}")
    browser.close()

print("\nRESULT:", "PASS" if not problems else f"{len(problems)} problem(s)")
sys.exit(1 if problems else 0)
