"""Export the API as static JSON so the dashboard can be hosted without a server.

The API computes nothing: every response is a pure function of data/reports/. So instead of
running FastAPI in production, this calls each route once and writes the response to a file
the frontend fetches directly (VITE_STATIC=1 switches api.js to these paths).

    python scripts/export_static.py                # -> frontend/public/api/
    python scripts/export_static.py --out site/api

Layout (mirrors the routes):
    forecast.json  sectors.json  performance.json  equity.json  news.json  health.json
    history/market.json
    history/sector/<TICKER>.json

History files carry the API's maximum window (1500 rows); the client slices to the limit
it asked for, so a static page and a live server show the same points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import HTTPException  # noqa: E402

from volradar import config  # noqa: E402
from volradar.api import server  # noqa: E402

HISTORY_LIMIT = 1500


def _write(path: Path, payload) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return len(text)


def export(out: Path) -> dict[str, int]:
    written: dict[str, int] = {}

    def put(rel: str, fn, *args, optional: bool = False, **kw):
        try:
            written[rel] = _write(out / rel, fn(*args, **kw))
        except HTTPException as exc:
            if not optional:
                sys.exit(f"Refusing to export an incomplete dashboard: {rel}: {exc.detail}")
            print(f"  skip {rel} ({exc.detail})")

    put("health.json", server.health)
    put("forecast.json", server.forecast)
    put("sectors.json", server.sectors)
    put("performance.json", server.performance)
    put("equity.json", server.equity)
    put("news.json", server.news, optional=True)
    put("history/market.json", server.history, "market", None, HISTORY_LIMIT)
    for ticker in config.SECTOR_ETFS:
        put(f"history/sector/{ticker}.json", server.history, "sector", ticker, HISTORY_LIMIT,
            optional=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "frontend" / "public" / "api"))
    args = ap.parse_args()
    out = Path(args.out)
    written = export(out)
    print(f"exported {len(written)} files, {sum(written.values()) / 1e6:.2f} MB -> {out}")


if __name__ == "__main__":
    main()
