"""API contract tests.

The dashboard trusts these shapes. They also encode the design rule that the API computes
nothing: if a report file is absent the endpoint must degrade with a clear message rather
than invent a number.

Tests that need generated artifacts skip themselves when the pipeline has not been run, so
CI stays green without network access or a compute budget.
"""

from __future__ import annotations

import json

import pytest

from volradar import config

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient      # noqa: E402

from volradar.api.server import app           # noqa: E402

client = TestClient(app)

HAVE_SNAPSHOT = (config.REPORT_DIR / "snapshot.json").exists()
HAVE_REPORTS = (config.REPORT_DIR / "pooled_oos.csv").exists()
needs_snapshot = pytest.mark.skipif(not HAVE_SNAPSHOT, reason="run `volradar.cli train` first")
needs_reports = pytest.mark.skipif(not HAVE_REPORTS, reason="run `volradar.cli evaluate` first")


def test_health_always_answers():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "artifacts" in body and isinstance(body["artifacts"], dict)


@needs_snapshot
def test_forecast_shape():
    f = client.get("/api/forecast").json()
    for key in ["as_of", "current_vol_ann", "forecast_vol_ann", "regime", "direction",
                "regime_bands_ann", "drivers"]:
        assert key in f, f"missing {key}"
    assert f["regime"] in config.REGIME_NAMES
    assert f["direction"] in ("rising", "falling")
    assert 0 < f["forecast_vol_ann"] < 300, "annualized vol outside any plausible range"


@needs_snapshot
def test_forecast_direction_agrees_with_the_levels():
    """The arrow must match the numbers it sits next to."""
    f = client.get("/api/forecast").json()
    rising = f["forecast_vol_ann"] > f["current_vol_ann"]
    assert (f["direction"] == "rising") == rising


@needs_snapshot
def test_sectors_are_ranked_most_stable_first():
    s = client.get("/api/sectors").json()
    ranking = s["ranking"]
    assert len(ranking) >= 9
    vols = [r["forecast_vol_ann"] for r in ranking]
    assert vols == sorted(vols), "ranking is not ordered by forecast volatility"
    assert [r["stability_rank"] for r in ranking] == list(range(1, len(ranking) + 1))


@needs_reports
def test_performance_exposes_the_withheld_horizon():
    """The unstable horizon must be declared, not silently absent."""
    p = client.get("/api/performance").json()
    assert p["horizons_served"] == config.PRODUCT_HORIZONS
    assert p["horizons_withheld"] == config.UNSTABLE_HORIZONS
    assert set(p["horizons_served"]) & set(p["horizons_withheld"]) == set()


@needs_reports
def test_history_is_json_safe():
    """NaN is not valid JSON and silently breaks the frontend, so it must never appear."""
    r = client.get("/api/history?panel=market&limit=200")
    assert r.status_code == 200
    raw = r.content.decode()
    assert "NaN" not in raw and "Infinity" not in raw
    series = json.loads(raw)["series"]
    assert len(series) > 0
    assert all({"date", "forecast", "realized"} <= set(p) for p in series)


def test_missing_artifact_returns_503_not_a_guess():
    """A missing report must produce an explicit error, never a fabricated default."""
    r = client.get("/api/history?panel=nonexistent")
    assert r.status_code in (404, 503)
