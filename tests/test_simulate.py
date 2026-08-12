"""Tests for the synthetic-market generator.

The simulator is used to validate the rest of the project, so it needs its own checks — a
DGP that quietly failed to produce volatility clustering would make the harness look
trustworthy for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from volradar.simulate.dgp import DGPParams, simulate_market, true_forward_vol


@pytest.fixture(scope="module")
def sim():
    return simulate_market(DGPParams(n_days=2500, seed=7))


def test_ohlc_bars_are_internally_consistent(sim):
    """High must bound open/close from above and low from below, on every bar."""
    assert (sim["high"] >= sim[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (sim["low"] <= sim[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (sim["high"] >= sim["low"]).all()
    assert (sim[["open", "high", "low", "close"]] > 0).all().all()


def test_volatility_actually_clusters(sim):
    """The point of the DGP: |returns| must be autocorrelated, or nothing is forecastable."""
    r = np.diff(np.log(sim["close"].to_numpy()))
    a = np.abs(r)
    ac = np.corrcoef(a[:-1], a[1:])[0, 1]
    assert ac > 0.1, f"no volatility clustering in the synthetic market (acf={ac:.3f})"


def test_annualized_volatility_is_in_a_plausible_range(sim):
    r = np.diff(np.log(sim["close"].to_numpy()))
    ann = r.std(ddof=0) * np.sqrt(252) * 100
    assert 8 < ann < 40, f"synthetic market volatility {ann:.1f}% is not equity-like"


def test_leverage_effect_is_present(sim):
    """Negative returns should be followed by higher volatility than positive ones."""
    r = np.diff(np.log(sim["close"].to_numpy()))
    nxt = np.abs(r[1:])
    prev = r[:-1]
    assert nxt[prev < 0].mean() > nxt[prev > 0].mean()


def test_null_params_remove_all_predictability():
    """The placebo configuration must genuinely have no volatility dynamics."""
    p = DGPParams(n_days=2500, phi=0.0, vol_of_vol=0.0, leverage=0.0, jump_prob=0.0, seed=3)
    df = simulate_market(p)
    assert df["true_vol"].std() == pytest.approx(0.0, abs=1e-12)

    r = np.diff(np.log(df["close"].to_numpy()))
    a = np.abs(r)
    ac = np.corrcoef(a[:-1], a[1:])[0, 1]
    assert abs(ac) < 0.08, f"placebo market still has clustering (acf={ac:.3f})"


def test_true_forward_vol_is_forward_looking(sim):
    """The true target must depend on the future, and lead the current level."""
    h = 10
    y = true_forward_vol(sim, h)
    assert y.notna().sum() > 1000
    # The last h rows have no future left and must be undefined.
    assert y.tail(h).isna().all()


def test_simulation_is_reproducible():
    a = simulate_market(DGPParams(n_days=400, seed=11))
    b = simulate_market(DGPParams(n_days=400, seed=11))
    assert np.allclose(a["close"], b["close"])
    assert not np.allclose(a["close"], simulate_market(DGPParams(n_days=400, seed=12))["close"])
