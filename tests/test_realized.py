"""Tests for the Oxford-Man Realized Library loader and the new training objectives.

The loader is cached to disk, so these tests skip themselves when the cache is absent
rather than downloading during CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volradar import config
from volradar.models import estimators as E

CACHE = config.RAW_DIR / "realized_library.parquet"
needs_cache = pytest.mark.skipif(
    not CACHE.exists(), reason="run `python -m volradar.data.realized` first")


# ---------------------------------------------------------------------------
# The realized library
# ---------------------------------------------------------------------------
@needs_cache
def test_measures_survive_parsing():
    """Trailing spaces in the source headers silently dropped bipower variation once."""
    from volradar.data.realized import load_realized_library

    df = load_realized_library()
    for col in ["rv5", "rv5_ss", "rv10", "bpv5", "rsv5", "rk", "medrv", "close"]:
        assert col in df.columns, f"{col} missing from the parsed library"
        assert df[col].notna().any(), f"{col} parsed but is entirely empty"


@needs_cache
def test_variance_measures_are_non_negative_and_plausible():
    from volradar.data.realized import SPX, load_symbol

    spx = load_symbol(SPX).dropna(subset=["rv5"])
    assert (spx["rv5"] >= 0).all()
    # Annualized from daily variance; the S&P's long-run vol is ~15-20%.
    ann = float(np.sqrt(spx["rv5"].mean() * 252) * 100)
    assert 10 < ann < 30, f"implausible S&P realized volatility {ann:.1f}%"


@needs_cache
def test_reconstructed_high_low_bracket_open_and_close():
    from volradar.data.realized import SPX, load_symbol

    s = load_symbol(SPX).dropna(subset=["open", "close", "hl_range"])
    assert (s["high"] >= s[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (s["low"] <= s[["open", "close"]].min(axis=1) + 1e-9).all()


# ---------------------------------------------------------------------------
# The new training objectives
# ---------------------------------------------------------------------------
def test_qlike_objective_gradient_is_zero_at_the_optimum():
    """QLIKE is minimised where prediction equals truth, so the gradient vanishes there."""
    y = np.array([-4.0, -3.5, -3.0])
    grad, hess = E.qlike_objective(y, y.copy())
    assert np.allclose(grad, 0.0, atol=1e-9)
    assert (hess > 0).all()


def test_qlike_gradient_signs_push_toward_the_truth():
    y = np.full(3, -4.0)
    over = y + 0.5      # forecast too high -> gradient should be positive (reduce p)
    under = y - 0.5     # forecast too low  -> gradient should be negative (raise p)
    assert (E.qlike_objective(y, over)[0] > 0).all()
    assert (E.qlike_objective(y, under)[0] < 0).all()


def test_qlike_hessian_is_floored():
    """Without a floor the Hessian decays to zero and boosting's step size explodes.

    This is not hypothetical: the unfloored version diverged to a loss of ~5e8.
    """
    y = np.full(5, -4.0)
    wildly_over = y + 10.0
    _, hess = E.qlike_objective(y, wildly_over)
    assert (hess >= 0.5).all()
    assert np.isfinite(hess).all()


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = -4.0 + 0.5 * x[:, 0] + 0.2 * rng.normal(size=n)
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    df[config.fwd_col(10)] = y
    return df, ["a", "b", "c"]


def test_qlike_trained_model_produces_sane_predictions():
    """Guards the divergence bug: predictions must stay in the target's neighbourhood."""
    df, feats = _toy()
    pred = E.fit_gbm_qlike(df, feats, 10).predict(df)
    assert np.isfinite(pred).all()
    assert -8 < pred.mean() < 0, f"predictions ran away (mean {pred.mean():.2f})"


def test_quantile_model_forecasts_above_the_mean_model():
    """A 70th-percentile fit should sit above a mean fit — that is its whole purpose."""
    df, feats = _toy()
    mean_pred = E.fit_gbm(df, feats, 10).predict(df)
    q70 = E.fit_gbm_quantile(df, feats, 10, alpha=0.70).predict(df)
    assert q70.mean() > mean_pred.mean()
