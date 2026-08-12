"""Synthetic markets with a KNOWN true volatility.

On real data, true volatility is unobservable. Every metric in this project is computed
against *realized* volatility, which is only a noisy estimate of it — so a persistent
question has no answer on real data: **how much of our forecast error is the model's fault,
and how much is irreducible noise in the yardstick?** An R² of 0.42 could be a mediocre
model against a clean target or an excellent model against a filthy one, and the two call
for completely different work.

Simulating the market answers it. Here the data-generating process is written down, so
sigma_t is known exactly. The same pipeline can then be scored twice — against the true
volatility and against the realized proxy — and the gap between those two numbers is the
noise ceiling.

The process is a discretised log-stochastic-volatility model with the three features that
matter for this project, each switchable so its individual contribution can be isolated:

* **persistence** — log-variance follows an AR(1) with a half-life of weeks, which is what
  makes volatility forecastable at all;
* **leverage** — negative returns raise future volatility more than positive ones do;
* **jumps** — occasional large moves that are not predictable from the state.

Defaults are calibrated to look like the S&P 500: ~16% annualized volatility, strong
clustering, fat tails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class DGPParams:
    """Parameters of the synthetic market. Defaults are S&P-500-like."""
    n_days: int = 5000
    # Log-variance AR(1): x_t = mu + phi (x_{t-1} - mu) + leverage*shock + vol_of_vol*eps
    mu_vol: float = 0.16 / np.sqrt(TRADING_DAYS)   # long-run daily volatility
    phi: float = 0.985                              # persistence (half-life ~46 days)
    vol_of_vol: float = 0.12
    leverage: float = -0.09         # negative return -> higher future vol
    jump_prob: float = 0.006        # ~1.5 jumps a year
    jump_scale: float = 3.5         # jump size in units of current daily vol
    drift: float = 0.07 / TRADING_DAYS
    seed: int = 42

    # Intraday shape, used to build plausible OHLC bars from the daily path.
    n_intraday: int = 78            # 5-minute bars in a session


def simulate_market(p: DGPParams | None = None) -> pd.DataFrame:
    """Generate one synthetic price series with its true volatility recorded.

    Returns a frame in the same shape the real pipeline consumes (date, ticker, OHLCV) plus
    a `true_vol` column that only the validation code is allowed to look at.
    """
    p = p or DGPParams()
    rng = np.random.default_rng(p.seed)
    n = p.n_days

    log_mu = np.log(p.mu_vol)
    x = np.empty(n)
    x[0] = log_mu
    ret = np.empty(n)
    true_vol = np.empty(n)

    for t in range(n):
        true_vol[t] = np.exp(x[t])
        z = rng.standard_normal()
        r = p.drift + true_vol[t] * z
        # Jumps are drawn independently of the state — genuinely unpredictable, which is
        # what makes the event study on real data hard and should make it hard here too.
        if rng.random() < p.jump_prob:
            r += true_vol[t] * p.jump_scale * rng.standard_normal()
        ret[t] = r
        if t + 1 < n:
            shock = p.vol_of_vol * rng.standard_normal()
            # Leverage enters through the standardised return, so its effect does not
            # depend on the current volatility level.
            x[t + 1] = log_mu + p.phi * (x[t] - log_mu) + p.leverage * z + shock

    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[100.0], close[:-1]])

    # Build High/Low from an intraday Brownian bridge at the true volatility, so that the
    # range-based estimators the model relies on are as informative here as in real data.
    steps = rng.standard_normal((n, p.n_intraday)) * (true_vol / np.sqrt(p.n_intraday))[:, None]
    path = np.cumsum(steps, axis=1)
    path = path - path[:, [-1]] + (np.log(close) - np.log(open_))[:, None]
    high = open_ * np.exp(np.maximum(path.max(axis=1), 0.0))
    low = open_ * np.exp(np.minimum(path.min(axis=1), 0.0))

    volume = 1e6 * np.exp(rng.standard_normal(n) * 0.3 + 2.0 * (true_vol / p.mu_vol - 1.0))

    return pd.DataFrame({
        "date": pd.bdate_range("2005-01-03", periods=n),
        "ticker": "SIM",
        "open": open_, "high": np.maximum(high, close), "low": np.minimum(low, close),
        "close": close, "volume": volume,
        "true_vol": true_vol,
    })



def true_forward_vol(df: pd.DataFrame, h: int) -> pd.Series:
    """The TRUE average volatility over the next h days — the unobservable target.

    This is what a perfect forecaster would predict. Comparing a model against this instead
    of against realized volatility separates model error from proxy noise.
    """
    tv = df.groupby("ticker", observed=True)["true_vol"]
    fwd = tv.transform(lambda s: np.sqrt((s ** 2).rolling(h).mean()).shift(-h))
    return np.log(fwd + 1e-8)
