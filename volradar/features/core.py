"""Per-series volatility features and forecast targets.

Given a long OHLCV frame this builds the "own history" block: multi-scale realized
volatility in logs (the HAR decomposition), range-based estimators that extract intraday
information from daily bars, the leverage/downside effect, momentum and turnover.

**Causality contract.** Every feature on row `t` is a function of data up to and including
day `t` only. Every target is a function of days `t+1 .. t+h` only. `tests/test_causality.py`
enforces this by perturbing a future bar and asserting no past feature moves.

Volatility is modelled in **logs** throughout: realized vol is close to log-normal and far
more persistent in log space, which is why the HAR literature reports much higher R^2 there
than in raw vol space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config

EPS = config.EPS

# The own-history feature block, shared by the market and sector models.
CORE_FEATURES = [
    # Multi-scale close-to-close realized vol (Corsi's HAR components), in logs.
    "log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q",
    # Range-based estimators. Parkinson uses the high-low range; Garman-Klass adds the
    # open-close move. Both estimate a day's variance several times more efficiently than
    # a single squared return, so their trailing averages are cleaner predictors.
    "log_pk_d", "log_gk_d", "log_gk_w", "log_gk_m",
    # Term structure of the series' own vol: is short-horizon vol elevated vs long?
    "rv_slope_wm", "rv_slope_mq",
    # Leverage / downside effect: vol of negative-return days only.
    "log_semivol_m",
    # Momentum — drawdowns and vol travel together.
    "ret_5", "ret_21", "drawdown_63",
    # Turnover: log volume vs its own trailing distribution.
    "turnover_z",
]


# Jump-robust block, kept separate from CORE_FEATURES so previously reported results stay
# reproducible. Volatility has a continuous (diffusive) part and a discontinuous (jump)
# part, and they forecast very differently: the continuous part is highly persistent, while
# jumps are largely transitory. Andersen-Bollerslev-Diebold's HAR-CJ result is that
# separating them beats lumping them together, because a model fed total volatility
# over-extrapolates yesterday's one-off gap.
JUMP_FEATURES = ["log_bpv_m", "jump_share_m", "log_jump_d", "rq_ratio"]

# Measurement-error and signed-semivariance blocks, for the HARQ and SHAR models.
#
# `rq_sqrt` feeds HARQ; the semivariance block feeds SHAR.
#
# HARQ (Bollerslev, Patton & Quaedvlieg 2016, J. Econometrics): realized volatility is an
# *estimate* of true volatility, so regressing on it suffers attenuation bias — and the
# amount of attenuation varies with how noisy today's estimate is. Realized quarticity
# estimates that noise, so letting the daily coefficient depend on it makes the model
# responsive when the signal is clean and cautious when it is not.
#
# SHAR (Patton & Sheppard 2015): splitting realized variance into the parts contributed by
# up-days and down-days. Downside variance forecasts future volatility considerably better
# than upside variance — the leverage effect, measured directly rather than proxied.
SEMIVAR_FEATURES = ["log_rs_down_m", "log_rs_up_m", "semivar_skew_m"]


def _roll_rv(g: pd.core.groupby.generic.SeriesGroupBy, window: int) -> pd.Series:
    """Root-mean-square of a squared-return column over a trailing window, per ticker."""
    return np.sqrt(g.transform(lambda s: s.rolling(window).mean()))


def add_jump_features(out: pd.DataFrame) -> pd.DataFrame:
    """Bipower variation and the jump/continuous split, from daily returns.

    Bipower variation estimates the *continuous* part of variance using products of
    adjacent absolute returns: a single large jump inflates one term but is damped by its
    small neighbour, so BPV survives jumps that wreck realized variance. The gap
    RV - BPV is the jump component.

    Standard scaling constant mu1^-2 = (pi/2) for the adjacent-absolute-return estimator.
    """
    df = out
    def gr(col: str):
        return df.groupby("ticker", observed=True)[col]

    absr = df["ret"].abs()
    df["_bp"] = (np.pi / 2.0) * absr * gr("ret").transform(lambda s: s.abs().shift(1))

    bpv_m = gr("_bp").transform(lambda s: s.rolling(21).mean())
    rv_m = df.groupby("ticker", observed=True)["_r2"].transform(lambda s: s.rolling(21).mean())

    df["log_bpv_m"] = np.log(np.sqrt(bpv_m.clip(lower=EPS)) + EPS)
    # Share of monthly variance attributable to jumps, floored at zero (the estimator can
    # go slightly negative on noise).
    df["jump_share_m"] = ((rv_m - bpv_m).clip(lower=0.0) / rv_m.clip(lower=EPS)).clip(0, 1)
    # Today's own jump size.
    df["log_jump_d"] = np.log(np.sqrt((df["_r2"] - df["_bp"]).clip(lower=0.0)) + EPS)
    # Realized quarticity ratio — how heavy-tailed the recent return distribution is.
    # High values mean the realized-vol estimate itself is noisy and should be trusted less.
    rq = df.groupby("ticker", observed=True)["_r4"].transform(lambda s: s.rolling(21).mean())
    df["rq_ratio"] = np.log(rq.clip(lower=EPS)) - 2.0 * np.log(rv_m.clip(lower=EPS))

    # --- HARQ measurement-error term ------------------------------------------------
    # Realized quarticity estimates the variance of the realized-variance estimator. The
    # (n/3) scaling is the standard estimator; with daily returns the "n" is the rolling
    # window. HARQ uses its square root, which is the natural scale of a standard error.
    rq_raw = (21.0 / 3.0) * df.groupby("ticker", observed=True)["_r4"].transform(
        lambda s: s.rolling(21).mean())
    df["rq_sqrt"] = np.sqrt(np.sqrt(rq_raw.clip(lower=0.0)))

    # --- SHAR signed semivariance ----------------------------------------------------
    # Realized variance split by the sign of the return that produced it. Down-variance is
    # the component that carries the forecasting power.
    df["_r2_up"] = np.where(df["ret"] > 0, df["_r2"], 0.0)
    rs_down = df.groupby("ticker", observed=True)["_r2_down"].transform(
        lambda s: s.rolling(21).mean())
    rs_up = df.groupby("ticker", observed=True)["_r2_up"].transform(
        lambda s: s.rolling(21).mean())
    df["log_rs_down_m"] = np.log(np.sqrt(rs_down.clip(lower=0.0)) + EPS)
    df["log_rs_up_m"] = np.log(np.sqrt(rs_up.clip(lower=0.0)) + EPS)
    # Asymmetry: how much more of the variance came from falling days than rising ones.
    df["semivar_skew_m"] = df["log_rs_down_m"] - df["log_rs_up_m"]
    return df


def build_core(df: pd.DataFrame, horizons: list[int] | None = None,
               with_jumps: bool = True) -> pd.DataFrame:
    """Add core volatility features and forward log-RV targets to a long OHLCV frame.

    Parameters
    ----------
    df : long frame with columns date, ticker, open, high, low, close, volume
    horizons : forecast horizons in trading days (defaults to config.HORIZONS)
    with_jumps : also compute the jump-robust block (JUMP_FEATURES)
    """
    horizons = horizons or config.HORIZONS
    out = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()

    def gr(col: str):
        return out.groupby("ticker", observed=True)[col]

    out["ret"] = gr("close").transform(lambda s: s.pct_change())
    out["_r2"] = out["ret"] ** 2
    out["_r4"] = out["ret"] ** 4

    # --- Multi-scale realized vol (daily, weekly, monthly, quarterly look-backs) -------
    out["log_rv_d"] = np.log(out["ret"].abs() + EPS)
    for name, w in [("w", 5), ("m", 21), ("q", 63)]:
        out[f"log_rv_{name}"] = np.log(_roll_rv(gr("_r2"), w) + EPS)

    # --- Range-based daily variance ---------------------------------------------------
    hl = np.log(out["high"] / out["low"])
    co = np.log(out["close"] / out["open"])
    pk_var = (hl ** 2) / (4.0 * np.log(2.0))
    gk_var = (0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2).clip(lower=EPS)
    out["_gk_var"] = gk_var
    out["log_pk_d"] = np.log(np.sqrt(pk_var) + EPS)
    out["log_gk_d"] = np.log(np.sqrt(gk_var) + EPS)
    out["log_gk_w"] = np.log(_roll_rv(gr("_gk_var"), 5) + EPS)
    out["log_gk_m"] = np.log(_roll_rv(gr("_gk_var"), 21) + EPS)

    # --- Own-vol term structure (differences of logs = log ratios) --------------------
    out["rv_slope_wm"] = out["log_rv_w"] - out["log_rv_m"]
    out["rv_slope_mq"] = out["log_rv_m"] - out["log_rv_q"]

    # --- Downside semivariance --------------------------------------------------------
    out["_r2_down"] = np.where(out["ret"] < 0, out["_r2"], 0.0)
    out["log_semivol_m"] = np.log(_roll_rv(gr("_r2_down"), 21) + EPS)

    # --- Momentum & drawdown ----------------------------------------------------------
    out["ret_5"] = gr("close").transform(lambda s: s.pct_change(5))
    out["ret_21"] = gr("close").transform(lambda s: s.pct_change(21))
    roll_max = gr("close").transform(lambda s: s.rolling(63).max())
    out["drawdown_63"] = out["close"] / roll_max - 1.0

    # --- Turnover ---------------------------------------------------------------------
    out["_logv"] = np.log(out["volume"].replace(0, np.nan).astype(float) + 1.0)
    mu = gr("_logv").transform(lambda s: s.rolling(21).mean())
    sd = gr("_logv").transform(lambda s: s.rolling(21).std())
    out["turnover_z"] = ((out["_logv"] - mu) / sd).replace([np.inf, -np.inf], np.nan)

    # --- Targets: log realized vol over the NEXT h days, plus the persistence baseline -
    for h in horizons:
        fwd = _roll_rv(gr("_r2"), h).groupby(out["ticker"], observed=True).shift(-h)
        out[config.fwd_col(h)] = np.log(fwd + EPS)
        out[config.persist_col(h)] = np.log(_roll_rv(gr("_r2"), h) + EPS)
        out[config.label_date_col(h)] = gr("date").transform(lambda s: s.shift(-h))

    if with_jumps:
        out = add_jump_features(out)

    return out.drop(columns=[c for c in out.columns if c.startswith("_")])

