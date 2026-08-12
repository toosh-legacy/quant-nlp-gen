"""Decision-utility simulation: would using the forecast have helped?

R^2 says the forecast is accurate. This says whether accuracy converts into a better
outcome, using the most standard real application of a volatility forecast:
**volatility targeting**. Hold exposure inversely proportional to expected volatility, so
the portfolio runs at a roughly constant risk level instead of a constant dollar level.

Three arms, identical except for how exposure is set:

* **buy & hold**  — always 100% invested. The thing to beat.
* **persistence** — size on *current* realized vol. The honest baseline: it uses a
  volatility estimate, just not a forecast. Beating buy & hold is easy; beating this is
  the actual test of whether *forecasting* added anything over *measuring*.
* **model**       — size on the walk-forward forecast.

Rules fixed in advance to keep it fair: exposure is capped, the forecast at day *t* is
applied to day *t+1*'s return (no same-day execution), and both sized arms pay the same
proportional cost on every exposure change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest.oos import annualize, load_or_build

TARGET_VOL = 15.0        # annualized % — the risk level the strategy tries to hold
MAX_EXPOSURE = 2.0       # leverage cap; sizing never exceeds this
COST_BPS = 1.0           # one-way cost, in basis points of the exposure traded


def _exposure(vol_ann: np.ndarray) -> np.ndarray:
    """Target vol / expected vol, capped. Smoothed over 5 days so the strategy is not
    whipsawed by a single day's forecast — a real desk would not re-lever daily on noise."""
    raw = np.clip(TARGET_VOL / np.maximum(vol_ann, 1e-6), 0.0, MAX_EXPOSURE)
    return pd.Series(raw).rolling(5, min_periods=1).mean().to_numpy()


def _stats(rets: np.ndarray, exposure: np.ndarray | None = None) -> dict:
    r = np.asarray(rets, float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {}
    ann_ret = float((1 + r).prod() ** (config.TRADING_DAYS / len(r)) - 1)
    ann_vol = float(r.std(ddof=0) * np.sqrt(config.TRADING_DAYS))
    equity = (1 + r).cumprod()
    dd = float((equity / np.maximum.accumulate(equity) - 1).min())
    out = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "max_drawdown": dd,
        "vol_miss": abs(ann_vol * 100 - TARGET_VOL),   # distance from the risk target
        "worst_day": float(r.min()),
    }
    if exposure is not None:
        e = np.asarray(exposure, float)
        out["avg_exposure"] = float(np.nanmean(e))
        out["turnover_ann"] = float(np.nansum(np.abs(np.diff(e))) /
                                    len(e) * config.TRADING_DAYS)
    return out


def simulate() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_or_build("market").sort_values("date").reset_index(drop=True)
    h = int(df["horizon"].iloc[0])

    # The return earned AFTER the decision is made. Shifting here is the only place
    # execution timing is expressed, so it can be audited in one line.
    fwd_ret = df["ret"].shift(-1).to_numpy(float)

    pred_vol = annualize(df["pred"])
    persist_vol = annualize(df[config.persist_col(h)])

    arms = {
        "buy_and_hold": np.ones(len(df)),
        "persistence_sized": _exposure(persist_vol),
        "model_sized": _exposure(pred_vol),
    }

    rows, series = [], {"date": df["date"]}
    for name, exp in arms.items():
        gross = exp * fwd_ret
        cost = np.abs(np.diff(exp, prepend=exp[0])) * (COST_BPS / 1e4)
        net = gross - cost
        series[f"{name}_ret"] = net
        series[f"{name}_exposure"] = exp
        rows.append({"strategy": name, **_stats(net, exp)})

    table = pd.DataFrame(rows)
    # Everything is judged against the realistic baseline, not the trivial one.
    base = table.loc[table["strategy"] == "persistence_sized"].iloc[0]
    table["sharpe_vs_persistence"] = table["sharpe"] - base["sharpe"]
    table["vol_miss_vs_persistence"] = base["vol_miss"] - table["vol_miss"]

    curves = pd.DataFrame(series)
    for name in arms:
        curves[f"{name}_equity"] = (1 + curves[f"{name}_ret"].fillna(0)).cumprod()
    return table, curves


def bootstrap_sharpe_gap(curves: pd.DataFrame, n_boot: int = 2000,
                         block: int = 21) -> dict:
    """Is the model arm's Sharpe advantage over persistence-sizing real, or noise?

    A single backtest produces one number, and Sharpe differences of a few hundredths are
    routinely produced by luck. This resamples the paired daily returns in overlapping
    21-day blocks — long enough to preserve volatility clustering, which is the entire
    phenomenon being modelled and would be destroyed by naive i.i.d. resampling — and
    reports the distribution of the Sharpe gap.

    If the confidence interval spans zero, the honest statement is that the decision-utility
    benefit was not established, regardless of the point estimate.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    a = curves["model_sized_ret"].to_numpy(float)
    b = curves["persistence_sized_ret"].to_numpy(float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    n_blocks = int(np.ceil(n / block))

    def sharpe(r):
        sd = r.std(ddof=0)
        return float(r.mean() / sd * np.sqrt(config.TRADING_DAYS)) if sd > 0 else 0.0

    gaps = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        gaps[i] = sharpe(a[idx]) - sharpe(b[idx])

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "sharpe_gap": sharpe(a) - sharpe(b),
        "ci_low": float(lo), "ci_high": float(hi),
        "p_gap_positive": float((gaps > 0).mean()),
        "significant": bool(lo > 0),
        "n_boot": n_boot, "block_days": block,
    }


def main() -> None:
    config.ensure_dirs()
    table, curves = simulate()
    print(f"\n=== Decision utility: volatility targeting at {TARGET_VOL:.0f}% "
          f"annualized, OOS ===")
    print(f"  {curves['date'].min().date()}..{curves['date'].max().date()}  "
          f"cost {COST_BPS}bp/trade  max exposure {MAX_EXPOSURE}x")
    show = table.copy()
    for c in ["ann_return", "ann_vol", "max_drawdown", "avg_exposure"]:
        show[c] = show[c].map(lambda v: f"{v: .1%}")
    for c in ["sharpe", "sharpe_vs_persistence", "vol_miss", "turnover_ann"]:
        show[c] = show[c].map(lambda v: f"{v: .2f}")
    print(show[["strategy", "ann_return", "ann_vol", "sharpe", "max_drawdown",
                "vol_miss", "avg_exposure", "turnover_ann",
                "sharpe_vs_persistence"]].to_string(index=False))

    boot = bootstrap_sharpe_gap(curves)
    verdict = ("REAL - the interval excludes zero" if boot["significant"]
               else "NOT ESTABLISHED - the interval spans zero")
    print(f"\n=== Is the Sharpe advantage over persistence-sizing real? ===")
    print(f"  Sharpe gap        : {boot['sharpe_gap']:+.3f}")
    print(f"  95% CI (bootstrap): [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]  "
          f"({boot['n_boot']} resamples, {boot['block_days']}-day blocks)")
    print(f"  P(gap > 0)        : {boot['p_gap_positive']:.1%}")
    print(f"  Verdict           : {verdict}")

    table.to_csv(config.REPORT_DIR / "decision.csv", index=False)
    pd.DataFrame([boot]).to_csv(config.REPORT_DIR / "decision_significance.csv", index=False)
    curves.to_parquet(config.REPORT_DIR / "decision_curves.parquet", index=False)
    print(f"\nWrote -> {config.REPORT_DIR / 'decision.csv'}")


if __name__ == "__main__":
    main()
