"""Monte Carlo scenario testing: one backtest is one sample.

The decision simulation on real data produced a single number — the forecast-sized strategy
beat the persistence-sized one by +0.046 Sharpe — and a block bootstrap said that gap was
not distinguishable from zero. A bootstrap resamples the one history that happened, so it
can only say how fragile that history was.

Simulation asks the sharper question: across many markets drawn from the *same* process,
how often does volatility targeting on a forecast beat targeting on a measurement, and by
how much? Because true volatility is known here, it also allows an arm that no backtest can
run — sizing on **perfect foresight** — which puts an upper bound on what the whole idea can
deliver. If even the oracle arm barely beats the baseline, the ceiling is the strategy, not
the forecast, and no amount of model work will fix it.

Run:
    python -m volradar.simulate.scenarios
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest.decision import COST_BPS, MAX_EXPOSURE, TARGET_VOL, _stats
from volradar.features.core import CORE_FEATURES, build_core
from volradar.models import dataset as D
from volradar.models import estimators as E
from volradar.simulate.dgp import DGPParams, simulate_market, true_forward_vol

H = config.HEADLINE_HORIZON
REPORT = config.REPORT_DIR / "simulation_scenarios.csv"
TRAIN_FRACTION = 0.5


def _exposure(vol_ann: np.ndarray) -> np.ndarray:
    raw = np.clip(TARGET_VOL / np.maximum(vol_ann, 1e-6), 0.0, MAX_EXPOSURE)
    return pd.Series(raw).rolling(5, min_periods=1).mean().to_numpy()


def _annualize(log_vol) -> np.ndarray:
    return np.exp(np.asarray(log_vol, float)) * np.sqrt(config.TRADING_DAYS) * 100.0


def run_one(seed: int, n_days: int = 4000) -> dict:
    """Simulate one market, fit on its first half, trade its second half."""
    raw = simulate_market(DGPParams(n_days=n_days, seed=seed))
    truth = raw[["date", "ticker", "true_vol"]]
    df = build_core(raw.drop(columns=["true_vol"]), horizons=[H])
    df = df.merge(truth, on=["date", "ticker"], how="left")
    df["y_true"] = true_forward_vol(df, H)

    sub = df.dropna(subset=CORE_FEATURES + [config.fwd_col(H), config.persist_col(H),
                                            "y_true", "ret"]).reset_index(drop=True)
    cut = sub["date"].iloc[int(len(sub) * TRAIN_FRACTION)]
    train = D.embargoed_train(sub, H, cut)
    test = D.time_slice(sub, cut).reset_index(drop=True)
    if len(train) < 500 or len(test) < 250:
        return {}

    pred = E.fit_gbm(train, CORE_FEATURES, H, E.GBM_DEFAULT).predict(test)
    fwd_ret = test["ret"].shift(-1).to_numpy(float)

    arms = {
        "buy_and_hold": np.ones(len(test)),
        "persistence_sized": _exposure(_annualize(test[config.persist_col(H)])),
        "model_sized": _exposure(_annualize(pred)),
        # Only computable in simulation: sizing on the volatility that actually occurs.
        "oracle_sized": _exposure(_annualize(test["y_true"])),
    }

    out = {"seed": seed, "n_test": len(test)}
    for name, exp in arms.items():
        cost = np.abs(np.diff(exp, prepend=exp[0])) * (COST_BPS / 1e4)
        st = _stats(exp * fwd_ret - cost, exp)
        for k in ("sharpe", "ann_vol", "max_drawdown", "vol_miss"):
            out[f"{name}_{k}"] = st.get(k, np.nan)
    return out


def run_scenarios(n_paths: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n_paths):
        r = run_one(seed=1000 + i)
        if r:
            rows.append(r)
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{n_paths} paths simulated")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = "persistence_sized"
    for arm in ["buy_and_hold", "persistence_sized", "model_sized", "oracle_sized"]:
        gap = df[f"{arm}_sharpe"] - df[f"{base}_sharpe"]
        rows.append({
            "arm": arm,
            "sharpe_mean": df[f"{arm}_sharpe"].mean(),
            "sharpe_std": df[f"{arm}_sharpe"].std(ddof=1),
            "vol_miss_mean": df[f"{arm}_vol_miss"].mean(),
            "max_dd_mean": df[f"{arm}_max_drawdown"].mean(),
            "sharpe_gap_vs_persistence": gap.mean(),
            "gap_ci_low": gap.quantile(0.025),
            "gap_ci_high": gap.quantile(0.975),
            "win_rate_vs_persistence": float((gap > 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    print(f"Monte Carlo: volatility targeting across simulated markets "
          f"(target {TARGET_VOL:.0f}%, {COST_BPS}bp costs)\n")
    paths = run_scenarios()
    if paths.empty:
        print("no usable paths"); return
    s = summarize(paths)

    print(f"\n=== {len(paths)} simulated markets ===")
    show = s.copy()
    for c in ["sharpe_mean", "sharpe_std", "vol_miss_mean", "sharpe_gap_vs_persistence",
              "gap_ci_low", "gap_ci_high"]:
        show[c] = show[c].map(lambda v: f"{v:+.3f}")
    show["max_dd_mean"] = show["max_dd_mean"].map(lambda v: f"{v:.1%}")
    show["win_rate_vs_persistence"] = show["win_rate_vs_persistence"].map(lambda v: f"{v:.0%}")
    print(show[["arm", "sharpe_mean", "sharpe_std", "vol_miss_mean", "max_dd_mean",
                "sharpe_gap_vs_persistence", "gap_ci_low", "gap_ci_high",
                "win_rate_vs_persistence"]].to_string(index=False))

    model = s[s["arm"] == "model_sized"].iloc[0]
    oracle = s[s["arm"] == "oracle_sized"].iloc[0]
    print(f"\n  forecast-sized beats measurement-sized in "
          f"{model['win_rate_vs_persistence']:.0%} of simulated markets "
          f"(mean gap {model['sharpe_gap_vs_persistence']:+.3f}, "
          f"95% range [{model['gap_ci_low']:+.3f}, {model['gap_ci_high']:+.3f}])")
    print(f"  PERFECT foresight would have earned {oracle['sharpe_gap_vs_persistence']:+.3f} "
          f"— the ceiling for this strategy.")
    if oracle["sharpe_gap_vs_persistence"] > 0:
        captured = model["sharpe_gap_vs_persistence"] / oracle["sharpe_gap_vs_persistence"]
        print(f"  the model captured {captured:.0%} of the attainable gain.")

    paths.to_csv(config.REPORT_DIR / "simulation_paths.csv", index=False)
    s.to_csv(REPORT, index=False)
    print(f"\nWrote -> {REPORT}")


if __name__ == "__main__":
    main()
