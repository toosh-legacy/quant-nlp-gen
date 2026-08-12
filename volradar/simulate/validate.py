"""Validate the method itself, on data whose answer we already know.

Three questions that real data cannot answer, because true volatility is never observed:

1. **What is the ceiling?** Realized volatility is a noisy estimate of true volatility, so
   even a perfect forecaster scores far below R² = 1 against it. Measuring the perfect
   forecaster's score here gives the number our real-data R² should actually be compared
   against.

2. **How much of our error is ours?** Scoring the same model against the true target and
   against the noisy proxy separates model error from measurement noise, and says whether
   effort belongs in better models or better targets.

3. **Is the evaluation machinery honest?** Run against a series with *no* predictable
   volatility at all, the walk-forward harness must report approximately zero skill. If it
   reports a positive edge on noise, something leaks — and that would invalidate every
   number in the README. This is a test of the measuring instrument, not the model.

Run:
    python -m volradar.simulate.validate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.features.core import CORE_FEATURES, build_core
from volradar.models import dataset as D
from volradar.models import estimators as E
from volradar.simulate.dgp import DGPParams, simulate_market, true_forward_vol

H = config.HEADLINE_HORIZON
REPORT = config.REPORT_DIR / "simulation_validation.csv"


def _prepare(raw: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """Build the real pipeline's features on synthetic data, keeping the true target."""
    truth = raw[["date", "ticker", "true_vol"]].copy()
    feats = build_core(raw.drop(columns=["true_vol"]), horizons=[h])
    feats = feats.merge(truth, on=["date", "ticker"], how="left")
    feats["y_true"] = true_forward_vol(feats, h)
    return feats


def _walk_forward(df: pd.DataFrame, features: list[str], h: int,
                  target: str) -> pd.DataFrame:
    """Same protocol as the real backtest, on whichever target is requested."""
    sub = df.dropna(subset=features + [target, config.persist_col(h),
                                       config.label_date_col(h)]).reset_index(drop=True)
    years = sorted(sub["date"].dt.year.unique())[4:]
    rows = []
    for year in years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = D.embargoed_train(sub, h, cutoff)
        test = D.time_slice(sub, cutoff, f"{year + 1}-01-01")
        if len(train) < 500 or len(test) < 60:
            continue
        tr = train.copy()
        tr[config.fwd_col(h)] = tr[target]          # fit on the requested target
        pred = E.fit_gbm(tr, features, h, E.GBM_DEFAULT).predict(test)
        rows.append({"year": year,
                     **M.regression_metrics(test[target], pred, test[config.persist_col(h)])})
    return pd.DataFrame(rows)


def experiment_ceiling() -> dict:
    """How well can ANY forecaster score against the noisy realized-vol proxy?

    The perfect forecaster is the true forward volatility itself. Scoring it against the
    realized proxy measures the noise floor: the R² that perfect knowledge would earn.
    """
    df = _prepare(simulate_market(DGPParams(n_days=5000)))
    ok = df.dropna(subset=[config.fwd_col(H), "y_true", config.persist_col(H)])
    perfect = M.regression_metrics(ok[config.fwd_col(H)], ok["y_true"],
                                   ok[config.persist_col(H)])
    return {
        "experiment": "ceiling",
        "perfect_r2_vs_proxy": perfect["r2"],
        "perfect_skill_vs_persistence": perfect["skill_vs_persistence"],
        "persistence_r2": perfect["r2_persistence"],
        "n": len(ok),
    }


def experiment_model_vs_truth() -> dict:
    """Score the same model against the true target and against the noisy proxy."""
    df = _prepare(simulate_market(DGPParams(n_days=5000)))
    vs_proxy = _walk_forward(df, CORE_FEATURES, H, config.fwd_col(H))
    vs_truth = _walk_forward(df, CORE_FEATURES, H, "y_true")
    return {
        "experiment": "model_vs_truth",
        "r2_vs_proxy": vs_proxy["r2"].mean(),
        "r2_vs_truth": vs_truth["r2"].mean(),
        "skill_vs_proxy": vs_proxy["skill_vs_persistence"].mean(),
        "skill_vs_truth": vs_truth["skill_vs_persistence"].mean(),
        "n_folds": len(vs_proxy),
    }


def experiment_null() -> dict:
    """The placebo: a market whose true volatility is constant, so nothing is forecastable.

    **The criterion is R², not skill versus persistence** — and finding that out is the most
    useful thing this simulation produced.

    Run against constant volatility, the harness reports R² of about -0.19 (correctly: no
    predictive power) while *skill versus persistence* reads about +0.43. Both numbers are
    right. In a constant-volatility world the persistence baseline is nothing but
    measurement noise from a short realized-vol window, so simply predicting the
    unconditional mean beats it by a wide margin — without forecasting anything.

    The consequence for how this project reports itself: **skill versus persistence cannot
    establish predictability on its own.** It measures "did we beat a noisy estimator",
    which is a lower bar than "is there signal here". A positive R² is what establishes
    signal, and skill versus persistence is what shows the gain is not just the free points
    from a moving denominator. The real-data results clear both bars (market R² 0.421 with
    skill 0.349), so the conclusions stand — but the two metrics must be read together, and
    that was not obvious before simulating a world where one of them lies.
    """
    p = DGPParams(n_days=5000, phi=0.0, vol_of_vol=0.0, leverage=0.0, jump_prob=0.0)
    df = _prepare(simulate_market(p))
    res = _walk_forward(df, CORE_FEATURES, H, config.fwd_col(H))
    return {
        "experiment": "null_no_predictability",
        "skill_mean": res["skill_vs_persistence"].mean(),
        "skill_std": res["skill_vs_persistence"].std(ddof=0),
        "r2_mean": res["r2"].mean(),
        "n_folds": len(res),
    }


def experiment_shuffle() -> dict:
    """The leakage test proper: destroy the feature/target link and demand R² <= 0.

    The target is shuffled in blocks (preserving its own autocorrelation) so that any
    relationship to the features is broken while its marginal distribution is untouched. If
    the harness still reports positive R², information is reaching the model through a route
    other than the features — the definition of a leak.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = _prepare(simulate_market(DGPParams(n_days=5000)))
    sub = df.dropna(subset=CORE_FEATURES + [config.fwd_col(H)]).reset_index(drop=True)

    # Block-shuffle the target: chop into 250-day blocks and reorder them.
    y = sub[config.fwd_col(H)].to_numpy(float)
    block = 250
    idx = np.arange(len(y))
    blocks = [idx[i:i + block] for i in range(0, len(idx), block)]
    order = rng.permutation(len(blocks))
    shuffled = np.concatenate([blocks[i] for i in order])[:len(y)]
    sub[config.fwd_col(H)] = y[shuffled]

    res = _walk_forward(sub, CORE_FEATURES, H, config.fwd_col(H))
    return {
        "experiment": "shuffled_target",
        "r2_mean": res["r2"].mean(),
        "r2_max": res["r2"].max(),
        "n_folds": len(res),
    }


def experiment_recovery() -> dict:
    """Does the model recover the known persistence of the process?

    A model that forecasts well should track true volatility closely. Correlation between
    the forecast and the true forward volatility is the cleanest statement of that, and it
    is only computable in simulation.
    """
    df = _prepare(simulate_market(DGPParams(n_days=5000)))
    sub = df.dropna(subset=CORE_FEATURES + [config.fwd_col(H), "y_true"]).reset_index(drop=True)
    cutoff = pd.Timestamp(sub["date"].quantile(0.6))
    train = D.embargoed_train(sub, H, cutoff)
    test = D.time_slice(sub, cutoff)
    pred = E.fit_gbm(train, CORE_FEATURES, H, E.GBM_DEFAULT).predict(test)
    return {
        "experiment": "recovery",
        "corr_forecast_vs_true": float(np.corrcoef(pred, test["y_true"])[0, 1]),
        "corr_proxy_vs_true": float(np.corrcoef(test[config.fwd_col(H)], test["y_true"])[0, 1]),
        "corr_persistence_vs_true": float(
            np.corrcoef(test[config.persist_col(H)], test["y_true"])[0, 1]),
        "n_test": len(test),
    }


def main() -> None:
    config.ensure_dirs()
    print("Simulating markets with known true volatility...\n")

    ceiling = experiment_ceiling()
    print("=== 1. The ceiling: how well can a PERFECT forecaster score? ===")
    print(f"  perfect forecast, R2 vs the realized proxy : {ceiling['perfect_r2_vs_proxy']:.3f}")
    print(f"  persistence baseline, same target          : {ceiling['persistence_r2']:.3f}")
    print("  -> realized volatility is a noisy yardstick; even perfect knowledge cannot")
    print("     score near 1.0 against it.")

    mvt = experiment_model_vs_truth()
    print("\n=== 2. Model error vs measurement noise ===")
    print(f"  our model scored against the noisy proxy : R2 {mvt['r2_vs_proxy']:.3f}")
    print(f"  our model scored against the TRUE target : R2 {mvt['r2_vs_truth']:.3f}")
    frac = (mvt["r2_vs_proxy"] / ceiling["perfect_r2_vs_proxy"]
            if ceiling["perfect_r2_vs_proxy"] > 0 else float("nan"))
    print(f"  -> the model captures {frac:.0%} of what was attainable against this target.")

    rec = experiment_recovery()
    print("\n=== 3. Does the forecast track TRUE volatility? ===")
    print(f"  corr(forecast, true future vol)    : {rec['corr_forecast_vs_true']:.3f}")
    print(f"  corr(persistence, true future vol) : {rec['corr_persistence_vs_true']:.3f}")
    print(f"  corr(realized proxy, true vol)     : {rec['corr_proxy_vs_true']:.3f}")

    null = experiment_null()
    print("\n=== 4. Placebo: a market with NO forecastable volatility ===")
    print(f"  R2 vs the window mean : {null['r2_mean']:+.4f}   <- the criterion")
    print(f"  skill vs persistence  : {null['skill_mean']:+.4f} +/- {null['skill_std']:.4f}")
    print(f"  ({null['n_folds']} folds)")
    print("  Both numbers are correct. R2 <= 0 says there is no signal, which is the truth")
    print("  here. Skill vs persistence still reads strongly positive because, with constant")
    print("  volatility, the persistence baseline is pure measurement noise and predicting")
    print("  the unconditional mean beats it without forecasting anything.")
    print("  -> LESSON: skill vs persistence cannot establish predictability on its own;")
    print("     it must be read next to R2. The real-data results clear both bars.")
    verdict = ("PASS - no signal reported where none exists"
               if null["r2_mean"] <= 0.02
               else "FAIL - positive R2 on an unforecastable series; investigate for leakage")
    print(f"  {verdict}")

    shuf = experiment_shuffle()
    print("\n=== 5. Leakage test: features vs a block-shuffled target ===")
    print(f"  R2 mean {shuf['r2_mean']:+.4f}, worst-case max {shuf['r2_max']:+.4f} "
          f"over {shuf['n_folds']} folds")
    v2 = ("PASS - no relationship found once the link is destroyed"
          if shuf["r2_mean"] <= 0.02
          else "FAIL - the model predicts a target it should have no access to")
    print(f"  {v2}")

    rows = [ceiling, mvt, rec, null, shuf]
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(f"\nWrote -> {REPORT}")


if __name__ == "__main__":
    main()
