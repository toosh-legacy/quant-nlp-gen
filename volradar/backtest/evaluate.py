"""The headline evaluation: fixed out-of-sample split and per-year walk-forward.

Both panels, all horizons, the full model ladder. This is the module whose output the
README quotes and the dashboard displays, so it writes machine-readable CSV/JSON to
`data/reports/` rather than only printing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.features.assemble import all_features, fit_and_eval_frames, load_panel
from volradar.models import estimators as E
from volradar.models import dataset as D


def _tuned_params(df: pd.DataFrame, features: list[str], h: int, quiet=False,
                  val_df: pd.DataFrame | None = None) -> dict:
    """Tune GBM on a chronological validation slice inside the training period."""
    fit = D.embargoed_train(df, h, pd.Timestamp(config.VAL_START))
    val = D.time_slice(val_df if val_df is not None else df,
                       config.VAL_START, config.VAL_END)
    if len(fit) < 200 or len(val) < 100:
        return E.GBM_DEFAULT
    return E.tune_gbm(fit, val, features, h, quiet=quiet)


def fixed_split(panel: str, quiet: bool = False) -> pd.DataFrame:
    """Train before TEST_START, score everything after. One row per (horizon, model)."""
    df = load_panel(panel)
    features = all_features(panel)
    cutoff = pd.Timestamp(config.TEST_START)
    rows = []

    for h in config.HORIZONS:
        tr_sub, sub = fit_and_eval_frames(panel, features, h)
        train = D.embargoed_train(tr_sub, h, cutoff)
        test = D.time_slice(sub, cutoff)
        params = _tuned_params(tr_sub, features, h, quiet=quiet, val_df=sub)
        y_test = test[config.fwd_col(h)].to_numpy(float)
        persist = test[config.persist_col(h)].to_numpy(float)

        for name, _ in E.LADDER:
            pred = E.fit_by_name(name, train, features, h, params).predict(test)
            row = {"panel": panel, "horizon": h, "model": name,
                   "n_train": len(train), "n_test": len(test)}
            row.update(M.regression_metrics(y_test, pred, persist))
            row.update(M.direction_metrics(y_test, pred, persist))
            # From the panel's own pre-cutoff rows, not the (possibly pooled) train frame.
            thr = D.regime_thresholds(D.embargoed_train(sub, h, cutoff), h)
            row.update(M.regime_metrics(D.to_regime(y_test, thr), D.to_regime(pred, thr)))
            rows.append(row)

    return pd.DataFrame(rows)


def walk_forward(panel: str, model_name: str = "gbm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain at the start of every year on all prior data; score that year alone.

    This is the number to trust. A fixed split gives one window that might have been a
    kind regime; thirteen consecutive out-of-sample years, each with a model that saw
    nothing after its own cutoff, gives a mean and a spread.
    """
    df = load_panel(panel)
    features = all_features(panel)
    folds = []

    for h in config.HORIZONS:
        tr_sub, sub = fit_and_eval_frames(panel, features, h)
        params = _tuned_params(tr_sub, features, h, quiet=True, val_df=sub)
        for year in config.WALK_YEARS:
            cutoff = pd.Timestamp(f"{year}-01-01")
            train = D.embargoed_train(tr_sub, h, cutoff)
            test = D.time_slice(sub, cutoff, f"{year + 1}-01-01")
            # Minimum sizes so a half-empty first fold cannot dominate the mean.
            if len(train) < 400 or len(test) < 60:
                continue
            pred = E.fit_by_name(model_name, train, features, h, params).predict(test)
            y = test[config.fwd_col(h)].to_numpy(float)
            per = test[config.persist_col(h)].to_numpy(float)
            fold = {"panel": panel, "horizon": h, "year": year, "model": model_name,
                    "n_train": len(train), "n_test": len(test)}
            fold.update(M.regression_metrics(y, pred, per))
            fold.update(M.direction_metrics(y, pred, per))
            folds.append(fold)

    fold_df = pd.DataFrame(folds)
    summary = M.summarize_folds(fold_df, ["panel", "horizon", "model"], "r2")
    return summary, fold_df


def pooled_oos(panel: str) -> pd.DataFrame:
    """Score the concatenated walk-forward predictions as one series — the headline number.

    Why this and not the average of per-year R^2: R^2 measures variance explained relative
    to the *mean of the scoring window*. Scored year by year, that denominator is whatever
    happened to vary within that year — 0.07 in calm 2022 versus 0.42 in 2020 — so a model
    can forecast well and still post a negative per-year R^2 simply because it could not
    have known that year's average level in advance. Nobody experiences one year in
    isolation, so the honest denominator is the whole out-of-sample period.

    This is not a softer test. Every prediction in the series still comes from a model that
    saw nothing after its own year's cutoff; only the scoring window changed. The per-year
    spread is still reported alongside, and the metric that is immune to the denominator
    problem — skill versus persistence, where both models face the same window — is the
    stability check.
    """
    from volradar.backtest.oos import build_oos

    rows = []
    for h in config.HORIZONS:
        oos = build_oos(panel, h)
        y = oos[config.fwd_col(h)]
        per = oos[config.persist_col(h)]
        row = {"panel": panel, "horizon": h, "n_oos": len(oos),
               "oos_start": str(oos["date"].min().date()),
               "oos_end": str(oos["date"].max().date()),
               "served": h in config.PRODUCT_HORIZONS}
        row.update(M.regression_metrics(y, oos["pred"], per))
        row.update(M.direction_metrics(y, oos["pred"], per))
        row.update(M.regime_metrics(oos["true_regime"], oos["pred_regime"]))
        rows.append(row)
    return pd.DataFrame(rows)


def per_sector_walk_forward() -> pd.DataFrame:
    """Walk-forward R^2 broken out by sector — the honesty check on the sector product.

    A good pooled number can hide a sector the model simply cannot forecast, and the
    dashboard ranks sectors individually, so each one has to stand on its own.
    """
    df = load_panel("sector")
    features = all_features("sector")
    h = config.HEADLINE_HORIZON
    tr_sub, sub = fit_and_eval_frames("sector", features, h)
    params = _tuned_params(tr_sub, features, h, quiet=True, val_df=sub)

    rows = []
    for year in config.WALK_YEARS:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = D.embargoed_train(tr_sub, h, cutoff)
        test = D.time_slice(sub, cutoff, f"{year + 1}-01-01")
        if len(train) < 400 or len(test) < 60:
            continue
        model = E.fit_gbm(train, features, h, params)
        test = test.copy()
        test["pred"] = model.predict(test)
        for ticker, g in test.groupby("ticker"):
            if len(g) < 40:
                continue
            m = M.regression_metrics(g[config.fwd_col(h)], g["pred"], g[config.persist_col(h)])
            m.update(M.direction_metrics(g[config.fwd_col(h)], g["pred"], g[config.persist_col(h)]))
            rows.append({"ticker": ticker, "sector": config.SECTOR_ETFS.get(ticker, ticker),
                         "year": year, **m})
    folds = pd.DataFrame(rows)
    summary = M.summarize_folds(folds, ["ticker", "sector"], "r2")
    extra = folds.groupby(["ticker"])[["dir_accuracy", "skill_vs_persistence"]].mean()
    return summary.merge(extra, on="ticker", how="left")


def _print_table(df: pd.DataFrame, cols: list[str], title: str) -> None:
    print(f"\n=== {title} ===")
    show = df[cols].copy()
    for c in cols:
        if show[c].dtype.kind == "f":
            show[c] = show[c].map(lambda v: f"{v: .3f}")
    print(show.to_string(index=False))


def main() -> None:
    config.ensure_dirs()
    all_fixed, all_summary, all_folds, all_pooled = [], [], [], []

    for panel in ("market", "sector"):
        print(f"\n{'#' * 70}\n# PANEL: {panel}\n{'#' * 70}")
        fixed = fixed_split(panel)
        all_fixed.append(fixed)
        _print_table(fixed, ["horizon", "model", "r2", "skill_vs_persistence",
                             "dir_accuracy", "dir_baseline", "dir_mcc"],
                     f"{panel}: fixed OOS split (train < {config.TEST_START}, "
                     f"test {config.TEST_START[:4]}-2024)")

        pooled = pooled_oos(panel)
        all_pooled.append(pooled)
        _print_table(pooled, ["horizon", "served", "n_oos", "r2", "r2_persistence",
                              "skill_vs_persistence", "dir_accuracy", "dir_mcc",
                              "regime_accuracy", "regime_baseline"],
                     f"{panel}: HEADLINE — pooled walk-forward OOS "
                     f"({config.WALK_YEARS[0]}-{config.WALK_YEARS[-1]})")

        summary, folds = walk_forward(panel)
        all_summary.append(summary)
        all_folds.append(folds)
        _print_table(summary, ["horizon", "n_folds", "r2_mean", "r2_std",
                               "r2_min", "r2_max"],
                     f"{panel}: per-year R^2 spread (denominator varies by year — "
                     f"see pooled table)")

        stab = folds.groupby("horizon")["skill_vs_persistence"].agg(
            mean="mean", std="std", min="min",
            years_positive=lambda s: int((s > 0).sum()), n="count").reset_index()
        _print_table(stab, ["horizon", "mean", "std", "min", "years_positive", "n"],
                     f"{panel}: STABILITY — skill vs persistence, per year "
                     f"(same denominator both sides)")

    fixed_df = pd.concat(all_fixed, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    folds_df = pd.concat(all_folds, ignore_index=True)
    pooled_df = pd.concat(all_pooled, ignore_index=True)
    pooled_df.to_csv(config.REPORT_DIR / "pooled_oos.csv", index=False)

    sec = per_sector_walk_forward()
    _print_table(sec.sort_values("skill_vs_persistence", ascending=False),
                 ["sector", "n_folds", "skill_vs_persistence", "dir_accuracy",
                  "r2_mean", "r2_std"],
                 "per-sector walk-forward (h=10, gbm) — ranked by skill vs persistence")

    fixed_df.to_csv(config.REPORT_DIR / "fixed_split.csv", index=False)
    summary_df.to_csv(config.REPORT_DIR / "walkforward_summary.csv", index=False)
    folds_df.to_csv(config.REPORT_DIR / "walkforward_folds.csv", index=False)
    sec.to_csv(config.REPORT_DIR / "per_sector.csv", index=False)
    print(f"\nWrote 4 reports -> {config.REPORT_DIR}")


if __name__ == "__main__":
    main()
