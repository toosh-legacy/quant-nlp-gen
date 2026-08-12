"""Settle the three open questions, under the rules already in force.

Each was left explicitly open in the README, so each is decided here with the same protocol
that decided everything else: pre-test folds only, paired against the incumbent fold by
fold, and checked at two fold granularities — because a result that appears at one calendar
slicing and vanishes at another is not a result (that is how the jump features were caught).

1. **ridge vs gbm on the market panel.** Ridge beat the shipped GBM in both the fixed split
   and the walk-forward, but at t=+1.45, below the adoption bar. Settle it.
2. **QLIKE-trained GBM.** Squared error is symmetric; the forecast's known failure modes are
   asymmetric. Does training on the volatility-native loss fix them?
3. **Quantile-trained GBM.** Same motivation, blunter instrument: forecast a high quantile
   instead of the mean.

Models 2 and 3 are deliberately NOT judged on R². A quantile forecast does not estimate the
mean, so R² would condemn it by construction; they are judged on QLIKE, which is the loss a
risk user actually cares about, with skill reported alongside for context.

Run:
    python -m volradar.backtest.settle
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.backtest.methods import fold_edges
from volradar.features.assemble import all_features, fit_and_eval_frames
from volradar.models import dataset as D
from volradar.models import estimators as E

PRE_TEST_YEARS = [y for y in config.WALK_YEARS if y < int(config.TEST_START[:4])]
REPORT = config.REPORT_DIR / "settled.csv"

CANDIDATES = {
    "gbm (incumbent)": lambda tr, f, h, p: E.fit_gbm(tr, f, h, p),
    "ridge": lambda tr, f, h, p: E.fit_ridge(tr, f, h),
    "gbm-qlike": lambda tr, f, h, p: E.fit_gbm_qlike(tr, f, h, p),
    "gbm-q60": lambda tr, f, h, p: E.fit_gbm_quantile(tr, f, h, p, alpha=0.60),
    "gbm-q70": lambda tr, f, h, p: E.fit_gbm_quantile(tr, f, h, p, alpha=0.70),
    "ridge+gbm": None,        # handled specially: average of the two
}


def _per_fold(panel: str, h: int, years: list[int], per_year: int) -> pd.DataFrame:
    feats = all_features(panel)
    tr_sub, ev_sub = fit_and_eval_frames(panel, feats, h)

    from volradar.backtest.evaluate import _tuned_params
    params = _tuned_params(tr_sub, feats, h, quiet=True, val_df=ev_sub)

    rows = []
    min_test = max(60 // per_year, 30)
    for cutoff, end in fold_edges(years, per_year):
        train = D.embargoed_train(tr_sub, h, cutoff)
        test = D.time_slice(ev_sub, cutoff, end)
        if len(train) < 400 or len(test) < min_test:
            continue
        y = test[config.fwd_col(h)]
        per = test[config.persist_col(h)]

        preds = {}
        for name, fn in CANDIDATES.items():
            if fn is None:
                continue
            preds[name] = fn(train, feats, h, params).predict(test)
        preds["ridge+gbm"] = 0.5 * preds["ridge"] + 0.5 * preds["gbm (incumbent)"]

        for name, p in preds.items():
            r = M.regression_metrics(y, p, per)
            r.update(M.direction_metrics(y, p, per))
            rows.append({"fold": str(cutoff.date()), "model": name, **r})
    return pd.DataFrame(rows)


def _paired(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
    base = df[df["model"] == "gbm (incumbent)"].set_index("fold")[metric]
    out = []
    for name, g in df.groupby("model"):
        if name == "gbm (incumbent)":
            continue
        s = g.set_index("fold")[metric]
        common = base.index.intersection(s.index)
        d = (s[common] - base[common]) * (1 if higher_is_better else -1)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 2 else np.nan
        out.append({"model": name, "n": len(d), "delta": d.mean(),
                    "t": d.mean() / se if se and se > 0 else np.nan,
                    "wins": int((d > 0).sum())})
    return pd.DataFrame(out)


def settle(panel: str, h: int | None = None) -> pd.DataFrame:
    h = h or config.HEADLINE_HORIZON
    print(f"\n{'=' * 74}\n=== Settling: {panel} panel, h={h}, pre-test folds only\n{'=' * 74}")
    all_rows = []
    for per_year, label in [(1, "annual"), (2, "semi-annual")]:
        folds = _per_fold(panel, h, PRE_TEST_YEARS, per_year)
        if folds.empty:
            continue
        means = folds.groupby("model")[["skill_vs_persistence", "qlike", "r2",
                                        "dir_accuracy"]].mean()
        print(f"\n--- {label} folds ---")
        print(means.round(4).to_string())

        for metric, hib in [("skill_vs_persistence", True), ("qlike", False)]:
            pv = _paired(folds, metric, hib)
            print(f"  paired on {metric} (t>2 adopts):")
            for _, r in pv.iterrows():
                flag = "ADOPT" if (r["t"] > 2 and r["delta"] > 0) else "hold "
                print(f"    {flag} {r['model']:<16} delta {r['delta']:+.4f} "
                      f"t={r['t']:+.2f}  wins {r['wins']}/{r['n']}")
            pv["metric"], pv["folds"] = metric, label
            pv["panel"], pv["horizon"] = panel, h
            all_rows.append(pv)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main() -> None:
    config.ensure_dirs()
    out = pd.concat([settle(p) for p in ("market", "sector")], ignore_index=True)
    out.to_csv(REPORT, index=False)
    print(f"\nWrote -> {REPORT}")

    print("\n=== Verdict (adopt only if t>2 on BOTH fold granularities) ===")
    for (panel, model, metric), g in out.groupby(["panel", "model", "metric"]):
        if len(g) < 2:
            continue
        if (g["t"] > 2).all() and (g["delta"] > 0).all():
            print(f"  ADOPT  {panel}/{model} on {metric}: "
                  f"t = {', '.join(f'{v:+.2f}' for v in g['t'])}")


if __name__ == "__main__":
    main()
