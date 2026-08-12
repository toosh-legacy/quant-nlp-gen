"""Factor-family ablation: what does each block of features actually buy?

Two views, because they answer different questions and can disagree:

* **incremental** — core alone, then core + one family at a time. "Does this family carry
  signal the price history doesn't already have?"
* **leave-one-out** — everything, then everything minus one family. "Once the others are
  present, is this family still needed?" A family can look good incrementally and be
  worthless leave-one-out, which simply means another family already said it.

Scored by walk-forward R^2 mean across years, not a single split, because the differences
between these variants are small enough that one lucky window could invent one.
"""

from __future__ import annotations

import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.features.assemble import feature_families, load_panel
from volradar.models import dataset as D
from volradar.models import estimators as E


def _wf_score(df: pd.DataFrame, features: list[str], h: int) -> dict:
    """Mean walk-forward R^2 and direction accuracy for one feature set."""
    sub = D.labelled(df, features, h)
    fit = D.embargoed_train(sub, h, pd.Timestamp(config.VAL_START))
    val = D.time_slice(sub, config.VAL_START, config.VAL_END)
    params = (E.tune_gbm(fit, val, features, h, quiet=True)
              if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)

    folds = []
    for year in config.WALK_YEARS:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = D.embargoed_train(sub, h, cutoff)
        test = D.time_slice(sub, cutoff, f"{year + 1}-01-01")
        if len(train) < 400 or len(test) < 60:
            continue
        pred = E.fit_gbm(train, features, h, params).predict(test)
        y = test[config.fwd_col(h)]
        per = test[config.persist_col(h)]
        r = M.regression_metrics(y, pred, per)
        r.update(M.direction_metrics(y, pred, per))
        folds.append(r)
    f = pd.DataFrame(folds)
    return {"n_folds": len(f), "r2_mean": f["r2"].mean(), "r2_std": f["r2"].std(ddof=0),
            "dir_accuracy": f["dir_accuracy"].mean(),
            "n_features": len(features)}


def ablate(panel: str, h: int | None = None) -> pd.DataFrame:
    h = h or config.HEADLINE_HORIZON
    df = load_panel(panel)
    fams = feature_families(panel)
    core = fams["core"]
    others = [k for k in fams if k != "core"]
    rows = []

    print(f"\n=== Ablation: {panel} panel, h={h} (walk-forward mean R^2) ===")
    base = _wf_score(df, core, h)
    rows.append({"variant": "core only", "family": "core", "mode": "baseline", **base})
    print(f"  core only                 R2={base['r2_mean']:.3f}")

    for fam in others:
        feats = core + fams[fam]
        s = _wf_score(df, feats, h)
        rows.append({"variant": f"core + {fam}", "family": fam,
                     "mode": "incremental", "delta_r2": s["r2_mean"] - base["r2_mean"], **s})
        print(f"  core + {fam:<18} R2={s['r2_mean']:.3f}  "
              f"({s['r2_mean'] - base['r2_mean']:+.3f})")

    full_feats = [c for fam in fams.values() for c in fam]
    full = _wf_score(df, full_feats, h)
    rows.append({"variant": "all families", "family": "all", "mode": "full", **full})
    print(f"  all families              R2={full['r2_mean']:.3f}")

    for fam in others:
        feats = [c for k, v in fams.items() if k != fam for c in v]
        s = _wf_score(df, feats, h)
        rows.append({"variant": f"all minus {fam}", "family": fam, "mode": "leave_one_out",
                     "delta_r2": full["r2_mean"] - s["r2_mean"], **s})
        print(f"  all minus {fam:<15} R2={s['r2_mean']:.3f}  "
              f"(family worth {full['r2_mean'] - s['r2_mean']:+.3f})")

    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    out = pd.concat([ablate(p).assign(panel=p) for p in ("market", "sector")],
                    ignore_index=True)
    path = config.REPORT_DIR / "ablation.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote -> {path}")


if __name__ == "__main__":
    main()
