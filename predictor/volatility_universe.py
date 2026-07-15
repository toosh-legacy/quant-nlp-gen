"""Task B (pro) — evaluate the best volatility forecaster on the large 2005-2024 universe.

Predicts LOG realized volatility over the next h days. Reports R^2 (the fraction of log-vol
variance explained) against escalating baselines:

  * persistence  — next log-vol = current log-vol (the standard, and a strong, baseline),
  * HAR-OLS      — Ridge on the 4 multi-scale HAR components (the classic model),
  * Ridge (all)  — linear model on the full feature set incl. cross-sectional signal,
  * XGBoost      — gradient-boosted trees.

Two evaluations:
  * a fixed out-of-sample split (train < 2018, test 2018-2024), and
  * a walk-forward across each year 2015-2024 (train = all prior years, embargoed), giving a
    mean +/- std R^2 across regimes.

A "will volatility rise?" classification (log-vol vs its current level) is reported too.

Run:
    python predictor/volatility_universe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from features.universe_data import (  # noqa: E402
    UNIV_FEATURES, UNIV_HORIZONS, build_universe_features, univ_fwd_col, univ_persist_col,
)

HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q"]
FIXED_TEST_START = "2018-01-01"
WALK_YEARS = list(range(2015, 2025))


def _r2(y, p) -> float:
    from sklearn.metrics import r2_score
    return float(r2_score(y, p))


def _fit_ridge(Xtr, ytr):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    return sc, m


def _fit_xgb(Xtr, ytr):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, reg_lambda=1.0, random_state=config.RANDOM_SEED,
                     n_jobs=-1, tree_method="hist")
    m.fit(Xtr, ytr)
    return m


def _embargoed_train(df: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows before `cutoff` whose h-day outcome also lands before it (purge the boundary)."""
    ld = f"label_date_{h}"
    return df[(df["date"] < cutoff) & (df[ld].notna()) & (df[ld] < cutoff)]


def fixed_split_table(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(FIXED_TEST_START)
    rows = []
    for h in UNIV_HORIZONS:
        y, per = univ_fwd_col(h), univ_persist_col(h)
        sub = df[df[y].notna()]
        train = _embargoed_train(sub, h, cutoff)
        test = sub[sub["date"] >= cutoff]
        Xtr, ytr = train[UNIV_FEATURES].to_numpy(float), train[y].to_numpy()
        Xte, yte = test[UNIV_FEATURES].to_numpy(float), test[y].to_numpy()

        rows.append({"horizon": h, "model": "persistence", "r2": _r2(yte, test[per].to_numpy())})

        sc, har = _fit_ridge(train[HAR_COLS].to_numpy(float), ytr)
        rows.append({"horizon": h, "model": "HAR-OLS",
                     "r2": _r2(yte, har.predict(sc.transform(test[HAR_COLS].to_numpy(float))))})

        sc2, ridge = _fit_ridge(Xtr, ytr)
        rows.append({"horizon": h, "model": "ridge(all)", "r2": _r2(yte, ridge.predict(sc2.transform(Xte)))})

        xgb = _fit_xgb(Xtr, ytr)
        rows.append({"horizon": h, "model": "xgboost", "r2": _r2(yte, xgb.predict(Xte))})
        print(f"  h={h:>2}: fixed split done (train={len(train)}, test={len(test)})")
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year walk-forward R^2 (XGBoost, all features): train on all prior years, embargoed."""
    per_fold = []
    for h in UNIV_HORIZONS:
        y = univ_fwd_col(h)
        sub = df[df[y].notna()]
        for yr in WALK_YEARS:
            cutoff = pd.Timestamp(f"{yr}-01-01")
            nxt = pd.Timestamp(f"{yr + 1}-01-01")
            train = _embargoed_train(sub, h, cutoff)
            test = sub[(sub["date"] >= cutoff) & (sub["date"] < nxt)]
            if len(train) < 5000 or len(test) < 500:
                continue
            xgb = _fit_xgb(train[UNIV_FEATURES].to_numpy(float), train[y].to_numpy())
            r2 = _r2(test[y].to_numpy(), xgb.predict(test[UNIV_FEATURES].to_numpy(float)))
            per_fold.append({"horizon": h, "year": yr, "n_test": len(test), "r2": r2})
    pf = pd.DataFrame(per_fold)
    out = []
    for h in UNIV_HORIZONS:
        s = pf[pf["horizon"] == h]
        out.append({"horizon": h, "n_folds": len(s),
                    "r2_mean": s["r2"].mean(), "r2_std": s["r2"].std(ddof=0),
                    "r2_min": s["r2"].min(), "r2_max": s["r2"].max()})
    return pd.DataFrame(out), pf


def classification_table(df: pd.DataFrame) -> pd.DataFrame:
    """'Will vol rise?' — log next-h-day vol > current log vol. Fixed split, XGBoost."""
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    from xgboost import XGBClassifier

    cutoff = pd.Timestamp(FIXED_TEST_START)
    rows = []
    for h in UNIV_HORIZONS:
        y, per = univ_fwd_col(h), univ_persist_col(h)
        sub = df[df[y].notna()].copy()
        sub["cls"] = (sub[y] > sub[per]).astype(int)
        train = _embargoed_train(sub, h, cutoff)
        test = sub[sub["date"] >= cutoff]
        clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, random_state=config.RANDOM_SEED,
                            n_jobs=-1, tree_method="hist", eval_metric="logloss")
        clf.fit(train[UNIV_FEATURES].to_numpy(float), train["cls"].to_numpy())
        pred = clf.predict(test[UNIV_FEATURES].to_numpy(float))
        base = max(test["cls"].mean(), 1 - test["cls"].mean())
        rows.append({"horizon": h, "baseline": base,
                     "accuracy": accuracy_score(test["cls"], pred),
                     "mcc": matthews_corrcoef(test["cls"], pred)})
    return pd.DataFrame(rows)


def _fmt(df, cols):
    show = df.copy()
    for c in cols:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    return show.to_string(index=False)


def main() -> None:
    df = build_universe_features()
    print(f"Universe features: {len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()}..{df['date'].max().date()}\n")

    print("=== Regression: log realized-vol R^2, fixed OOS split (train<2018, test 2018-24) ===")
    ft = fixed_split_table(df)
    piv = ft.pivot(index="horizon", columns="model", values="r2")[
        ["persistence", "HAR-OLS", "ridge(all)", "xgboost"]].reset_index()
    print(_fmt(piv, ["persistence", "HAR-OLS", "ridge(all)", "xgboost"]))

    print("\n=== Regression walk-forward (XGBoost) — R^2 per-year, mean +/- std ===")
    wf, _ = walk_forward(df)
    for _, r in wf.iterrows():
        print(f"  h={int(r['horizon']):>2}  ({int(r['n_folds'])} yearly folds):  "
              f"R^2 {r['r2_mean']:.3f} +/- {r['r2_std']:.3f}  "
              f"(range {r['r2_min']:.3f}..{r['r2_max']:.3f})")

    print("\n=== Classification: 'will vol rise?' fixed split (XGBoost) ===")
    ct = classification_table(df)
    print(_fmt(ct, ["baseline", "accuracy", "mcc"]))

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ft.to_csv(config.DATA_DIR / "universe_vol_regression.csv", index=False)
    print(f"\nWrote -> {config.DATA_DIR / 'universe_vol_regression.csv'}")


if __name__ == "__main__":
    main()
