"""TSLA volatility forecasting — the result that should actually be strong, and
the direct test of "does intraday data buy a better forecast?".

Two experiments, both reusing the universe run's escalating-baseline design
(persistence -> HAR-OLS -> Ridge(all) -> XGBoost) and its log-realized-vol R^2.

  A. FULL-HISTORY, close-to-close target (2015-2026): the apples-to-apples
     comparison against the 80-stock universe run (R^2 ~ 0.42). Fixed OOS split +
     yearly walk-forward, plus the "will vol rise?" classification.

  B. INTRADAY LEVER (hourly window, ~2 years): on the same recent sessions, we
     cross the TARGET (close-to-close `cc` vs intraday `iv` realized vol) with the
     FEATURE set (daily-only vs daily+intraday). Skill over each target's own
     persistence baseline is the honest, comparable quantity (R^2 levels across
     different targets are not directly comparable — different denominators).

Prior expectation (before the numbers): a single ticker has far less data than
380k universe ticker-days and no cross-sectional feature, so the full-history R^2
may well land BELOW 0.42 even though vol is predictable. Intraday data may add
skill for the intraday target; whether it beats plain daily HAR is exactly the
open question. Reported straight either way.

Run:
    python tesla/volatility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from tesla.features import (  # noqa: E402
    HAR_CC_FEATURES, VOL_DAILY_FEATURES, VOL_INTRADAY_FEATURES, build_features,
    cc_fwd_col, cc_persist_col, iv_fwd_col, iv_persist_col,
)

FIXED_TEST_START = "2022-01-01"       # full-history OOS split
WALK_YEARS = list(range(2018, 2027))
# Right-sized for a SINGLE stock's ~1,700 training rows. The universe run used 500
# deep trees on 380k rows; reused here they overfit badly (test R^2 collapses to ~0
# / negative). A shallow, small ensemble is the fair configuration for this sample
# size — verified by a size sweep (150x depth-2 ~ 0.13-0.21 vs 500x depth-4 ~ 0.00).
XGB_FIXED = dict(subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                 random_state=config.RANDOM_SEED, n_jobs=-1, tree_method="hist")
XGB_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.05)


def _r2(y, p) -> float:
    from sklearn.metrics import r2_score
    return float(r2_score(y, p))


def _fit_ridge(Xtr, ytr):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    return sc, Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)


def _fit_xgb(Xtr, ytr):
    from xgboost import XGBRegressor
    m = XGBRegressor(**XGB_FIXED, **XGB_PARAMS)
    m.fit(Xtr, ytr)
    return m


def _embargoed_train(df: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    ld = config.label_date_col(h)
    return df[(df["date"] < cutoff) & df[ld].notna() & (df[ld] < cutoff)]


# ---------------------------------------------------------------------------
# A. Full-history close-to-close volatility (vs the universe R^2 ~ 0.42)
# ---------------------------------------------------------------------------
def full_history_regression(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(FIXED_TEST_START)
    rows = []
    for h in config.TSLA_VOL_HORIZONS:
        y, per = cc_fwd_col(h), cc_persist_col(h)
        sub = df[df[y].notna()]
        train = _embargoed_train(sub, h, cutoff)
        test = sub[sub["date"] >= cutoff]
        Xtr, ytr = train[VOL_DAILY_FEATURES].to_numpy(float), train[y].to_numpy()
        Xte, yte = test[VOL_DAILY_FEATURES].to_numpy(float), test[y].to_numpy()

        rows.append({"horizon": h, "model": "persistence", "r2": _r2(yte, test[per].to_numpy())})
        sc, har = _fit_ridge(train[HAR_CC_FEATURES].to_numpy(float), ytr)
        rows.append({"horizon": h, "model": "HAR-OLS",
                     "r2": _r2(yte, har.predict(sc.transform(test[HAR_CC_FEATURES].to_numpy(float))))})
        sc2, ridge = _fit_ridge(Xtr, ytr)
        rows.append({"horizon": h, "model": "ridge(all)", "r2": _r2(yte, ridge.predict(sc2.transform(Xte)))})
        rows.append({"horizon": h, "model": "xgboost", "r2": _r2(yte, _fit_xgb(Xtr, ytr).predict(Xte))})
    return pd.DataFrame(rows)


def full_history_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in config.TSLA_VOL_HORIZONS:
        y = cc_fwd_col(h)
        sub = df[df[y].notna()]
        r2s = []
        for yr in WALK_YEARS:
            cutoff, nxt = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr + 1}-01-01")
            train = _embargoed_train(sub, h, cutoff)
            test = sub[(sub["date"] >= cutoff) & (sub["date"] < nxt)]
            if len(train) < 400 or len(test) < 60:
                continue
            m = _fit_xgb(train[VOL_DAILY_FEATURES].to_numpy(float), train[y].to_numpy())
            r2s.append(_r2(test[y].to_numpy(), m.predict(test[VOL_DAILY_FEATURES].to_numpy(float))))
        if r2s:
            rows.append({"horizon": h, "n_folds": len(r2s), "r2_mean": float(np.mean(r2s)),
                         "r2_std": float(np.std(r2s)), "r2_min": float(np.min(r2s)),
                         "r2_max": float(np.max(r2s))})
    return pd.DataFrame(rows)


def vol_rise_classification(df: pd.DataFrame) -> pd.DataFrame:
    """'Will vol rise?' — forward log-vol > current log-vol. Fixed split, XGBoost.
    Comparable to the universe run's 72% / MCC 0.45."""
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    from xgboost import XGBClassifier

    cutoff = pd.Timestamp(FIXED_TEST_START)
    rows = []
    for h in config.TSLA_VOL_HORIZONS:
        y, per = cc_fwd_col(h), cc_persist_col(h)
        sub = df[df[y].notna()].copy()
        sub["cls"] = (sub[y] > sub[per]).astype(int)
        train = _embargoed_train(sub, h, cutoff)
        test = sub[sub["date"] >= cutoff]
        clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=2.0, random_state=config.RANDOM_SEED,
                            n_jobs=-1, tree_method="hist", eval_metric="logloss")
        clf.fit(train[VOL_DAILY_FEATURES].to_numpy(float), train["cls"].to_numpy())
        pred = clf.predict(test[VOL_DAILY_FEATURES].to_numpy(float))
        base = float(max(test["cls"].mean(), 1 - test["cls"].mean()))
        rows.append({"horizon": h, "baseline": base,
                     "accuracy": float(accuracy_score(test["cls"], pred)),
                     "mcc": float(matthews_corrcoef(test["cls"], pred))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. Intraday lever — target (cc vs iv) x features (daily vs +intraday)
# ---------------------------------------------------------------------------
def intraday_lever(df: pd.DataFrame) -> pd.DataFrame:
    """On the hourly window, a chronological 70/30 split. For each (target, feature
    set) report XGBoost R^2 and its skill over the target's own persistence
    baseline (the comparable quantity across targets)."""
    rows = []
    for h in config.TSLA_VOL_HORIZONS:
        # Rows where BOTH targets + intraday features are defined (fair common sample).
        need = [cc_fwd_col(h), iv_fwd_col(h), *VOL_INTRADAY_FEATURES]
        sub = df.dropna(subset=need).sort_values("date").reset_index(drop=True)
        if len(sub) < 200:
            rows.append({"horizon": h, "note": f"too few intraday rows ({len(sub)})"})
            continue
        cut = int(len(sub) * 0.70)
        train, test = sub.iloc[:cut], sub.iloc[cut:]
        for tgt, per, tname in [(cc_fwd_col(h), cc_persist_col(h), "cc"),
                                (iv_fwd_col(h), iv_persist_col(h), "iv")]:
            base_r2 = _r2(test[tgt].to_numpy(), test[per].to_numpy())
            for feats, fname in [(VOL_DAILY_FEATURES, "daily"),
                                 (VOL_INTRADAY_FEATURES, "daily+intraday")]:
                m = _fit_xgb(train[feats].to_numpy(float), train[tgt].to_numpy())
                r2 = _r2(test[tgt].to_numpy(), m.predict(test[feats].to_numpy(float)))
                rows.append({"horizon": h, "target": tname, "features": fname,
                             "n_train": len(train), "n_test": len(test),
                             "persistence_r2": base_r2, "xgb_r2": r2,
                             "skill_over_persist": r2 - base_r2})
    return pd.DataFrame(rows)


def _fmt(df, cols):
    show = df.copy()
    for c in cols:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    return show.to_string(index=False)


def main() -> None:
    df = build_features()
    print(f"TSLA volatility: {len(df)} rows, {df['date'].min().date()}..{df['date'].max().date()}\n")

    print("=== A. Full-history close-to-close log-RV R^2, fixed OOS split (train<2022, test 22-26) ===")
    ft = full_history_regression(df)
    piv = ft.pivot(index="horizon", columns="model", values="r2")[
        ["persistence", "HAR-OLS", "ridge(all)", "xgboost"]].reset_index()
    print(_fmt(piv, ["persistence", "HAR-OLS", "ridge(all)", "xgboost"]))
    print("  (compare xgboost column against the 80-stock universe run: R^2 ~ 0.37/0.42/0.41)")

    print("\n=== A. Full-history walk-forward (XGBoost) — R^2 per-year, mean +/- std ===")
    wf = full_history_walk_forward(df)
    for _, r in wf.iterrows():
        print(f"  h={int(r['horizon']):>2} ({int(r['n_folds'])} folds): R^2 "
              f"{r['r2_mean']:.3f} +/- {r['r2_std']:.3f} (range {r['r2_min']:.3f}..{r['r2_max']:.3f})")

    print("\n=== A. 'Will vol rise?' classification (fixed split, XGBoost) ===")
    print(_fmt(vol_rise_classification(df), ["baseline", "accuracy", "mcc"]))

    print("\n=== B. Intraday lever: target (cc/iv) x features (daily/+intraday), 70/30 split ===")
    print("  skill_over_persist is the comparable number (R^2 minus that target's persistence R^2)")
    il = intraday_lever(df)
    if "xgb_r2" in il.columns:
        print(_fmt(il[["horizon", "target", "features", "n_test", "persistence_r2",
                       "xgb_r2", "skill_over_persist"]],
                   ["persistence_r2", "xgb_r2", "skill_over_persist"]))
    else:
        print(il.to_string(index=False))

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ft.to_csv(config.TSLA_VOLATILITY_RESULTS, index=False)
    print(f"\nWrote full-history vol regression -> {config.TSLA_VOLATILITY_RESULTS}")


if __name__ == "__main__":
    main()
