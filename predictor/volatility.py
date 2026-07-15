"""Task B — volatility-regime prediction: "will volatility rise or fall from here?"

Direction is near-random, but **realized volatility is strongly autocorrelated and
mean-reverting** (volatility clustering / the ARCH effect): calm follows calm, turbulent
follows turbulent, and extreme levels tend to revert. So predicting the *direction of
volatility* over the next h trading days is genuinely learnable — honestly ~0.70+ accuracy
at short horizons with a strongly positive MCC, no tricks. It is also a real quant task
(risk and options desks care far more about volatility than price direction).

Label: for horizon h, 1 if the next-h-day realized volatility (config.fwd_vol_col(h)) is
GREATER than the current h-day realized volatility (config.current_vol_col(h)), else 0 —
i.e. "will volatility rise?". Comparing to the current level (known at t) keeps the two
classes roughly balanced in every period, so accuracy is a fair, interpretable metric and a
result above ~0.50 is unambiguous skill. The current level is a feature, so the model can
learn mean-reversion; the forward level appears only in the label, never as a feature. The
same overlapping-window embargo as the direction task applies (the outcome window is the same
t..t+h span, so we reuse label_split_col(h)).

Run:
    python predictor/volatility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from predictor.evaluate import score  # noqa: E402
from predictor.train_predictor import (  # noqa: E402
    MIN_DEV_ROWS, XGB_DEFAULT, _fit_logreg, _fit_xgboost, load_combined,
    split_frame, tune_xgboost,
)

VOL_FEATURE_SETS: dict[str, list[str]] = {
    "price_only": config.VOL_PRICE_FEATURE_COLS,
    "combined": config.VOL_COMBINED_FEATURE_COLS,
}

# Quarterly walk-forward windows, matching predictor/walkforward.py.
FOLD_STARTS = ["2014-10-01", "2015-01-01", "2015-04-01", "2015-07-01", "2015-10-01"]
FOLD_LENGTH = pd.Timedelta(days=91)


def vol_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Embargoed rows for the volatility task at horizon h, with an integer target `y` = 1
    iff next-h-day realized vol exceeds the current h-day realized vol ("will vol rise?").

    Embargo: train/dev rows kept only if the outcome day is in the same split (purge);
    test rows kept whenever the label is defined (outcome may run past the window).
    """
    fv, cur = config.fwd_vol_col(h), config.current_vol_col(h)
    label_defined = df[fv].notna() & df[cur].notna()
    same_split = df["split"] == df[config.label_split_col(h)]
    is_test = df["split"] == "test"
    frame = df[label_defined & (same_split | is_test)].copy()
    # The longer vol look-backs (volatility_20d) are NaN for the earliest rows of a ticker;
    # drop those so the linear model has complete feature rows.
    frame = frame.dropna(subset=config.VOL_PRICE_FEATURE_COLS)
    frame["y"] = (frame[fv] > frame[cur]).astype(int)
    return frame.reset_index(drop=True)


def _train_and_eval(frame: pd.DataFrame, cols: list[str]) -> tuple[dict, dict, str]:
    """Fit LogReg + (dev-tuned) XGBoost on this frame's train split; score both on test."""
    from sklearn.preprocessing import StandardScaler

    train, dev, test = (split_frame(frame, s) for s in ("train", "dev", "test"))
    y_train, y_test = train["y"].to_numpy(), test["y"].to_numpy()

    scaler = StandardScaler().fit(train[cols].to_numpy(dtype=float))
    logreg = _fit_logreg(scaler.transform(train[cols].to_numpy(dtype=float)), y_train)
    lr_pred = logreg.predict(scaler.transform(test[cols].to_numpy(dtype=float)))

    if len(dev) >= MIN_DEV_ROWS:
        params, _ = tune_xgboost(train, dev, cols)
        note = "tuned"
    else:
        params, note = XGB_DEFAULT, "defaults"
    xgb = _fit_xgboost(train[cols].to_numpy(dtype=float), y_train, params)
    xgb_pred = xgb.predict(test[cols].to_numpy(dtype=float))

    return score(y_test, lr_pred), score(y_test, xgb_pred), note


def fixed_window_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in config.VOL_HORIZONS:
        frame = vol_frame(df, h)
        test = split_frame(frame, "test")
        rate = test["y"].mean()
        baseline = max(rate, 1 - rate)  # accuracy of always predicting the majority class
        for fs_name, cols in VOL_FEATURE_SETS.items():
            lr, xgb, note = _train_and_eval(frame, cols)
            rows.append({"horizon": h, "features": fs_name, "model": "logreg",
                         "baseline": baseline, **lr})
            rows.append({"horizon": h, "features": fs_name, "model": f"xgboost({note})",
                         "baseline": baseline, **xgb})
        print(f"  h={h:>2}: built (test 'vol-rises' rate={rate:.3f}, majority baseline={baseline:.3f})")
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward (LogReg/combined) for the volatility task — mean +/- std per horizon."""
    cols = config.VOL_COMBINED_FEATURE_COLS
    per_fold = []
    for h in config.VOL_HORIZONS:
        fv, cur = config.fwd_vol_col(h), config.current_vol_col(h)
        for ts_str in FOLD_STARTS:
            ts = pd.Timestamp(ts_str)
            te = ts + FOLD_LENGTH
            ld = config.label_date_col(h)
            labeled = df[fv].notna() & df[cur].notna() & df[cols].notna().all(axis=1)
            train = df[labeled & (df["date"] < ts) & (df[ld] < ts)].copy()
            test = df[labeled & (df["date"] >= ts) & (df["date"] < te)].copy()
            if len(train) < 200 or len(test) < 30:
                continue
            train["y"] = (train[fv] > train[cur]).astype(int)
            test["y"] = (test[fv] > test[cur]).astype(int)
            if train["y"].nunique() < 2 or test["y"].nunique() < 2:
                continue
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(train[cols].to_numpy(dtype=float))
            clf = _fit_logreg(sc.transform(train[cols].to_numpy(dtype=float)), train["y"].to_numpy())
            pred = clf.predict(sc.transform(test[cols].to_numpy(dtype=float)))
            m = score(test["y"].to_numpy(), pred)
            per_fold.append({"horizon": h, **m})
    pf = pd.DataFrame(per_fold)
    out = []
    for h in config.VOL_HORIZONS:
        sub = pf[pf["horizon"] == h]
        out.append({"horizon": h, "n_folds": len(sub),
                    "acc_mean": sub["accuracy"].mean(), "acc_std": sub["accuracy"].std(ddof=0),
                    "mcc_mean": sub["mcc"].mean(), "mcc_std": sub["mcc"].std(ddof=0)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Regression — predict the actual volatility LEVEL (magnitude), not just direction
# ---------------------------------------------------------------------------
def _reg_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Embargoed rows for the volatility-magnitude regression at horizon h. Target `yv` is
    the continuous next-h-day realized volatility (config.fwd_vol_col(h))."""
    fv, cur = config.fwd_vol_col(h), config.current_vol_col(h)
    ok = df[fv].notna() & df[cur].notna()
    same_split = df["split"] == df[config.label_split_col(h)]
    is_test = df["split"] == "test"
    frame = df[ok & (same_split | is_test)].copy()
    frame = frame.dropna(subset=config.VOL_PRICE_FEATURE_COLS)
    frame["yv"] = frame[fv]
    return frame.reset_index(drop=True)


def _reg_scores(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import mean_squared_error, r2_score

    return {"r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def regression_table(df: pd.DataFrame) -> pd.DataFrame:
    """Predict the volatility magnitude. Report R^2 / RMSE for:
      * persistence baseline  — predict next vol = current vol (a strong, standard baseline),
      * HAR-OLS               — Ridge on just the 3 HAR-RV components (the classic model),
      * Ridge (all features)  — linear model on the full volatility feature set,
      * XGBoost regressor.
    A model only shows skill if it beats persistence.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    cols = config.VOL_COMBINED_FEATURE_COLS
    rows = []
    for h in config.VOL_HORIZONS:
        frame = _reg_frame(df, h)
        train, test = split_frame(frame, "train"), split_frame(frame, "test")
        y_tr, y_te = train["yv"].to_numpy(), test["yv"].to_numpy()
        cur = config.current_vol_col(h)

        # Persistence baseline: next vol ≈ current vol (no fitting).
        rows.append({"horizon": h, "model": "persistence(current vol)", **_reg_scores(y_te, test[cur].to_numpy())})

        # HAR-OLS: Ridge on the three HAR components only.
        sc_h = StandardScaler().fit(train[config.HAR_FEATURE_COLS].to_numpy(float))
        har = Ridge().fit(sc_h.transform(train[config.HAR_FEATURE_COLS].to_numpy(float)), y_tr)
        rows.append({"horizon": h, "model": "HAR-OLS", **_reg_scores(
            y_te, har.predict(sc_h.transform(test[config.HAR_FEATURE_COLS].to_numpy(float))))})

        # Ridge on all volatility features.
        sc = StandardScaler().fit(train[cols].to_numpy(float))
        ridge = Ridge().fit(sc.transform(train[cols].to_numpy(float)), y_tr)
        rows.append({"horizon": h, "model": "ridge(all features)", **_reg_scores(
            y_te, ridge.predict(sc.transform(test[cols].to_numpy(float))))})

        # XGBoost regressor.
        xgb = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=config.RANDOM_SEED, n_jobs=-1, tree_method="hist")
        xgb.fit(train[cols].to_numpy(float), y_tr)
        rows.append({"horizon": h, "model": "xgboost", **_reg_scores(y_te, xgb.predict(test[cols].to_numpy(float)))})
    return pd.DataFrame(rows)


def regression_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward R^2 (Ridge on all features) — mean +/- std per horizon."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    cols = config.VOL_COMBINED_FEATURE_COLS
    per_fold = []
    for h in config.VOL_HORIZONS:
        fv, cur = config.fwd_vol_col(h), config.current_vol_col(h)
        ld = config.label_date_col(h)
        for ts_str in FOLD_STARTS:
            ts = pd.Timestamp(ts_str)
            te = ts + FOLD_LENGTH
            ok = df[fv].notna() & df[cur].notna() & df[cols].notna().all(axis=1)
            train = df[ok & (df["date"] < ts) & (df[ld] < ts)]
            test = df[ok & (df["date"] >= ts) & (df["date"] < te)]
            if len(train) < 200 or len(test) < 30:
                continue
            sc = StandardScaler().fit(train[cols].to_numpy(float))
            r = Ridge().fit(sc.transform(train[cols].to_numpy(float)), train[fv].to_numpy())
            s = _reg_scores(test[fv].to_numpy(), r.predict(sc.transform(test[cols].to_numpy(float))))
            per_fold.append({"horizon": h, **s})
    pf = pd.DataFrame(per_fold)
    out = []
    for h in config.VOL_HORIZONS:
        sub = pf[pf["horizon"] == h]
        out.append({"horizon": h, "n_folds": len(sub),
                    "r2_mean": sub["r2"].mean(), "r2_std": sub["r2"].std(ddof=0)})
    return pd.DataFrame(out)


def _fmt(df: pd.DataFrame) -> str:
    show = df.copy()
    for c in ("baseline", "accuracy", "f1_macro", "mcc", "r2", "rmse"):
        if c in show.columns:
            show[c] = show[c].map(lambda v: f"{v:.4f}")
    return show.to_string(index=False)


def main() -> None:
    df = load_combined()

    print("=== Task B: volatility-regime prediction — fixed test window ===")
    print("(target: will next-h-day volatility exceed current volatility? 'baseline' = "
          "always-predict-majority accuracy)")
    table = fixed_window_table(df)
    print(_fmt(table[["horizon", "features", "model", "baseline", "accuracy", "f1_macro", "mcc"]]))

    print("\n=== Task B: classification walk-forward (LogReg/combined) — mean +/- std ===")
    wf = walk_forward(df)
    for _, r in wf.iterrows():
        print(f"  h={int(r['horizon']):>2}  ({int(r['n_folds'])} folds):  "
              f"accuracy {r['acc_mean']:.3f} +/- {r['acc_std']:.3f}   "
              f"MCC {r['mcc_mean']:+.3f} +/- {r['mcc_std']:.3f}")

    print("\n=== Task B (regression): predict the volatility MAGNITUDE — R^2 / RMSE ===")
    print("(R^2 = fraction of variance explained; a model shows skill only by beating "
          "'persistence')")
    reg = regression_table(df)
    print(_fmt(reg[["horizon", "model", "r2", "rmse"]]))

    print("\n=== Task B (regression) walk-forward (Ridge/all) — R^2 mean +/- std ===")
    rwf = regression_walk_forward(df)
    for _, r in rwf.iterrows():
        print(f"  h={int(r['horizon']):>2}  ({int(r['n_folds'])} folds):  "
              f"R^2 {r['r2_mean']:.3f} +/- {r['r2_std']:.3f}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.DATA_DIR / "volatility_results.csv", index=False)
    reg.to_csv(config.DATA_DIR / "volatility_regression_results.csv", index=False)
    print(f"\nWrote volatility results -> {config.DATA_DIR / 'volatility_results.csv'}")


if __name__ == "__main__":
    main()
