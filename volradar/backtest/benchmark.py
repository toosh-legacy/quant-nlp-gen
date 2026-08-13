"""External benchmark: how does this model do against the literature, on the literature's data?

Every other comparison in this project is internal — our GBM against our HAR, on our panel,
at our horizon. That is rigorous and it is also self-referential, and no amount of internal
rigour can tell you whether the model is any good *by the standards of the field*. This
module is the answer to that, and it is deliberately built to be unflattering.

**The data is not ours.** The Oxford-Man Realized Library: 21 international equity indices,
2000-01-03 to 2016-09-26, with realized variance computed from 5-minute intraday returns.
This is the dataset a large share of the realized-volatility literature is written on.

**The target is not ours either.** The production model forecasts a 10-day average of
*close-to-close daily* variance. Here the target is log realized volatility built from
**5-minute realized variance**, at **h = 1, 5 and 22** — the horizons the HAR literature
reports. Our own headline horizon is deliberately not privileged.

**The benchmarks are the field's, not a strawman.** Log-HAR is the reference model, because
the recent large-scale comparisons (including the 2026 time-series-foundation-model study
across 50 assets) find it is the thing to beat and that most challengers do not:

* `persistence` — last h days repeated. Free, and hard to beat.
* `HAR` — Corsi's daily/weekly/monthly cascade on log RV. This *is* Log-HAR, the reference.
* `HAR-J` — RV split into its continuous (bipower) and jump parts.
* `HAR-RS` — RV split into up-day and down-day semivariance (Patton & Sheppard).
* `HARQ` — the measurement-error correction (Bollerslev, Patton & Quaedvlieg).
* `ridge` — linear on the full feature set; isolates capacity from feature count.
* `gbm` — this project's estimator, on the full feature set.

**The metric is QLIKE**, measured against Log-HAR and averaged across assets with **equal
weight** — exactly how the comparison papers report it, so one easy index cannot carry the
result. Two forms are given: the *gain* (positive beats Log-HAR), which is unit-free and is
the number to trust, and the *ratio* (below 1.000 beats it), which is comparable in spirit
to published tables but depends on the units variance is measured in. R² and skill are
reported alongside, and the Model Confidence Set says which models cannot be separated from
the best.

Run:
    python -m volradar.backtest.benchmark
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.data.realized import load_realized_library
from volradar.models import dataset as D
from volradar.models import estimators as E

REPORT = config.REPORT_DIR / "benchmark.csv"
REPORT_MCS = config.REPORT_DIR / "benchmark_mcs.csv"

# The literature's horizons: one day, one week, one month.
HORIZONS = [1, 5, 22]

# Walk-forward years. The library starts in 2000; the first fold trains on 2000-2004 so no
# model is fitted on fewer than ~1,200 observations.
YEARS = list(range(2005, 2017))

# Corsi's cascade, in logs. `HAR` on these columns is what the literature calls Log-HAR.
HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m"]
JUMP_COLS = ["log_bpv_d", "log_bpv_w", "log_bpv_m", "log_jump_d"]
SEMI_COLS = ["log_rs_down_d", "log_rs_up_d", "log_rv_w", "log_rv_m"]
EXTRA_COLS = ["log_rk_d", "log_medrv_d", "rv_slope_wm", "ret_1", "ret_5", "log_range_d"]
FULL_COLS = HAR_COLS + JUMP_COLS + SEMI_COLS[:2] + EXTRA_COLS


# QLIKE = log(h) + sigma^2/h carries a log term, so its LEVEL depends on the units variance
# is measured in — and on daily-return units it comes out around -8.8, i.e. negative. A
# "loss ratio", the way the comparison papers report it, is only meaningful for a positive
# loss: dividing two negative numbers silently inverts the ranking, so the best model looks
# like the worst. (It did exactly that on the first run here.)
#
# Rescaling variance to annualized percent puts QLIKE back in positive territory without
# changing any comparison, because scaling variance by k shifts QLIKE by exactly log(k) and
# leaves every *difference* untouched. The ratio is then well defined, and the units it is
# defined in are stated rather than assumed.
QLIKE_UNIT_SHIFT = float(np.log(config.TRADING_DAYS * 100.0 ** 2))


def qlike_ann(y_log_vol, pred_log_vol) -> float:
    """QLIKE with variance in annualized-percent units, so it is positive and ratio-safe."""
    return M.qlike(y_log_vol, pred_log_vol) + QLIKE_UNIT_SHIFT


def _lv(var) -> pd.Series:
    """Daily variance -> log daily volatility."""
    return np.log(np.sqrt(pd.Series(var).clip(lower=0.0)) + config.EPS)


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One index's realized measures -> HAR-family features and forward targets.

    Everything on row t uses data up to and including t; every target uses t+1..t+h only.
    """
    # Compact to complete observations BEFORE any rolling window. The library carries ~4.4%
    # incomplete days (local holidays, feed gaps), and a rolling window returns NaN if a
    # single observation inside it is missing — so a 22-day window over the raw series is
    # empty almost everywhere. Skipping this step silently discards two thirds of the sample.
    base = ["rv5", "rv5_ss", "rv10", "bpv5", "rsv5", "rk", "medrv", "hl_range", "ret_oxf"]
    d = (df.sort_values("date").dropna(subset=base).reset_index(drop=True).copy())
    rv = d["rv5_ss"].fillna(d["rv5"])
    d["_rv"] = rv

    d["log_rv_d"] = _lv(rv)
    d["log_rv_w"] = _lv(rv.rolling(5).mean())
    d["log_rv_m"] = _lv(rv.rolling(22).mean())
    d["rv_slope_wm"] = d["log_rv_w"] - d["log_rv_m"]

    # Continuous / jump split. Bipower variation survives jumps that wreck RV.
    bpv = d["bpv5"]
    d["log_bpv_d"] = _lv(bpv)
    d["log_bpv_w"] = _lv(bpv.rolling(5).mean())
    d["log_bpv_m"] = _lv(bpv.rolling(22).mean())
    d["log_jump_d"] = _lv((rv - bpv).clip(lower=0.0))

    # Signed semivariance. The library publishes the DOWNSIDE component; the upside part is
    # the remainder, which is the identity Patton & Sheppard's decomposition rests on.
    rs_down = d["rsv5"]
    d["log_rs_down_d"] = _lv(rs_down)
    d["log_rs_up_d"] = _lv((rv - rs_down).clip(lower=0.0))

    d["log_rk_d"] = _lv(d["rk"])
    d["log_medrv_d"] = _lv(d["medrv"])
    d["log_range_d"] = np.log(d["hl_range"].abs() + config.EPS)
    d["ret_1"] = d["ret_oxf"]
    d["ret_5"] = d["ret_oxf"].rolling(5).sum()

    # HARQ's measurement-error term. With genuine intraday data the disagreement between
    # independent estimators of the same daily variance is a usable proxy for the sampling
    # error of RV — the quantity realized quarticity estimates, and the thing daily bars
    # could not supply.
    spread = (rv - d["rv10"]).abs() + (rv - d["rk"]).abs()
    d["rq_sqrt"] = np.sqrt(np.sqrt(spread.clip(lower=0.0)))

    for h in HORIZONS:
        fwd = rv.rolling(h).mean().shift(-h)
        d[config.fwd_col(h)] = _lv(fwd)
        d[config.persist_col(h)] = _lv(rv.rolling(h).mean())
        d[config.label_date_col(h)] = d["date"].shift(-h)
    return d


# --- the model set --------------------------------------------------------------------
class _FittedHARQ(E.Fitted):
    """HARQ over this module's three-component cascade.

    `estimators.fit_harq` is hard-wired to the production panel's four-component HAR
    (which includes a quarterly term the literature's specification does not), so the
    benchmark builds its own rather than quietly comparing a different model to the
    published one.
    """

    def predict(self, df):
        X = _harq_design(df, self.params["rq_mu"], self.params["rq_sd"])
        return np.asarray(self.estimator.predict(self.scaler.transform(X)), dtype=float)


def _harq_design(df, mu: float, sd: float) -> np.ndarray:
    base = df[HAR_COLS].to_numpy(float)
    z = ((df["rq_sqrt"].to_numpy(float) - mu) / sd).reshape(-1, 1)
    return np.hstack([base, base[:, [0]] * z])       # daily lag x standardised sqrt(RQ)


def _fit_harq(train, h: int) -> E.Fitted:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    mu = float(train["rq_sqrt"].mean())
    sd = float(train["rq_sqrt"].std()) or 1.0
    X = _harq_design(train, mu, sd)
    sc = StandardScaler().fit(X)
    est = Ridge(alpha=1.0).fit(sc.transform(X),
                               train[config.fwd_col(h)].to_numpy(float))
    return _FittedHARQ(name="HARQ", columns=HAR_COLS + ["rq_sqrt"], estimator=est,
                       scaler=sc, params={"rq_mu": mu, "rq_sd": sd})


# The GBM's capacity is tuned in production, so it must be tuned here too. Handing every
# linear benchmark its natural specification while giving the tree model an untuned default
# is a rigged comparison — in whichever direction it happens to land. Tuning happens once
# per (index, horizon) on a validation slice that ends BEFORE the first evaluation year, so
# no fold is ever scored by a model whose capacity was chosen using that fold.
TUNE_FIT_END = "2003-01-01"
TUNE_VAL_START, TUNE_VAL_END = "2003-01-01", f"{YEARS[0]}-01-01"


def _tuned_params(sub: pd.DataFrame, h: int) -> dict:
    fit = D.embargoed_train(sub, h, pd.Timestamp(TUNE_FIT_END))
    val = D.time_slice(sub, TUNE_VAL_START, TUNE_VAL_END)
    if len(fit) < 300 or len(val) < 150:
        return E.GBM_DEFAULT
    return E.tune_gbm(fit, val, FULL_COLS, h, quiet=True)


def _fit(name: str, train, h: int, params: dict | None = None):
    if name == "HAR":
        return E._fit_ridge(train, HAR_COLS, h, "HAR")
    if name == "HAR-J":
        return E._fit_ridge(train, JUMP_COLS + ["log_rv_w", "log_rv_m"], h, "HAR-J")
    if name == "HAR-RS":
        return E._fit_ridge(train, SEMI_COLS, h, "HAR-RS")
    if name == "HARQ":
        return _fit_harq(train, h)
    if name == "ridge":
        return E._fit_ridge(train, FULL_COLS, h, "ridge")
    if name == "gbm":
        return E.fit_gbm(train, FULL_COLS, h, params or E.GBM_DEFAULT)
    raise KeyError(name)


MODELS = ["HAR", "HAR-J", "HAR-RS", "HARQ", "ridge", "gbm"]
REFERENCE = "HAR"          # Log-HAR: the benchmark every ratio is taken against


def run_symbol(symbol: str, panel: pd.DataFrame, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward every model on one index at one horizon."""
    need = FULL_COLS + SEMI_COLS + ["rq_sqrt", config.fwd_col(h),
                                    config.persist_col(h), config.label_date_col(h)]
    sub = panel.dropna(subset=list(dict.fromkeys(need))).reset_index(drop=True)
    if len(sub) < 800:
        return pd.DataFrame(), pd.DataFrame()

    params = _tuned_params(sub, h)
    rows, obs = [], []
    for year in YEARS:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = D.embargoed_train(sub, h, cutoff)
        test = D.time_slice(sub, cutoff, f"{year + 1}-01-01")
        if len(train) < 500 or len(test) < 60:
            continue
        y = test[config.fwd_col(h)].to_numpy(float)
        per = test[config.persist_col(h)].to_numpy(float)

        preds = {"persistence": per}
        for name in MODELS:
            preds[name] = _fit(name, train, h, params).predict(test)

        for name, p in preds.items():
            r = M.regression_metrics(y, p, per)
            rows.append({"symbol": symbol, "horizon": h, "year": year,
                         "model": name, "r2": r["r2"],
                         "qlike": r["qlike"] + QLIKE_UNIT_SHIFT,
                         "skill": r["skill_vs_persistence"], "n_test": len(test)})
            obs.append(pd.DataFrame({"date": test["date"].to_numpy(), "model": name,
                                     "loss": M.qlike_obs(y, p)}))
    return pd.DataFrame(rows), (pd.concat(obs, ignore_index=True) if obs else pd.DataFrame())


def main() -> None:
    config.ensure_dirs()
    lib = load_realized_library()
    symbols = sorted(lib["symbol"].unique())
    print(f"External benchmark: {len(symbols)} indices, "
          f"{lib['date'].min().date()}..{lib['date'].max().date()}")
    print(f"Target: log realized vol from 5-minute RV. Horizons {HORIZONS}. "
          f"Walk-forward {YEARS[0]}-{YEARS[-1]}.\n")

    all_rows, mcs_rows = [], []
    for h in HORIZONS:
        per_symbol, losses = [], []
        for sym in symbols:
            panel = build_panel(lib[lib["symbol"] == sym])
            rows, obs = run_symbol(sym, panel, h)
            if rows.empty:
                continue
            per_symbol.append(rows)
            obs["symbol"] = sym
            losses.append(obs)
        if not per_symbol:
            continue
        res = pd.concat(per_symbol, ignore_index=True)
        all_rows.append(res)

        # Per-asset mean loss, then the ratio against Log-HAR, then an EQUAL-WEIGHT average
        # across assets. Equal weighting is the point: pooling observations would let the
        # noisiest index dominate and turn a broad result into a narrow one.
        per_asset = res.groupby(["symbol", "model"])["qlike"].mean().unstack()
        ratio = per_asset.div(per_asset[REFERENCE], axis=0)
        # The DIFFERENCE is the primary number: it is unit-free, so it cannot be distorted
        # by the scale variance happens to be measured in. The ratio is reported alongside
        # for comparability with published tables, in the units named above.
        diff = per_asset[REFERENCE].values[:, None] - per_asset
        summary = pd.DataFrame({
            "qlike_gain_vs_logHAR": diff.median(),
            "qlike_gain_mean": diff.mean(),
            "qlike_ratio_vs_logHAR": ratio.mean(),
            "beats_logHAR_on_n_assets": (diff > 0).sum(),
            "r2": res.groupby("model")["r2"].mean(),
            "skill_vs_persistence": res.groupby("model")["skill"].mean(),
        }).loc[["persistence"] + MODELS]
        summary["n_assets"] = len(per_asset)
        summary["horizon"] = h

        print(f"=== h = {h} day{'s' if h > 1 else ''} "
              f"({len(per_asset)} indices, equal-weighted) ===")
        show = summary.copy()
        show["qlike_ratio_vs_logHAR"] = show["qlike_ratio_vs_logHAR"].map("{:.4f}".format)
        print(show[["qlike_gain_vs_logHAR", "beats_logHAR_on_n_assets", "r2",
                    "skill_vs_persistence"]].round(4).to_string())

        # Model Confidence Set on the pooled per-observation QLIKE.
        L = (pd.concat(losses, ignore_index=True)
               .pivot_table(index=["symbol", "date"], columns="model", values="loss"))
        mcs = M.model_confidence_set(L.dropna())
        mcs["horizon"] = h
        mcs_rows.append(mcs)
        kept = mcs[mcs["in_mcs"]]["model"].tolist()
        print(f"  MCS (alpha=0.10): {len(kept)}/{len(mcs)} survive -> {', '.join(kept)}\n")

        summary.reset_index().rename(columns={"index": "model"}).to_csv(
            REPORT, mode="a", header=not REPORT.exists(), index=False)

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            config.REPORT_DIR / "benchmark_folds.csv", index=False)
    if mcs_rows:
        pd.concat(mcs_rows, ignore_index=True).to_csv(REPORT_MCS, index=False)
    print(f"Wrote -> {REPORT}")


if __name__ == "__main__":
    REPORT.unlink(missing_ok=True)
    main()
