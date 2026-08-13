"""A separate head for the question the mean forecast answers badly: is stress coming?

The event study records the shipped model's weakest result — it warned ahead of a minority
of stress episodes — and diagnoses it correctly: a squared-error regression estimates the
conditional *mean*, and the conditional mean of a right-skewed variable shrinks toward the
middle of the distribution. The forecast therefore rarely reaches the stressed band until a
spike is already forming. That is a property of the loss function, not a bug in the fit.

`settle.py` already tested quantile-trained GBMs — and rejected them, correctly, because it
graded them on skill and QLIKE. A quantile forecast does not estimate the mean, so those
metrics condemn it by construction. The mistake was not the verdict; it was asking one
model to do two jobs.

So this module keeps the mean forecast exactly as it is and asks a **separate** question
with its own model and its own metrics:

* the target is binary — will the next h days land in the stressed regime?
* the thresholds are the same frozen, training-period ones the rest of the project uses,
  so an alarm here means the same thing an alarm means everywhere else;
* the adoption metric is **MCC on that binary label**, not recall. Recall alone is trivially
  gamed by alarming every day, and a model that screams constantly is worthless — MCC is
  the balanced measure that cannot be gamed that way. Detection rate, lead time and
  false-alarm rate are reported alongside, descriptively.

Candidates are adopted by the project-wide bar: paired against the incumbent rule fold by
fold, t > 2, at both fold granularities, on pre-`TEST_START` folds only.

Run:
    python -m volradar.backtest.alarm
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from volradar import config
from volradar.backtest import metrics as M
from volradar.features.assemble import all_features, fit_and_eval_frames
from volradar.models import dataset as D
from volradar.models import estimators as E

PRE_TEST_YEARS = [y for y in config.WALK_YEARS if y < int(config.TEST_START[:4])]
REPORT = config.REPORT_DIR / "alarm.csv"

# The quantile the alarm model targets. Well above the mean and below the 80th-percentile
# regime cut itself, so the model still has to discriminate rather than always firing.
ALARM_QUANTILE = 0.85

# A fold with almost no stress cannot rank alarm rules, so those folds are skipped — and
# with enough of them skipped there is no test left to run. Below this many usable folds the
# module reports "not tested" rather than a verdict. This is not a formality: the pre-test
# window is a calm stretch for the market panel, and reporting its NaN t-statistic as "hold"
# would file an untested candidate under the same heading as a rejected one.
MIN_FOLDS_FOR_A_VERDICT = 3


def _stress_cuts(rows: pd.DataFrame, h: int) -> pd.Series:
    """Each series' own stressed threshold, from the rows given (all pre-cutoff).

    Per-ticker rather than pooled: the pooled frame contains single stocks, whose volatility
    distribution sits well above an index's. One shared cut point would label ordinary
    single-stock days "stressed" and quiet index days "calm" no matter what happened.

    The market panel carries no `ticker` column — it is a single series — so it gets one cut
    under a fixed key.
    """
    hi_q = config.REGIME_QUANTILES[1]
    if "ticker" not in rows.columns:
        return pd.Series({config.MARKET_TICKER: rows[config.fwd_col(h)].quantile(hi_q)})
    return rows.groupby("ticker", observed=True)[config.fwd_col(h)].quantile(hi_q)


def _flags(rows: pd.DataFrame, cuts: pd.Series, h: int) -> np.ndarray:
    """Boolean stressed labels for `rows` under per-ticker cut points."""
    cut = _cut_for(rows, cuts)
    return (rows[config.fwd_col(h)].to_numpy(float) > cut)


def _cut_for(rows: pd.DataFrame, cuts: pd.Series) -> np.ndarray:
    if "ticker" not in rows.columns:
        return np.full(len(rows), float(cuts.iloc[0]))
    return rows["ticker"].map(cuts).to_numpy(float)


def _fit_classifier(train: pd.DataFrame, features: list[str], y: np.ndarray,
                    params: dict) -> object:
    from xgboost import XGBClassifier

    p = {k: v for k, v in params.items()
         if k in ("n_estimators", "max_depth", "learning_rate")}
    # Stressed days are ~20% of the sample by construction, so the positive class is
    # up-weighted to its inverse frequency; without it the classifier maximises accuracy by
    # never alarming, which is the exact failure this module exists to fix.
    pos = max(int(y.sum()), 1)
    est = XGBClassifier(**E.GBM_FIXED, **p, eval_metric="logloss",
                        scale_pos_weight=(len(y) - pos) / pos)
    est.fit(train[features].to_numpy(float), y.astype(int))
    return est


def _rule_alarms(train, test, features, h, params, cuts) -> dict[str, np.ndarray]:
    """Every alarm rule's boolean output on one fold's test rows."""
    y_train = _flags(train, cuts, h)
    test_cut = _cut_for(test, cuts)

    out: dict[str, np.ndarray] = {}

    # 1. The incumbent rule: threshold the mean forecast at the stressed cut.
    mean_pred = E.fit_gbm(train, features, h, params).predict(test)
    out["mean-threshold"] = mean_pred > test_cut

    # 2. A high-quantile regression, thresholded the same way. Same model family, same cut,
    #    only the estimated functional of the distribution differs.
    q_pred = E.fit_gbm_quantile(train, features, h, params,
                                alpha=ALARM_QUANTILE).predict(test)
    out[f"quantile-{int(ALARM_QUANTILE * 100)}"] = q_pred > test_cut

    # 3. A classifier trained directly on the binary label.
    if y_train.sum() >= 30 and (~y_train).sum() >= 30:
        clf = _fit_classifier(train, features, y_train, params)
        prob = clf.predict_proba(test[features].to_numpy(float))[:, 1]
        out["classifier"] = prob >= 0.5
        # 4. The same classifier at the incumbent's own alarm budget. This is the fairest
        #    comparison available: at exactly the number of alarms the shipped rule raises,
        #    does ranking days by the classifier catch more episodes? It removes threshold
        #    choice from the comparison entirely.
        budget = int(out["mean-threshold"].sum())
        matched = np.zeros(len(test), dtype=bool)
        if 0 < budget < len(test):
            matched[np.argsort(-prob)[:budget]] = True
        out["classifier-matched"] = matched
    return out


def _score(y_true: np.ndarray, alarm: np.ndarray) -> dict:
    """Balanced quality of a binary alarm, plus the two numbers a user reads."""
    t, a = y_true.astype(int), alarm.astype(int)
    tp = int((t & a).sum())
    return {
        # MCC is the adoption metric: it collapses to zero for a rule that always fires and
        # for one that never does, which recall and precision individually do not.
        "alarm_mcc": float(matthews_corrcoef(t, a)) if t.sum() and a.sum() else 0.0,
        "stressed_recall": float(tp / max(t.sum(), 1)),
        "precision": float(tp / max(a.sum(), 1)),
        "alarm_share": float(a.mean()),
    }


def per_fold(panel: str, h: int, years: list[int], per_year: int) -> pd.DataFrame:
    features = all_features(panel)
    tr_sub, ev_sub = fit_and_eval_frames(panel, features, h)

    fit = D.embargoed_train(tr_sub, h, pd.Timestamp(config.VAL_START))
    val = D.time_slice(ev_sub, config.VAL_START, config.VAL_END)
    params = (E.tune_gbm(fit, val, features, h, quiet=True)
              if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)

    rows = []
    min_test = max(60 // per_year, 30)
    for cutoff, end in D.fold_edges(years, per_year):
        train = D.embargoed_train(tr_sub, h, cutoff)
        test = D.time_slice(ev_sub, cutoff, end)
        if len(train) < 400 or len(test) < min_test:
            continue
        # Cut points come from the EVALUATION panel's own pre-cutoff rows — the same rule
        # `oos.py` uses — so "stressed" means the same thing here as on the dashboard.
        cuts = _stress_cuts(D.embargoed_train(ev_sub, h, cutoff), h)
        if cuts.empty:
            continue
        y_true = _flags(test, cuts, h)
        if y_true.sum() < 5:
            continue          # a fold with almost no stress cannot rank alarm rules
        for name, alarm in _rule_alarms(train, test, features, h, params, cuts).items():
            rows.append({"fold": str(cutoff.date()), "rule": name, **_score(y_true, alarm)})
    return pd.DataFrame(rows)


def evaluate(panel: str, h: int | None = None) -> pd.DataFrame:
    h = h or config.HEADLINE_HORIZON
    print(f"\n{'=' * 74}\n=== Alarm head: {panel} panel, h={h}, pre-test folds only"
          f"\n{'=' * 74}")
    out = []
    # Three granularities here rather than the project's usual two. Annual and semi-annual
    # folds leave only two to four *usable* folds, because a fold with almost no stress in it
    # cannot rank alarm rules and 2012-2017 is a calm stretch — so the paired test has almost
    # no power. Quarterly folds recover some.
    #
    # This is a power decision, not a threshold decision, and it makes the bar STRICTER: a
    # rule must now clear t > 2 at three slicings instead of two. It was added after seeing
    # that the first two were underpowered, which is worth stating plainly; it would not be
    # legitimate if it had been added after seeing which way a *verdict* went.
    for per_year, label in [(1, "annual"), (2, "semi-annual"), (4, "quarterly")]:
        folds = per_fold(panel, h, PRE_TEST_YEARS, per_year)
        if folds.empty:
            continue
        print(f"\n--- {label} folds ---")
        print(folds.groupby("rule")[["alarm_mcc", "stressed_recall", "precision",
                                     "alarm_share"]].mean().round(3).to_string())

        base = folds[folds["rule"] == "mean-threshold"].set_index("fold")["alarm_mcc"]
        n_usable = len(base)
        if n_usable < MIN_FOLDS_FOR_A_VERDICT:
            # Say so, rather than printing a NaN t-statistic that reads like a measurement.
            # Only folds containing real stress can rank alarm rules, and the pre-test window
            # is a calm stretch for the market panel, so this is expected there — but "the
            # test could not run" and "the candidates lost" must not look the same.
            print(f"  NOT TESTED: only {n_usable} of the pre-test folds contain enough "
                  f"stressed days to rank alarm rules (need {MIN_FOLDS_FOR_A_VERDICT}). "
                  f"The means above are descriptive only.")

        print(f"  paired on alarm_mcc (t>{M.ADOPT_T:g} adopts):")
        for rule, g in folds.groupby("rule"):
            if rule == "mean-threshold":
                continue
            s = g.set_index("fold")["alarm_mcc"]
            common = base.index.intersection(s.index)
            v = M.paired_test(s[common].to_numpy(), base[common].to_numpy())
            testable = v["n"] >= MIN_FOLDS_FOR_A_VERDICT
            verdict = "ADOPT" if M.adopts(v) else ("hold " if testable else "n/a  ")
            detail = (f"delta {v['delta']:+.4f}  t={v['t']:+.2f}  wins {v['wins']}/{v['n']}"
                      if testable else f"only {v['n']} usable folds — no verdict")
            print(f"    {verdict:<6} {rule:<20} {detail}")
            out.append({"panel": panel, "horizon": h, "granularity": label, "rule": rule,
                        "delta_mcc": v["delta"], "t": v["t"], "wins": v["wins"],
                        "n_folds": v["n"], "testable": testable,
                        "adopts": bool(M.adopts(v)),
                        **g[["alarm_mcc", "stressed_recall", "precision",
                             "alarm_share"]].mean().to_dict()})
    return pd.DataFrame(out)


def main() -> None:
    config.ensure_dirs()
    out = pd.concat([evaluate(p) for p in ("market", "sector")], ignore_index=True)
    out.to_csv(REPORT, index=False)
    print(f"\nWrote -> {REPORT}")

    print(f"\n=== Verdict (adopt only if t>{M.ADOPT_T:g} at BOTH fold granularities) ===")
    adopted = False
    for (panel, rule), g in out.groupby(["panel", "rule"]):
        if len(g) >= 3 and g["adopts"].all():
            adopted = True
            print(f"  ADOPT  {panel}/{rule}: t = "
                  f"{', '.join(f'{v:+.2f}' for v in g['t'])}")
    if not adopted:
        print("  Nothing clears the bar. The mean-threshold rule stands, and with it the "
              "README's finding that this is not a reliable crash alarm.")
    untested = out[~out["testable"]].groupby("panel")["rule"].nunique()
    for panel, n in untested.items():
        print(f"  NOTE   {panel}: {n} rule(s) could not be tested at one or both "
              f"granularities for want of stressed folds — read those as untested, not as "
              f"rejected.")


if __name__ == "__main__":
    main()
