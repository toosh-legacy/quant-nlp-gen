"""Methodology bake-off: which modelling changes actually improve the forecast?

Each candidate is a different *method*, not a different hyperparameter — a different
training set, loss function, feature block or combination rule. They are scored on one
protocol so the comparison is fair:

* selection runs on walk-forward folds **before** `config.TEST_START` only, so nothing here
  can be tuned against the window the README reports;
* the primary metric is **skill versus persistence**, because each fold's R^2 denominator
  is that year's own variance and is not comparable across folds;
* **QLIKE** is reported alongside. Squared error on log-vol is not the natural loss for a
  volatility forecast: it treats a 2-point miss at 10% vol the same as at 60% vol. QLIKE
  (Patton 2011) is the standard robust alternative, and crucially it stays consistent even
  though realized volatility is a *noisy proxy* for true volatility — most losses do not.
* a method must clear the incumbent by more than one standard error to be adopted.

Run:
    python -m volradar.backtest.methods
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.features.assemble import all_features, build_pooled_panel, load_panel
from volradar.features.core import CORE_FEATURES, JUMP_FEATURES, SEMIVAR_FEATURES
from volradar.models import dataset as D
from volradar.models import estimators as E


# ---------------------------------------------------------------------------
# QLIKE — the volatility-native loss
# ---------------------------------------------------------------------------
def qlike(y_log_vol, pred_log_vol) -> float:
    """Mean QLIKE loss. Inputs are log DAILY volatility; QLIKE is defined on variance.

    QLIKE(sigma^2, h) = log(h) + sigma^2 / h, up to a constant. With h = exp(2*pred) and
    sigma^2 = exp(2*y) this is 2*pred + exp(2*(y - pred)).

    Why it matters here: squared error on logs punishes a miss at 10% volatility exactly as
    hard as the same miss at 60%, whereas QLIKE penalises *under*-forecasting risk more than
    over-forecasting it — which is the asymmetry a risk user actually cares about.
    """
    y = np.asarray(y_log_vol, float)
    p = np.asarray(pred_log_vol, float)
    d = np.clip(2.0 * (y - p), -20, 20)     # guard the exponential against outliers
    return float(np.mean(2.0 * p + np.exp(d)))


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
@dataclass
class Method:
    name: str
    note: str

    def features(self, panel: str) -> list[str]:
        return all_features(panel)

    def fit_predict(self, panel: str, h: int, cutoff: pd.Timestamp,
                    test: pd.DataFrame, ctx: dict) -> np.ndarray:
        raise NotImplementedError


class Incumbent(Method):
    """What ships today: GBM on core features, trained on this panel only."""

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        train = D.embargoed_train(ctx["sub"], h, cutoff)
        return E.fit_gbm(train, feats, h, ctx["params"]).predict(test)


class WithJumps(Incumbent):
    """Adds the bipower-variation / jump-share block to the incumbent's features."""

    def features(self, panel):
        return all_features(panel) + JUMP_FEATURES


class WithSemivar(Incumbent):
    """Adds the signed-semivariance block to the incumbent's features.

    Motivated by a clean replication rather than a hunch: fitted as its own model, SHAR
    (Patton & Sheppard 2015) beat plain HAR on both panels — market t=+2.55 with 6/6 fold
    wins, sector t=+2.30. The incumbent already carries a single downside-volatility term,
    but not the up/down *split* or the asymmetry between them, which is where that result
    says the information lives.
    """

    def features(self, panel):
        return all_features(panel) + SEMIVAR_FEATURES


class SemivarJumps(Incumbent):
    """Both literature blocks together, to check they are not substitutes."""

    def features(self, panel):
        return all_features(panel) + SEMIVAR_FEATURES + JUMP_FEATURES


class Pooled(Method):
    """Train on ALL 91 series at once; predict the target panel.

    Tests whether the market model's instability is a sample-size problem. Volatility
    dynamics are broadly shared across equities and every feature is scale-free, so the
    cross-section can supply the data a single index series cannot.
    """
    use_jumps = False

    def features(self, panel):
        base = list(CORE_FEATURES) + (JUMP_FEATURES if self.use_jumps else [])
        return base + ["is_index", "is_sector"]

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        pool = ctx["pooled"]
        train = D.embargoed_train(pool, h, cutoff)
        return E.fit_gbm(train, feats, h, ctx["pooled_params"]).predict(test)


class PooledJumps(Pooled):
    use_jumps = True


class PooledPlusOwn(Pooled):
    """Pooled fit, then a second model on the panel's own history corrects the residual.

    Two-stage: the pooled model learns volatility dynamics generally, and a small
    panel-specific model learns what is idiosyncratic about this series. This keeps the
    data advantage of pooling without forcing one set of coefficients onto every series.
    """
    use_jumps = True

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        pool = ctx["pooled"]
        pool_train = D.embargoed_train(pool, h, cutoff)
        base = E.fit_gbm(pool_train, feats, h, ctx["pooled_params"])

        own = D.embargoed_train(ctx["sub_pooled"], h, cutoff)
        if len(own) < 300:
            return base.predict(test)
        resid = own[config.fwd_col(h)].to_numpy(float) - base.predict(own)
        own2 = own.copy()
        own2["_resid"] = resid

        from xgboost import XGBRegressor
        corr = XGBRegressor(**E.GBM_FIXED, n_estimators=200, max_depth=2, learning_rate=0.03)
        corr.fit(own2[feats].to_numpy(float), own2["_resid"].to_numpy(float))
        return base.predict(test) + corr.predict(test[feats].to_numpy(float))


class ShrinkToPersistence(Incumbent):
    """Blend the model with the persistence baseline, weight chosen on the training data.

    Forecast combination is one of the most reliably useful ideas in forecasting: two
    imperfect predictors with partly independent errors beat either alone. Persistence is
    strong and unbiased; the model is sharper but noisier. The weight is fitted on the
    training period only.
    """

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        sub = ctx["sub"]
        train = D.embargoed_train(sub, h, cutoff)
        model = E.fit_gbm(train, feats, h, ctx["params"])

        # Fit the blend weight on the last 2 years of training data, held out from the fit.
        inner_cut = cutoff - pd.DateOffset(years=2)
        inner_train = D.embargoed_train(sub, h, inner_cut)
        holdout = sub[(sub["date"] >= inner_cut) & (sub["date"] < cutoff)]
        if len(inner_train) < 300 or len(holdout) < 100:
            return model.predict(test)

        inner = E.fit_gbm(inner_train, feats, h, ctx["params"])
        pm = inner.predict(holdout)
        pp = holdout[config.persist_col(h)].to_numpy(float)
        y = holdout[config.fwd_col(h)].to_numpy(float)
        # Least-squares optimal weight on the model, clipped to a sane blend.
        denom = float(np.sum((pm - pp) ** 2))
        w = float(np.sum((y - pp) * (pm - pp)) / denom) if denom > 0 else 1.0
        w = float(np.clip(w, 0.0, 1.0))
        ctx.setdefault("weights", []).append(w)
        return w * model.predict(test) + (1 - w) * test[config.persist_col(h)].to_numpy(float)


class RecencyWeighted(Incumbent):
    """Exponentially down-weight old observations.

    Market structure changes — decimalisation, HFT, the 2010s vol regime, 2020. A model
    that treats 2008 and 2024 as equally informative may be learning dynamics that no
    longer hold. Half-life of 3 years.
    """
    half_life_days = 756

    def fit_predict(self, panel, h, cutoff, test, ctx):
        from xgboost import XGBRegressor
        feats = self.features(panel)
        train = D.embargoed_train(ctx["sub"], h, cutoff)
        age = (cutoff - train["date"]).dt.days.to_numpy(float)
        w = 0.5 ** (age / self.half_life_days)
        m = XGBRegressor(**E.GBM_FIXED, **ctx["params"])
        m.fit(train[feats].to_numpy(float), train[config.fwd_col(h)].to_numpy(float),
              sample_weight=w)
        return m.predict(test[feats].to_numpy(float))


class HarGbmAverage(Incumbent):
    """Simple average of the HAR benchmark and the GBM.

    Equal-weight combination is famously hard to beat ("the forecast combination puzzle").
    HAR is linear and stable; GBM is flexible and unstable. Their errors differ in kind.
    """

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        train = D.embargoed_train(ctx["sub"], h, cutoff)
        gbm = E.fit_gbm(train, feats, h, ctx["params"]).predict(test)
        har = E.fit_har(train, feats, h).predict(test)
        return 0.5 * gbm + 0.5 * har


class RollingRefit(Method):
    """Rolling training window, re-estimated frequently — the fitting scheme, not the model.

    Audrino & Chassot (IJF 2025, "HARd to Beat") tested 1,445 stocks and found the HAR
    model exceptionally hard to beat *when it is re-estimated in a rolling window of roughly
    two and a half to four years*, and that machine-learning models failed to surpass it once
    the linear benchmark was fitted that way. Their point is that the fitting scheme — window
    length and re-estimation frequency — matters more than the model class, and is routinely
    overlooked.

    This project used the overlooked scheme: an expanding window re-estimated once a year.
    An expanding window weights 2008 as heavily as last month; annual re-estimation means a
    December forecast is made by a model that last saw data eleven months earlier. This
    method tests the alternative directly, refitting every `refit_days` on the trailing
    `window_days` only.
    """
    window_days = 756          # ~3 years, the middle of the paper's range
    refit_days = 21            # monthly re-estimation
    estimator = "gbm"

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        sub = ctx["sub"]
        test = test.sort_values("date")
        preds = np.full(len(test), np.nan)
        dates = test["date"].to_numpy()

        # Re-estimate at each chunk boundary on the trailing window ending at that date.
        starts = list(range(0, len(test), self.refit_days))
        for s in starts:
            e = min(s + self.refit_days, len(test))
            asof = pd.Timestamp(dates[s])
            train = D.embargoed_train(sub, h, asof)
            if self.window_days:
                train = train[train["date"] >= asof - pd.Timedelta(days=int(
                    self.window_days * 1.45))]      # calendar days ~ trading days / 0.69
            if len(train) < 250:
                continue
            if self.estimator == "gbm":
                m = E.fit_gbm(train, feats, h, ctx["params"])
            else:
                m = dict(E.LADDER)[self.estimator](train, feats, h)
            preds[s:e] = m.predict(test.iloc[s:e])

        # Any chunk too thin to fit falls back to the persistence baseline rather than NaN.
        missing = np.isnan(preds)
        if missing.any():
            preds[missing] = test[config.persist_col(h)].to_numpy(float)[missing]
        return preds


class RollingHAR(RollingRefit):
    """The paper's actual recommendation: plain linear HAR, rolling window, frequent refit."""
    estimator = "HAR"


class RollingHAR2y(RollingHAR):
    window_days = 504


class RollingHAR4y(RollingHAR):
    window_days = 1008


class ExpandingHARFreq(RollingHAR):
    """Frequent re-estimation but an expanding window — isolates which lever matters."""
    window_days = 0


class HARQMethod(Incumbent):
    """HAR with the measurement-error correction (Bollerslev, Patton & Quaedvlieg 2016)."""

    def fit_predict(self, panel, h, cutoff, test, ctx):
        train = D.embargoed_train(ctx["sub"], h, cutoff)
        return E.fit_harq(train, None, h).predict(test)


class SHARMethod(Incumbent):
    """HAR on signed semivariance (Patton & Sheppard 2015)."""

    def fit_predict(self, panel, h, cutoff, test, ctx):
        train = D.embargoed_train(ctx["sub"], h, cutoff)
        return E.fit_shar(train, None, h).predict(test)


class RollingHARQ(RollingRefit):
    """The two strongest ideas combined: HARQ, rolling window, monthly refit."""
    estimator = "HARQ"


class GbmPlusRollingHAR(RollingRefit):
    """Average the incumbent GBM with a rolling-refit HAR.

    Combination is the most reliably useful idea in forecasting, and these two err
    differently: the GBM is flexible but fitted on stale, equally-weighted history, while
    the rolling HAR is rigid but always current.
    """
    estimator = "HAR"

    def fit_predict(self, panel, h, cutoff, test, ctx):
        rolling = super().fit_predict(panel, h, cutoff, test, ctx)
        feats = self.features(panel)
        train = D.embargoed_train(ctx["sub"], h, cutoff)
        gbm = E.fit_gbm(train, feats, h, ctx["params"]).predict(test.sort_values("date"))
        return 0.5 * rolling + 0.5 * gbm


METHODS: list[Method] = [
    Incumbent("incumbent", "GBM on core features, this panel only (what ships today)"),
    WithJumps("jumps", "+ bipower variation / jump-share block"),
    WithSemivar("semivar", "+ signed semivariance block (SHAR insight)"),
    SemivarJumps("semivar+jumps", "+ both literature blocks"),
    Pooled("pooled", "train on all 91 series"),
    PooledJumps("pooled+jumps", "train on all 91 series, with jump block"),
    PooledPlusOwn("pooled+own", "pooled fit + panel-specific residual correction"),
    ShrinkToPersistence("shrink", "blend with persistence, weight fitted on training data"),
    RecencyWeighted("recency", "exponential sample weighting, 3-year half-life"),
    HarGbmAverage("har+gbm", "equal-weight average of HAR and GBM"),
    # --- literature-backed additions ---------------------------------------------------
    HARQMethod("harq", "HAR + realized-quarticity correction (Bollerslev/Patton/Quaedvlieg)"),
    SHARMethod("shar", "HAR on signed semivariance (Patton/Sheppard)"),
    RollingRefit("roll-gbm-3y", "GBM, 3-year rolling window, monthly refit"),
    RollingHAR("roll-har-3y", "HAR, 3-year rolling window, monthly refit"),
    RollingHAR2y("roll-har-2y", "HAR, 2-year rolling window, monthly refit"),
    RollingHAR4y("roll-har-4y", "HAR, 4-year rolling window, monthly refit"),
    ExpandingHARFreq("exp-har-freq", "HAR, expanding window, monthly refit"),
    RollingHARQ("roll-harq-3y", "HARQ, 3-year rolling window, monthly refit"),
    GbmPlusRollingHAR("gbm+roll-har", "average of GBM and rolling-refit HAR"),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _context(panel: str, h: int, method: Method) -> dict:
    """Shared per-(panel, horizon) setup: labelled frames and tuned parameters."""
    df = load_panel(panel)
    feats = method.features(panel)

    pooled_all = build_pooled_panel()
    ctx: dict = {}

    if isinstance(method, Pooled):
        pool = D.labelled(pooled_all, feats, h)
        ctx["pooled"] = pool
        # The target panel's own rows, expressed in the pooled feature space.
        tick = config.MARKET_TICKER if panel == "market" else None
        ctx["sub_pooled"] = pool[pool["ticker"] == tick] if tick else \
            pool[pool["ticker"].isin(config.SECTOR_ETFS)]
        fit = D.embargoed_train(pool, h, pd.Timestamp(config.VAL_START))
        val = D.time_slice(pool, config.VAL_START, config.VAL_END)
        ctx["pooled_params"] = (E.tune_gbm(fit, val, feats, h, quiet=True)
                                if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)
        # Test rows must come from the pooled frame so the extra columns exist.
        ctx["sub"] = ctx["sub_pooled"]
    else:
        sub = D.labelled(df, feats, h)
        ctx["sub"] = sub
        fit = D.embargoed_train(sub, h, pd.Timestamp(config.VAL_START))
        val = D.time_slice(sub, config.VAL_START, config.VAL_END)
        ctx["params"] = (E.tune_gbm(fit, val, feats, h, quiet=True)
                         if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)
    return ctx


def fold_edges(years: list[int], per_year: int = 1) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Fold boundaries. `per_year=2` splits each year in half.

    Splitting folds more finely is a legitimate way to gain statistical power for a paired
    comparison: it adds independent evaluation points without touching the test period or
    changing what any model is trained on before its own cutoff. It is not a second bite at
    the apple — the same data is simply scored in smaller, still-disjoint blocks.
    """
    out = []
    for y in years:
        for k in range(per_year):
            m0 = 1 + (12 // per_year) * k
            start = pd.Timestamp(f"{y}-{m0:02d}-01")
            end = (start + pd.DateOffset(months=12 // per_year))
            out.append((start, end))
    return out


def run_method(method: Method, panel: str, h: int, years: list[int],
               per_year: int = 1) -> dict:
    ctx = _context(panel, h, method)
    sub = ctx["sub"]
    rows = []
    min_test = max(60 // per_year, 30)
    for cutoff, end in fold_edges(years, per_year):
        test = D.time_slice(sub, cutoff, end)
        train_n = len(D.embargoed_train(sub, h, cutoff))
        if train_n < 300 or len(test) < min_test:
            continue
        pred = method.fit_predict(panel, h, cutoff, test, ctx)
        y = test[config.fwd_col(h)].to_numpy(float)
        per = test[config.persist_col(h)].to_numpy(float)
        r = M.regression_metrics(y, pred, per)
        r.update(M.direction_metrics(y, pred, per))
        r["qlike"] = qlike(y, pred)
        r["qlike_persist"] = qlike(y, per)
        r["qlike_gain"] = r["qlike_persist"] - r["qlike"]
        rows.append(r)

    f = pd.DataFrame(rows)
    n = max(len(f), 1)
    return {
        "method": method.name, "panel": panel, "horizon": h, "note": method.note,
        "n_folds": len(f),
        "skill_mean": f["skill_vs_persistence"].mean(),
        "skill_std": f["skill_vs_persistence"].std(ddof=0),
        "skill_se": f["skill_vs_persistence"].std(ddof=0) / n ** 0.5,
        "skill_min": f["skill_vs_persistence"].min(),
        "years_positive": int((f["skill_vs_persistence"] > 0).sum()),
        "qlike_gain": f["qlike_gain"].mean(),
        "dir_accuracy": f["dir_accuracy"].mean(),
        "r2_mean": f["r2"].mean(),
        # Retained for the paired comparison below — every method sees identical folds.
        "_per_fold": f["skill_vs_persistence"].to_numpy(),
        "_years": years[:len(f)],
    }


def paired_verdict(cand: dict, base: dict) -> dict:
    """Compare a candidate to the incumbent fold by fold.

    Comparing two marginal means and their separate standard errors is the wrong test here,
    and it fails in a specific way that matters: an *unstable* incumbent has a large
    standard error, which inflates the bar a challenger must clear — so the noisier the
    thing being replaced, the harder it becomes to replace it. Exactly backwards.

    Every method is evaluated on the same folds, so the differences are paired and most of
    the shared year-to-year variation cancels. The test is on the mean of the per-fold
    difference.
    """
    a, b = np.asarray(cand["_per_fold"], float), np.asarray(base["_per_fold"], float)
    n = min(len(a), len(b))
    if n < 3:
        return {"delta": np.nan, "delta_se": np.nan, "t": np.nan, "wins": 0, "n": n}
    d = a[:n] - b[:n]
    se = d.std(ddof=1) / np.sqrt(n)
    return {
        "delta": float(d.mean()),
        "delta_se": float(se),
        "t": float(d.mean() / se) if se > 0 else np.nan,
        "wins": int((d > 0).sum()),
        "n": n,
    }


def bake_off(panel: str, h: int | None = None, years: list[int] | None = None) -> pd.DataFrame:
    h = h or config.HEADLINE_HORIZON
    years = years or [y for y in config.WALK_YEARS if y < int(config.TEST_START[:4])]
    print(f"\n{'=' * 78}\n=== Method bake-off: {panel} panel, h={h}, "
          f"folds {years[0]}-{years[-1]} (pre-test only)\n{'=' * 78}")
    rows = []
    for m in METHODS:
        try:
            r = run_method(m, panel, h, years)
        except Exception as exc:
            print(f"  {m.name:<14} FAILED: {exc}")
            continue
        rows.append(r)
        print(f"  {r['method']:<14} skill {r['skill_mean']:+.3f} +/- {r['skill_std']:.3f}"
              f"  (SE {r['skill_se']:.3f})  QLIKE gain {r['qlike_gain']:+.4f}"
              f"  dir {r['dir_accuracy']:.3f}  {r['years_positive']}/{r['n_folds']} yrs+")
    base = next((r for r in rows if r["method"] == "incumbent"), None)
    if base is not None:
        print(f"\n  Paired vs incumbent (same folds; t > 2 = clears noise):")
        for r in rows:
            if r["method"] == "incumbent":
                continue
            v = paired_verdict(r, base)
            r.update({f"paired_{k}": val for k, val in v.items()})
            verdict = ("ADOPT" if v["t"] is not np.nan and v["t"] > 2 and v["delta"] > 0
                       else "hold")
            print(f"    {verdict:<6} {r['method']:<14} delta {v['delta']:+.3f} "
                  f"+/- {v['delta_se']:.3f}  t={v['t']:+.2f}  "
                  f"wins {v['wins']}/{v['n']}")

    out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in rows])
    return out.sort_values("skill_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    config.ensure_dirs()
    out = pd.concat([bake_off(p) for p in ("market", "sector")], ignore_index=True)
    path = config.REPORT_DIR / "method_bakeoff.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote -> {path}")


if __name__ == "__main__":
    main()
