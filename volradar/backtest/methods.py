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
* a method is adopted only if it beats the incumbent on a **paired** fold-by-fold t-test at
  t > 2, on skill *and* on QLIKE — the same bar `select_features.py` and `settle.py` use,
  implemented once in `metrics.paired_test`;
* the table also reports the **Model Confidence Set** (Hansen, Lunde & Nason 2011): the
  methods that cannot be statistically distinguished from the best one. Naming a single
  winner from a table this tightly packed claims more than the data supports.

Run:
    python -m volradar.backtest.methods
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.features.assemble import (all_features, build_pooled_panel,
                                        fit_and_eval_frames, load_panel)
from volradar.features.core import CORE_FEATURES, JUMP_FEATURES, SEMIVAR_FEATURES
from volradar.models import dataset as D
from volradar.models import estimators as E
from volradar.models.dataset import fold_edges          # re-exported: callers expect it here

# QLIKE, the paired adoption test and the fold boundaries all live in one place now —
# `metrics.qlike`, `metrics.paired_test` and `dataset.fold_edges`. This module used to
# carry its own copies, which is how a project ends up with two subtly different bars.
qlike = M.qlike


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
    """What ships today: GBM on the selected features, trained the way production trains.

    "The way production trains" is load-bearing and used to be wrong here. The market panel
    fits on its own history and the sector panel fits on the pooled cross-section
    (`config.TRAINING_SOURCE`); this class now routes through `fit_and_eval_frames` so the
    incumbent every challenger is measured against is the model that actually ships. Scoring
    challengers against a sector model trained on 37k rows, then shipping one trained on
    471k, would make the whole bake-off a comparison nobody runs.
    """

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
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
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
        model = E.fit_gbm(train, feats, h, ctx["params"])

        # Fit the blend weight on the last 2 years, held out from the fit. The inner model
        # is fitted on the training frame; the holdout comes from the frame being predicted,
        # because that is the series whose persistence baseline the weight has to balance.
        inner_cut = cutoff - pd.DateOffset(years=2)
        inner_train = D.embargoed_train(ctx["train_sub"], h, inner_cut)
        sub = ctx["sub"]
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
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
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
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
        gbm = E.fit_gbm(train, feats, h, ctx["params"]).predict(test)
        har = E.fit_har(train, feats, h).predict(test)
        return 0.5 * gbm + 0.5 * har


class MLPMethod(Incumbent):
    """A small two-layer network instead of the gradient-boosted trees.

    Two independent results motivate it. Christensen, Siggaard & Veliyev report a two-layer
    network as the best single model for realized volatility once the feature set is wider
    than plain HAR; Zhang, Cucuringu et al. report neural nets dominating linear and
    tree-based models specifically on *pooled* cross-sections of equities. The sector panel
    here is exactly that setting, so this is a replication rather than a guess.

    Its capacity is tuned on the same frozen validation slice the GBM uses.
    """

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
        return E.fit_mlp(train, feats, h, ctx.get("mlp_params")).predict(test)


class EnsembleMethod(Incumbent):
    """Equal-weight average of HAR, HARQ, SHAR, ridge and the GBM.

    The bake-off already contains a two-member average (`har+gbm`); this is the full-ladder
    version. Combination is the single most reliably replicated improvement in this
    literature, and these five err in different kinds — linear vs flexible, level vs
    measurement-error-corrected vs leverage-corrected — which is the condition under which
    averaging helps rather than just smoothing.
    """
    weights = "equal"

    def fit_predict(self, panel, h, cutoff, test, ctx):
        feats = self.features(panel)
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
        return E.fit_ensemble(train, feats, h, ctx["params"],
                              weights=self.weights).predict(test)


class EnsembleNNLS(EnsembleMethod):
    """Same members, weights fitted by non-negative least squares on an inner holdout."""
    weights = "nnls"


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
        test = test.sort_values("date")
        preds = np.full(len(test), np.nan)
        dates = test["date"].to_numpy()

        # Re-estimate at each chunk boundary on the trailing window ending at that date.
        starts = list(range(0, len(test), self.refit_days))
        for s in starts:
            e = min(s + self.refit_days, len(test))
            asof = pd.Timestamp(dates[s])
            train = D.embargoed_train(ctx["train_sub"], h, asof)
            if self.window_days:
                train = train[train["date"] >= asof - pd.Timedelta(days=int(
                    self.window_days * 1.45))]      # calendar days ~ trading days / 0.69
            if len(train) < 250:
                continue
            if self.estimator == "gbm":
                m = E.fit_gbm(train, feats, h, ctx["params"])
            else:
                m = E.fit_by_name(self.estimator, train, feats, h, ctx.get("params"))
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
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
        return E.fit_harq(train, None, h).predict(test)


class SHARMethod(Incumbent):
    """HAR on signed semivariance (Patton & Sheppard 2015)."""

    def fit_predict(self, panel, h, cutoff, test, ctx):
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
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
        train = D.embargoed_train(ctx["train_sub"], h, cutoff)
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
    MLPMethod("mlp", "two-layer neural network (Christensen et al.; Zhang et al.)"),
    EnsembleMethod("ensemble", "equal-weight average of HAR, HARQ, SHAR, ridge, GBM"),
    EnsembleNNLS("ensemble-nnls", "same members, non-negative least-squares weights"),
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
    """Shared per-(panel, horizon) setup: labelled frames and tuned parameters.

    Two frames, always: `train_sub` is what a model is fitted on, `sub` is what it is
    scored on. They are the same object's rows for a panel that trains on its own history
    and genuinely different when it trains pooled.
    """
    feats = method.features(panel)
    ctx: dict = {}

    if isinstance(method, Pooled):
        pool = D.labelled(build_pooled_panel(), feats, h)
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
        ctx["sub"] = ctx["train_sub"] = ctx["sub_pooled"]
    else:
        tr_sub, sub = fit_and_eval_frames(panel, feats, h)
        ctx["sub"], ctx["train_sub"] = sub, tr_sub
        fit = D.embargoed_train(tr_sub, h, pd.Timestamp(config.VAL_START))
        val = D.time_slice(sub, config.VAL_START, config.VAL_END)
        ctx["params"] = (E.tune_gbm(fit, val, feats, h, quiet=True)
                         if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)
        if isinstance(method, MLPMethod):
            ctx["mlp_params"] = (E.tune_mlp(fit, val, feats, h, quiet=True)
                                 if len(fit) > 200 and len(val) > 100 else E.MLP_DEFAULT)
    return ctx


def run_method(method: Method, panel: str, h: int, years: list[int],
               per_year: int = 1) -> dict:
    ctx = _context(panel, h, method)
    sub, train_sub = ctx["sub"], ctx["train_sub"]
    rows, obs = [], []
    key_cols = ["date"] + (["ticker"] if panel == "sector" else [])
    min_test = max(60 // per_year, 30)
    for cutoff, end in fold_edges(years, per_year):
        test = D.time_slice(sub, cutoff, end)
        train_n = len(D.embargoed_train(train_sub, h, cutoff))
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
        # Per-observation loss, keyed so the MCS can align methods whose feature sets drop
        # different rows. Aligning on position instead would silently compare day i of one
        # method with day j of another.
        piece = test[key_cols].copy()
        piece["loss"] = M.qlike_obs(y, pred)
        obs.append(piece)

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
        "_per_fold_qlike": f["qlike_gain"].to_numpy(),
        "_obs_loss": (pd.concat(obs, ignore_index=True).set_index(key_cols)["loss"]
                      if obs else None),
        "_years": years[:len(f)],
    }


def paired_verdict(cand: dict, base: dict, key: str = "_per_fold") -> dict:
    """Paired fold-by-fold comparison — see `metrics.paired_test`, the only implementation."""
    return M.paired_test(cand[key], base[key])


def _mcs_table(rows: list[dict], panel: str, h: int) -> pd.DataFrame:
    """Which methods cannot be distinguished from the best, under QLIKE."""
    series = {r["method"]: r["_obs_loss"] for r in rows if r["_obs_loss"] is not None}
    if len(series) < 2:
        return pd.DataFrame()
    # Inner join: the MCS compares models on the observations they all scored.
    losses = pd.concat(series, axis=1, join="inner").dropna()
    mcs = M.model_confidence_set(losses)
    mcs.insert(0, "panel", panel)
    mcs.insert(1, "horizon", h)
    kept = mcs[mcs["in_mcs"]]["model"].tolist()
    print(f"\n  Model Confidence Set (QLIKE, alpha=0.10, {len(losses):,} shared "
          f"observations): {len(kept)} of {len(mcs)} methods survive")
    print(f"    {', '.join(kept)}")
    return mcs


GRANULARITIES = [("annual", 1), ("semi-annual", 2)]


def bake_off(panel: str, h: int | None = None,
             years: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every method at both fold granularities; adopt only if both agree.

    This used to run annual folds alone, which quietly made it the one place in the project
    that did *not* apply the rule the rest of it advertises. It mattered: `jumps` clears the
    bar decisively on annual folds (t = +3.26 on skill, +2.91 on QLIKE) and collapses to
    about +0.3 when the same window is cut in half. A table printing only the first number
    publishes an ADOPT for a method this project has already rejected twice, for exactly
    that reason.
    """
    h = h or config.HEADLINE_HORIZON
    years = years or [y for y in config.WALK_YEARS if y < int(config.TEST_START[:4])]

    tables, mcs = [], pd.DataFrame()
    for label, per_year in GRANULARITIES:
        print(f"\n{'=' * 78}\n=== Method bake-off: {panel} panel, h={h}, "
              f"{label} folds {years[0]}-{years[-1]} (pre-test only)\n{'=' * 78}")
        rows = []
        for m in METHODS:
            try:
                r = run_method(m, panel, h, years, per_year)
            except Exception as exc:
                print(f"  {m.name:<14} FAILED: {exc}")
                continue
            r["granularity"] = label
            rows.append(r)
            print(f"  {r['method']:<14} skill {r['skill_mean']:+.3f} "
                  f"+/- {r['skill_std']:.3f}  (SE {r['skill_se']:.3f})  "
                  f"QLIKE gain {r['qlike_gain']:+.4f}  dir {r['dir_accuracy']:.3f}  "
                  f"{r['years_positive']}/{r['n_folds']} folds+")

        base = next((r for r in rows if r["method"] == "incumbent"), None)
        if base is not None:
            print(f"\n  Paired vs incumbent (same folds; t > {M.ADOPT_T} on skill AND QLIKE):")
            for r in rows:
                if r["method"] == "incumbent":
                    r["adopts"] = False
                    continue
                v = paired_verdict(r, base)
                vq = paired_verdict(r, base, "_per_fold_qlike")
                r.update({f"paired_{k}": val for k, val in v.items()})
                r.update({f"paired_qlike_{k}": val for k, val in vq.items()})
                r["adopts"] = M.adopts(v) and M.adopts(vq)
                print(f"    {'ADOPT' if r['adopts'] else 'hold':<6} {r['method']:<14} "
                      f"delta {v['delta']:+.3f} +/- {v['delta_se']:.3f}  t={v['t']:+.2f}  "
                      f"t_qlike={vq['t']:+.2f}  wins {v['wins']}/{v['n']}")

        # The MCS is reported once, on annual folds: it is a statement about per-observation
        # loss, and both granularities score the very same observations.
        if per_year == 1:
            mcs = _mcs_table(rows, panel, h)

        tables.append(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                                    for r in rows]))

    out = pd.concat(tables, ignore_index=True)
    if not mcs.empty:
        out = out.merge(mcs[["model", "mcs_pvalue", "in_mcs"]],
                        left_on="method", right_on="model", how="left").drop(columns="model")

    # The verdict is the intersection: a method ships only if it clears the bar everywhere.
    passed = out.groupby("method")["adopts"].all()
    winners = [m for m, ok in passed.items() if ok]
    print(f"\n  === {panel}: adopted at BOTH granularities ===")
    print(f"    {', '.join(winners) if winners else 'nothing - the incumbent stands'}")
    split = [m for m in out[out["adopts"]]["method"].unique() if m not in winners]
    if split:
        print(f"    (cleared at ONE granularity only, therefore not adopted: "
              f"{', '.join(split)})")
    return out.sort_values(["granularity", "skill_mean"],
                           ascending=[True, False]).reset_index(drop=True), mcs


def main() -> None:
    config.ensure_dirs()
    tables, mcs_tables = [], []
    for p in ("market", "sector"):
        out, mcs = bake_off(p)
        tables.append(out)
        if not mcs.empty:
            mcs_tables.append(mcs)
    path = config.REPORT_DIR / "method_bakeoff.csv"
    pd.concat(tables, ignore_index=True).to_csv(path, index=False)
    print(f"\nWrote -> {path}")
    if mcs_tables:
        mpath = config.REPORT_DIR / "model_confidence_set.csv"
        pd.concat(mcs_tables, ignore_index=True).to_csv(mpath, index=False)
        print(f"Wrote -> {mpath}")


if __name__ == "__main__":
    main()
