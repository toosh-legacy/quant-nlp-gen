"""The model zoo: an escalating ladder of forecasters, weakest first.

The ladder exists so no result can be reported without context. A number is only
interesting relative to what the previous rung already achieved:

1. **persistence** — next log-vol equals current log-vol. Free, and genuinely hard to beat
   because volatility is extremely persistent. Any model that cannot beat this is noise.
2. **HAR** — Ridge on the four multi-scale realized-vol components (Corsi 2009). The
   standard academic benchmark; frequently beats far fancier models.
3. **ridge** — Ridge on the full feature set. Isolates what the extra factor families add
   while holding model capacity fixed and linear.
4. **gbm** — gradient-boosted trees on the full feature set. Adds nonlinearity and
   interactions; also the rung most able to overfit, which the walk-forward will expose.

Every estimator implements the same two-method interface so the backtest code never
branches on model type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from volradar import config

HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q"]

# Fixed GBM settings (not tuned): shared across every fit so the tuning grid only varies
# the three parameters that actually matter for capacity.
GBM_FIXED = dict(subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                 random_state=config.RANDOM_SEED, n_jobs=-1, tree_method="hist")
GBM_DEFAULT = dict(n_estimators=400, max_depth=3, learning_rate=0.05)

# Deliberately conservative grid. The market panel has only ~4k daily rows, so deep trees
# memorize; capping depth at 4 is a modelling decision made on the validation slice, not a
# convenience.
GBM_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in (300, 600) for d in (2, 3, 4) for lr in (0.03, 0.05)
]


@dataclass
class Fitted:
    """A trained model plus everything needed to reproduce and serve it."""
    name: str
    columns: list[str]
    estimator: object = None
    scaler: object = None
    params: dict = field(default_factory=dict)

    def predict(self, df) -> np.ndarray:
        if self.name == "persistence":
            return df[self.columns[0]].to_numpy(float)
        X = df[self.columns].to_numpy(float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(self.estimator.predict(X), dtype=float)


def fit_persistence(_train, _features, h: int) -> Fitted:
    """No fitting required — the prediction IS the current realized vol."""
    return Fitted(name="persistence", columns=[config.persist_col(h)])


def _fit_ridge(train, columns, h, name, alpha=1.0) -> Fitted:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X = train[columns].to_numpy(float)
    y = train[config.fwd_col(h)].to_numpy(float)
    sc = StandardScaler().fit(X)
    est = Ridge(alpha=alpha).fit(sc.transform(X), y)
    return Fitted(name=name, columns=list(columns), estimator=est, scaler=sc,
                  params={"alpha": alpha})


def fit_har(train, _features, h: int) -> Fitted:
    return _fit_ridge(train, HAR_COLS, h, "HAR")


def fit_ridge(train, features, h: int) -> Fitted:
    return _fit_ridge(train, features, h, "ridge")


def fit_gbm(train, features, h: int, params: dict | None = None) -> Fitted:
    from xgboost import XGBRegressor

    p = dict(params or GBM_DEFAULT)
    est = XGBRegressor(**GBM_FIXED, **p)
    est.fit(train[features].to_numpy(float), train[config.fwd_col(h)].to_numpy(float))
    return Fitted(name="gbm", columns=list(features), estimator=est, params=p)


def tune_gbm(fit_df, val_df, features: list[str], h: int, quiet: bool = False) -> dict:
    """Pick GBM capacity on a held-out time slice of the training period.

    Never touches the test period. The validation slice is itself chronologically after
    the fitting slice, so the selection mimics how the model will actually be used.
    """
    from sklearn.metrics import r2_score

    y_val = val_df[config.fwd_col(h)].to_numpy(float)
    best, best_score = GBM_DEFAULT, -np.inf
    for p in GBM_GRID:
        pred = fit_gbm(fit_df, features, h, p).predict(val_df)
        s = r2_score(y_val, pred)
        if s > best_score:
            best_score, best = s, p
    if not quiet:
        print(f"    h={h:>2} tuned {best}  (val R2={best_score:.3f})")
    return best


class FittedHARQ(Fitted):
    """HAR with a measurement-error-dependent daily coefficient.

    The daily coefficient becomes time-varying:

        alpha_1,t = alpha_d + alpha_Q * (sqrt(RQ)_t - mean)

    implemented by handing the linear model an extra interaction column. When realized
    quarticity is high — today's volatility estimate is noisy — alpha_Q (typically negative)
    pulls the daily term down, so the forecast leans on the smoother weekly and monthly
    components instead. When the estimate is clean it leans in.

    The RQ mean and scale come from the TRAINING rows only; using full-sample moments here
    would leak the future into a standardisation, which is the quiet way this model gets
    reported wrong.
    """

    def predict(self, df):
        X = _harq_design(df, self.params["rq_mu"], self.params["rq_sd"])
        return np.asarray(self.estimator.predict(self.scaler.transform(X)), dtype=float)


def _harq_design(df, mu: float, sd: float) -> np.ndarray:
    base = df[HAR_COLS].to_numpy(float)
    z = ((df["rq_sqrt"].to_numpy(float) - mu) / sd).reshape(-1, 1)
    interaction = base[:, [0]] * z          # daily lag x standardised sqrt(RQ)
    return np.hstack([base, interaction])


def fit_harq(train, _features, h: int) -> Fitted:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    mu = float(train["rq_sqrt"].mean())
    sd = float(train["rq_sqrt"].std()) or 1.0
    X = _harq_design(train, mu, sd)
    y = train[config.fwd_col(h)].to_numpy(float)
    sc = StandardScaler().fit(X)
    est = Ridge(alpha=1.0).fit(sc.transform(X), y)
    return FittedHARQ(name="HARQ", columns=HAR_COLS + ["rq_sqrt"], estimator=est,
                      scaler=sc, params={"rq_mu": mu, "rq_sd": sd})


def qlike_objective(y_true: np.ndarray, y_pred: np.ndarray):
    """Custom XGBoost objective: train directly on QLIKE instead of squared error.

    On log daily volatility, QLIKE is  L(y, p) = 2p + exp(2(y - p)) , giving

        dL/dp   = 2 - 2*exp(2(y - p))
        d2L/dp2 = 4*exp(2(y - p))

    Why bother: squared error is symmetric, so it is indifferent between forecasting 12%
    when the truth is 20% and forecasting 28% when the truth is 20%. For a risk forecast
    those errors are not equally bad, and the asymmetry is also the mechanism behind two of
    this project's known weaknesses — forecasts shrink toward the mean, which makes stress
    alarms fire late and flattens the exposure signal that the volatility-targeting
    simulation needs. QLIKE punishes under-forecasting harder, so the fitted model leans
    the other way.

    Two numerical guards, both necessary — without them this objective diverges instead of
    training, which is exactly what it did on the first attempt (final loss ~5e8):

    * the exponent is clipped, so one wild residual cannot produce an enormous gradient;
    * **the Hessian is floored.** As the prediction rises above the truth the Hessian decays
      to zero, and boosting's step is grad/hess, so a vanishing Hessian produces an
      unbounded step and the fit runs away. Flooring it caps the step size.

    The starting point matters too, and is handled by the caller: log daily volatility sits
    near -4, while XGBoost's default base score is 0.5, so an unset base score starts the
    optimisation deep in the exponential's stiff region.
    """
    d = np.clip(2.0 * (y_true - y_pred), -6.0, 6.0)
    e = np.exp(d)
    grad = 2.0 - 2.0 * e
    hess = np.maximum(4.0 * e, 0.5)
    return grad, hess


def fit_gbm_qlike(train, features, h: int, params: dict | None = None) -> Fitted:
    """GBM trained on the QLIKE objective, started at the target's own mean."""
    from xgboost import XGBRegressor

    p = dict(params or GBM_DEFAULT)
    y = train[config.fwd_col(h)].to_numpy(float)
    est = XGBRegressor(**GBM_FIXED, **p, objective=qlike_objective,
                       base_score=float(np.mean(y)))
    est.fit(train[features].to_numpy(float), y)
    return Fitted(name="gbm-qlike", columns=list(features), estimator=est, params=p)


def fit_gbm_quantile(train, features, h: int, params: dict | None = None,
                     alpha: float = 0.6) -> Fitted:
    """GBM fitted to a conditional QUANTILE of log volatility rather than its mean.

    A squared-error model targets the conditional mean, and the conditional mean of a
    right-skewed variable sits below the outcomes that matter for risk. Forecasting the
    60th or 70th percentile instead deliberately biases the forecast upward, which is the
    direction a risk user wants to be wrong in.

    Scored on R² this will look *worse* by construction — it is not estimating the mean —
    so it is judged on QLIKE and on whether it detects stress episodes earlier.
    """
    from xgboost import XGBRegressor

    p = dict(params or GBM_DEFAULT)
    est = XGBRegressor(**GBM_FIXED, **p, objective="reg:quantileerror",
                       quantile_alpha=alpha)
    est.fit(train[features].to_numpy(float), train[config.fwd_col(h)].to_numpy(float))
    return Fitted(name=f"gbm-q{int(alpha * 100)}", columns=list(features), estimator=est,
                  params={**p, "alpha": alpha})


SHAR_COLS = ["log_rs_down_m", "log_rs_up_m", "log_rv_w", "log_rv_m", "log_rv_q"]


def fit_shar(train, _features, h: int) -> Fitted:
    """HAR with realized variance split into its down-day and up-day components."""
    return _fit_ridge(train, SHAR_COLS, h, "SHAR")


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
# Deliberately small. Christensen, Siggaard & Veliyev find a two-layer network is the best
# single model for realized volatility once the feature set is wider than plain HAR, and
# Zhang, Cucuringu et al. find neural nets dominate linear and tree models specifically on
# *pooled* cross-sections of series — which is exactly how the sector panel trains. Both
# results come from networks of this size, not large ones; capacity is not the binding
# constraint on a few hundred thousand rows of daily data.
MLP_DEFAULT = dict(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3)
MLP_GRID = [
    {"hidden_layer_sizes": hl, "alpha": a, "learning_rate_init": 1e-3}
    for hl in ((16,), (32, 16), (64, 32))
    for a in (1e-4, 1e-3, 1e-2)
]


def fit_mlp(train, features, h: int, params: dict | None = None) -> Fitted:
    """A small two-layer network on the standardised feature set.

    Standardisation is not optional for a network the way it is optional for trees, so this
    reuses the same scaler-in-the-Fitted pattern the ridge models use. `early_stopping`
    holds out the last slice of the training rows internally, which keeps the fit from
    running away on the folds where training data is thin.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    # The ladder hands every tuned rung the same `params` object, and for most callers that
    # object holds GBM capacity settings. Keep only the keys this estimator understands, so
    # a caller that has not tuned an MLP gets the default rather than a TypeError.
    p = {k: v for k, v in dict(params or {}).items() if k in MLP_DEFAULT} or dict(MLP_DEFAULT)
    X = train[features].to_numpy(float)
    y = train[config.fwd_col(h)].to_numpy(float)
    sc = StandardScaler().fit(X)
    # max_iter is a safety stop, not the training schedule — `early_stopping` ends the fit
    # when a held-out slice stops improving, which is what should terminate it. At 400 the
    # cap was binding on every market-panel fold (sklearn raised ConvergenceWarning each
    # time), so the bake-off was comparing a truncated network against fully-fitted rivals
    # and reporting the result as "neural nets lose here". That is not a fair test.
    est = MLPRegressor(**p, max_iter=2000, early_stopping=True, n_iter_no_change=20,
                       random_state=config.RANDOM_SEED).fit(sc.transform(X), y)
    return Fitted(name="mlp", columns=list(features), estimator=est, scaler=sc, params=p)


def tune_mlp(fit_df, val_df, features: list[str], h: int, quiet: bool = False) -> dict:
    """Pick network size and regularisation on the same held-out slice `tune_gbm` uses."""
    from sklearn.metrics import r2_score

    y_val = val_df[config.fwd_col(h)].to_numpy(float)
    best, best_score = MLP_DEFAULT, -np.inf
    for p in MLP_GRID:
        try:
            pred = fit_mlp(fit_df, features, h, p).predict(val_df)
        except Exception:
            continue
        s = r2_score(y_val, pred)
        if s > best_score:
            best_score, best = s, p
    if not quiet:
        print(f"    h={h:>2} tuned MLP {best}  (val R2={best_score:.3f})")
    return best


# ---------------------------------------------------------------------------
# Forecast combination
# ---------------------------------------------------------------------------
class FittedEnsemble(Fitted):
    """A weighted average of several fitted models."""

    def predict(self, df) -> np.ndarray:
        members = self.params["members"]
        weights = self.params["weights"]
        return np.sum([w * m.predict(df) for m, w in zip(members, weights)], axis=0)


ENSEMBLE_MEMBERS = ["HAR", "HARQ", "SHAR", "ridge", "gbm"]


def fit_ensemble(train, features, h: int, params: dict | None = None,
                 members: list[str] | None = None, weights: str = "equal") -> Fitted:
    """Combine the ladder's arms into one forecast.

    Equal-weight combination is one of the most reliably replicated results in forecasting
    — the "forecast combination puzzle" is that it is so hard to beat with fitted weights.
    The reason applies squarely here: these models err in different *kinds*. HAR is linear
    and stable but cannot represent an interaction; the GBM is flexible but fits noise;
    HARQ leans on the measurement-error correction; SHAR on the leverage effect. Averaging
    cancels a large part of the idiosyncratic error while keeping the shared signal.

    `weights="nnls"` fits non-negative least-squares weights on an inner holdout — the last
    year of the training rows, held out of the members' own fits — instead of averaging.
    Non-negative and sum-to-one because unconstrained regression on correlated forecasts
    produces enormous offsetting weights that do not survive out of sample.
    """
    names = members or ENSEMBLE_MEMBERS
    ladder = dict(LADDER)

    def _fit_members(rows):
        out = []
        for n in names:
            fn = ladder[n]
            out.append(fn(rows, features, h, params) if n in ("gbm", "mlp")
                       else fn(rows, features, h))
        return out

    fitted = _fit_members(train)
    w = np.full(len(fitted), 1.0 / len(fitted))

    if weights == "nnls":
        cut = train["date"].max() - pd.DateOffset(years=1)
        inner = train[train["date"] < cut]
        hold = train[train["date"] >= cut]
        if len(inner) > 300 and len(hold) > 60:
            from scipy.optimize import nnls
            P = np.column_stack([m.predict(hold) for m in _fit_members(inner)])
            y = hold[config.fwd_col(h)].to_numpy(float)
            sol, _ = nnls(P, y)
            if sol.sum() > 0:
                w = sol / sol.sum()

    return FittedEnsemble(name=f"ensemble-{weights}", columns=list(features),
                          params={"members": fitted, "weights": w,
                                  "member_names": names})


# The ladder, in reporting order.
LADDER = [
    ("persistence", fit_persistence),
    ("HAR", fit_har),
    ("HARQ", fit_harq),
    ("SHAR", fit_shar),
    ("ridge", fit_ridge),
    ("gbm", fit_gbm),
    ("mlp", fit_mlp),
    ("ensemble", fit_ensemble),
]

# The rungs whose capacity is tuned; everything else ignores the `params` argument.
TUNED = {"gbm", "mlp", "ensemble"}


def fit_by_name(name: str, train, features: list[str], h: int,
                params: dict | None = None) -> Fitted:
    """Fit one rung of the ladder by name.

    Exists so no caller has to remember which estimators take tuned parameters — that list
    grew, and each caller having its own `if name == "gbm"` was one rung away from silently
    fitting an untuned model.
    """
    fn = dict(LADDER)[name]
    return fn(train, features, h, params) if name in TUNED else fn(train, features, h)
