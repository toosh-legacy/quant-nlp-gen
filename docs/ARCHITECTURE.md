# Architecture

How the pipeline is laid out, and why each boundary is where it is.

```
volradar/
  config.py              single source of truth: tickers, horizons, split dates, selected families
  cli.py                 one entrypoint for every stage
  data/
    fetch.py             download + cache raw series, with a coverage guard
                         (prices from Yahoo; the vol complex from Cboe, Yahoo as fallback)
    realized.py          Oxford-Man Realized Library (intraday measures, one cached file)
  features/
    core.py              per-series vol features: HAR, range (PK/GK/RS/YZ), jump, semivar
    factors.py           the candidate external factor families
    assemble.py          market, sector and pooled panels; train-vs-eval frame split
  models/
    dataset.py           splitting, embargo, fold edges, regime thresholds  ← honesty layer
    estimators.py        the ladder: persistence → HAR → HARQ → SHAR → ridge → gbm →
                         mlp → ensemble
    conformal.py         calibrated prediction intervals, adaptively corrected
    train.py             production fit + the snapshot the dashboard reads
  backtest/
    metrics.py           every metric, defined once — QLIKE, the paired adoption test,
                         and the Model Confidence Set
    evaluate.py          fixed split, pooled OOS, walk-forward, per-sector
    oos.py               walk-forward out-of-sample prediction series + intervals
    select_features.py   which factor families ship (decided pre-test)
    ablation.py          per-family walk-forward R2 -- descriptive only
    methods.py           methodology bake-off, paired against the incumbent
    settle.py            decides open model questions on pre-test folds
    intraday_study.py    is intraday data worth paying for? (scoped, 2000-2016)
    events.py            did it flag real volatility spikes?
    alarm.py             can a dedicated stress head beat thresholding the mean forecast?
    decision.py          would acting on it have helped?
  simulate/
    dgp.py               synthetic markets with a KNOWN true volatility
    validate.py          ceiling, model-vs-truth, placebo and leakage tests
    scenarios.py         Monte Carlo strategy testing, incl. a perfect-foresight arm
  api/server.py          FastAPI — reads artifacts, computes nothing
  sentiment/             optional DistilBERT headline classifier (not a model input)
frontend/                React + Vite dashboard
tests/                   causality, embargo, metrics, conformal, DGP, API contract
```

## Data flow

```
fetch ──> data/raw/*.parquet
             │
          build ──> data/build/panel_{market,sector}.parquet
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      train        evaluate         oos ──> events, decision
        │              │                        │
        ▼              ▼                        ▼
   snapshot.json   *.csv reports          reports + curves
        └──────────────┴────────────────────────┘
                       │
                    api/server ──> frontend
                       ▲
   RSS feeds ──> news/ (tag, score) ──> news_radar.json + data/news/ archive
```

The news branch reads `snapshot.json` (to compare news and forecast ranks) but never writes
to anything the forecast reads — it is an overlay by construction, not by convention.

Every stage caches to disk and is idempotent. Re-running a later stage never re-downloads.
`--force` rebuilds a stage from scratch.

## The four rules the layout enforces

**1. Configuration lives in exactly one file.**
Split boundaries, horizons, tickers and the selected feature families are all in
`config.py`. The no-lookahead discipline is auditable because there is one place to check
that `TEST_START` is an absolute date and nothing downstream can move it.

**2. The honesty layer is isolated and tested alone.**
`models/dataset.py` holds splitting, embargoing and regime thresholds — three small
functions whose bugs are silent. An off-by-one in the embargo does not crash; it just
inflates every number in the README by a little. So they live apart with dedicated tests.

**3. The API computes nothing.**
`api/server.py` reads files from `data/reports/` and reshapes them for display. It contains
no modelling logic, so the dashboard physically cannot show a figure that disagrees with
the backtest. A missing report returns 503 with an instruction, never an improvised default.

**4. Selection never sees the reporting window.**
Hyperparameters are tuned on a validation slice inside the training period. Factor families
are chosen on walk-forward folds strictly before `TEST_START`. Both are chronologically
prior to anything reported.

**5. One adoption bar, implemented once.**
Every "should we change this?" question in the project — a feature family, a method, a
model, an alarm rule — is answered the same way: score the candidate and the incumbent on
identical pre-`TEST_START` folds, take the paired fold-by-fold difference, and adopt only at
`t > 2` at both an annual and a semi-annual fold slicing. That test lives in
`metrics.paired_test` and nowhere else. It used to live in three places with a fourth,
different rule in `select_features.py`, which meant the answer depended on which file asked
the question.

## Why two panels

The market panel is one row per day (SPY); the sector panel is one row per day per sector
ETF. They answer different questions and have very different sample sizes — ~5,400 rows
versus ~54,000 — which matters, because the market model has far less data to fit and is
correspondingly easier to overfit. Keeping them separate means that constraint is visible
rather than averaged away.

## Why there is a simulator

Real data cannot answer "how much of our error is irreducible?", because true volatility is
never observed. `simulate/` writes down a process, so σₜ is known and the same pipeline can
be scored against truth and against the noisy proxy. It also provides the only honest test of
the evaluation harness itself: run it on a market with no forecastable volatility and it must
report nothing. That test is what caught the fact that skill-versus-persistence reads +0.43
on pure noise — a property of the metric that is invisible on real data.

## Extending it

- **A new factor family**: add a builder in `features/factors.py`, register it in
  `feature_families()`, and add it to the shortlist in `select_features.py`. **The selection
  study decides**, not the ablation: `ablation.py` averages per-year R², whose denominator is
  each year's own variance, and its fold-to-fold spread (~0.35) dwarfs every difference it
  reports (<0.06). Read it for direction, never for a verdict. Do not add anything to
  `SELECTED_FAMILIES` by hand.
- **A new model**: add a `fit_*` function in `models/estimators.py` and append it to
  `LADDER`. Every backtest picks it up automatically. If it takes tuned parameters, add its
  name to `TUNED` so `fit_by_name` passes them.
- **A new family that starts later than the others**: give it its own entry rather than
  folding it into a neighbour. Every column is required by `dataset.labelled`, so a series
  beginning in 2011 deletes all training rows before 2011 — including 2008. That cost has to
  be priced separately from the family's information content, which is why the VIX complex
  is split into `vix` and `vix_short`.
- **A new asset class**: add its tickers to `config.py` and build a third panel. The
  feature and evaluation layers are asset-agnostic.

## Operational notes

- **Refreshing**: `python -m volradar.cli fetch --force && python -m volradar.cli build
  --force && python -m volradar.cli train` brings the dashboard to the latest close without
  re-running the full backtest.
- **The coverage guard**: providers silently truncate series. `data/fetch.py` compares each
  ticker's row count against its peers, retries short ones individually, and warns loudly
  if they stay short. This is not defensive padding — a truncated ^TNX response once nulled
  two features and dropped every macro row from the panel without raising anything.
- **Reproducibility**: one seed in `config.RANDOM_SEED`; model artifacts carry their
  training window, feature list and parameters.
