# Architecture

How the pipeline is laid out, and why each boundary is where it is.

```
volradar/
  config.py              single source of truth: tickers, horizons, split dates, selected families
  cli.py                 one entrypoint for every stage
  data/fetch.py          download + cache raw series, with a coverage guard
  features/
    core.py              per-series volatility features, jump block, forward targets
    factors.py           the four candidate external factor families
    assemble.py          market, sector and pooled panels; train-vs-eval frame split
  models/
    dataset.py           splitting, embargo, regime thresholds  ← the honesty layer
    estimators.py        the model ladder: persistence → HAR → ridge → gbm
    conformal.py         calibrated prediction intervals, adaptively corrected
    train.py             production fit + the snapshot the dashboard reads
  backtest/
    metrics.py           every metric, defined once (incl. QLIKE)
    evaluate.py          fixed split, pooled OOS, walk-forward, per-sector
    oos.py               walk-forward out-of-sample prediction series + intervals
    select_features.py   which factor families ship (decided pre-test)
    ablation.py          what each family is worth
    methods.py           methodology bake-off, paired against the incumbent
    events.py            did it flag real volatility spikes?
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
```

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
  `feature_families()`, and add it to the shortlist in `select_features.py`. The ablation
  and selection will tell you whether it earns a place; do not add it to
  `SELECTED_FAMILIES` by hand.
- **A new model**: add a `fit_*` function in `models/estimators.py` and append it to
  `LADDER`. Every backtest picks it up automatically.
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
