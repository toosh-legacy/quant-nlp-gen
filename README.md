# Volatility Radar

Forecasts how much the **market** and each of the **eleven S&P sectors** will move over the
next 5–10 trading days, and ships the track record that qualifies every number.

```bash
pip install -r requirements.txt
python -m volradar.cli all        # fetch, build, backtest, train, news
cd frontend && npm install && npm run build
python -m volradar.cli serve      # dashboard at http://127.0.0.1:8000
python -m volradar.cli refresh    # daily: latest close -> retrain -> news radar
```

---

## 1. Project summary

**What it predicts.** Realized volatility — the size of price movement over the next 5 or 10
trading days, expressed as an annualized percentage — for SPY and for the eleven SPDR sector
ETFs. It ships a calibrated uncertainty band with every forecast and a frozen
calm/normal/stressed regime label.

**What it deliberately does not predict.** Direction. Short-horizon direction is very close to
unpredictable; an earlier phase of this project measured it and got MCC ≤ 0.09, which is
noise. Volatility is a different problem: it clusters, it is strongly persistent, and it is
genuinely forecastable. The project pivoted on that evidence rather than on preference.

**The headline.** Pooled out-of-sample over **2012-01-03 → 2026-08-04**, with a model
retrained at the start of every year and scored only on the year that follows:

| | market (SPY) | sectors (11 ETFs) |
|---|:---:|:---:|
| out-of-sample days | 3,662 | 37,558 |
| **R² (log realized vol)** | **0.421** | **0.481** |
| error removed vs persistence | 35% | 35% |
| rise/fall accuracy | **70.5%** (base 53.5%) | **70.9%** (base 51.0%) |
| 80% interval, realized coverage | 78.0% | 78.5% |

**What that R² actually means.** 42% of the variance in log realized volatility, explained
out-of-sample. It should be read against a measured ceiling of **0.887**, not against 1.0:
realized volatility is a noisy estimate of a quantity never directly observed, and on
simulated markets a forecaster with *perfect knowledge of the future* scores only 0.887
against the same noisy yardstick. In plain terms, the typical forecast misses by **2.7
annualized volatility points** where the persistence baseline misses by 3.5, on a series whose
median level is 11.5.

**The governing principle.** Every number here is one a user could actually have seen on the
day. Where a result did not survive testing, it is reported as a result, not omitted — the
sections below on what failed are as load-bearing as the table above.

---

## 2. What this affects, and who it is for

Volatility is the input to position sizing, option pricing, risk limits and margin. A better
short-horizon volatility estimate propagates into all of them. This project targets the
first-order question — *how much movement should be planned for over the next two weeks?* —
and is honest about which downstream uses the evidence supports.

**Supported uses**

| question | how well it answers | evidence |
|---|---|---|
| Is the market about to get choppier? | well | 70.5% accurate out-of-sample, 15 consecutive years |
| Which sectors are structurally calm or fragile right now? | well | all 11 sectors beat the baseline individually; the ranking recovers the classic defensive/cyclical ordering from data alone |
| How much movement should I plan for? | well | annualized number with a band whose *measured* coverage is published, not its nominal one |
| How confident should I be in that number? | well | adaptive conformal intervals; realized coverage 78% against a nominal 80% |

**Explicitly unsupported uses**

- **Not a crash alarm.** At the pre-registered threshold it warned ahead of 4 of 15 stress
  episodes. Exogenous shocks — COVID-19, the 2024 yen carry unwind — are not foreshadowed in
  market data.
- **Not a trading signal.** Volatility targeting on the forecast beat the same strategy sized
  on persistence by +0.046 Sharpe, with a 95% interval of [−0.067, +0.156]. It spans zero.
- **Not a 21-day forecast.** Built and measured, deliberately withheld — see §6.

Audience: anyone who needs a defensible short-horizon risk number, and anyone evaluating
whether a forecasting result is real. The methodology is the transferable part.

---

## 3. Methodology at a high level

**The target.** Log realized volatility over the next *h* trading days, `h ∈ {5, 10, 21}`,
computed from daily close-to-close returns. Modelling in logs throughout is deliberate:
realized volatility is close to log-normal and far more persistent in log space, which is why
the HAR literature reports much higher R² there.

**The features.** The starting point is Corsi's HAR cascade — volatility at daily, weekly,
monthly and quarterly scales — extended with range-based estimators (Parkinson, Garman-Klass,
Rogers-Satchell, Yang-Zhang) that recover intraday information from daily bars, a downside
semivariance term, momentum and drawdown, and turnover. Ten further candidate families are
implemented and tested but **none ships**; see §6.

**The models.** An escalating ladder, so no result is reported without context:
`persistence → HAR → HARQ → SHAR → ridge → gbm → mlp → ensemble`. Gradient-boosted trees are
what ships. The sector model trains on a **pooled** cross-section of all 91 series (79 single
stocks, 11 sector ETFs, SPY) — 471k rows instead of 37k — because volatility dynamics are
broadly shared across equities and every feature is scale-free.

**How anything gets adopted.** One rule, implemented once in `metrics.paired_test`:

> Score the candidate and the incumbent on **identical folds strictly before the test start**.
> Take the **paired** fold-by-fold difference. Adopt only at **t > 2**, on **skill *and*
> QLIKE**, at **both** an annual and a semi-annual fold slicing.

Paired rather than marginal, because comparing two separate means lets an *unstable* incumbent
inflate its own replacement bar — the noisier the thing being replaced, the harder it becomes
to replace. Two granularities, because a result that appears at one calendar cut and vanishes
at another is not a result. Two metrics, because a change that improves one and harms the
other is telling you something specific rather than giving you a metric to choose from.

**How results are reported.** The **Model Confidence Set** (Hansen–Lunde–Nason) states which
models cannot be statistically separated from the best, instead of crowning the lowest loss in
a table whose top rows sit within noise of each other.

**Four independent checks that the harness itself works.** The whole pipeline is run on
simulated markets from a written-down stochastic-volatility process where true volatility is
known exactly. This gives the accuracy ceiling, the model's score against *truth* rather than
the proxy, a market with no forecastable volatility (the harness must find nothing), and a
block-shuffled target (it must find nothing).

**One external check.** Everything above is internal. So the same models are also run on the
**Oxford-Man Realized Library** — 21 international indices, volatility measured from 5-minute
intraday returns — against Log-HAR, the reference model recent large-scale comparisons find
hardest to beat. Different data, different target, the literature's horizons.

---

## 4. Architecture overview

### 4.1 Repository layout

```
volradar/
  config.py              single source of truth: tickers, horizons, split dates, selected families
  cli.py                 one entry point for every stage
  data/
    fetch.py             download + cache raw series, with a coverage guard
    realized.py          Oxford-Man Realized Library (intraday measures, one cached file)
  features/
    core.py              per-series vol features: HAR, range (PK/GK/RS/YZ), jump, semivariance
    factors.py           the candidate external factor families
    assemble.py          market, sector and pooled panels; train-vs-eval frame split
  models/
    dataset.py           splitting, embargo, fold edges, regime thresholds  ← the honesty layer
    estimators.py        the model ladder
    conformal.py         calibrated prediction intervals, adaptively corrected
    train.py             production fit + the snapshot the dashboard reads
  backtest/
    metrics.py           every metric once — QLIKE, the paired test, the Model Confidence Set
    oos.py               walk-forward out-of-sample prediction series + intervals
    evaluate.py          fixed split, pooled OOS, walk-forward, per-sector
    select_features.py   which factor families ship (decided pre-test)
    methods.py           methodology bake-off, paired, both granularities, + MCS
    settle.py            decides open model questions on pre-test folds
    ablation.py          per-family walk-forward R² — descriptive only
    events.py            did it flag real volatility spikes?
    alarm.py             can a dedicated stress head beat thresholding the mean forecast?
    decision.py          would acting on it have helped?
    intraday_study.py    is intraday data worth paying for?
    benchmark.py         vs the literature, on the literature's own data
  simulate/              synthetic markets with a KNOWN true volatility; ceiling + leakage tests
  news/
    sources.py           24 free RSS feeds from news channels, fetched in parallel, de-duplicated
    semantics.py         sector / risk-theme / shock-language tagging (inspectable word lists)
    scoring.py           tone from the fine-tuned DistilBERT, lexicon fallback
    radar.py             market + per-sector news reading, archive, validation status
  sentiment/             DistilBERT fine-tuning + Hugging Face publish scripts
  api/server.py          FastAPI — reads artifacts, computes nothing
frontend/                React + Vite dashboard (radar, gauge, news panel, track record)
scripts/ui_check.py      Playwright end-to-end check of the running dashboard
tests/                   108 tests: causality, embargo, metrics, conformal, panels, fetch, API, news
```

### 4.2 Data flow

```
   fetch ────────> data/raw/*.parquet
                        │
                     build ────> data/build/panel_{market,sector,pooled}.parquet
                                      │
        ┌─────────────────────────────┼──────────────────────────┐
        ▼                             ▼                          ▼
     select                         oos ──> events, alarm, decision
   (which features ship)             │                          │
        │                            ▼                          ▼
        └──────────> evaluate ──> data/reports/*.csv  <──────────┘
                                      │
                  methods, settle, benchmark, simulate, ablation
                                      │
                                    train ──> snapshot.json
                                      │
                                 api/server ──> frontend
```

Every stage caches to disk and is idempotent; re-running a later stage never re-downloads.

### 4.3 Where the data comes from

All free, no API key:

| source | series | why this source |
|---|---|---|
| Yahoo Finance (`yfinance`) | SPY, 11 sector ETFs, 79 large caps, rates/credit/dollar | one dependency for everything carrying a volume figure |
| **Cboe daily-history CSVs** | VIX, VIX9D, VIX3M, VIX6M, VVIX, SKEW, VXN, RVX, OVX | the primary publisher, the whole term structure, and OHLC rather than close alone — which is what makes a range-based vol-of-vol measure possible |
| Oxford-Man Realized Library (archived) | 5-minute realized measures, 21 indices, 2000–2016 | the intraday study and the external benchmark only; the original domain no longer resolves, so this reads a 2016 GitHub vintage and never feeds a live forecast |

Prices from 2005 to today. Cboe is fetched over plain HTTP with the standard library and falls
back to Yahoo if unreachable, so an outage degrades a run instead of ending it.

### 4.4 How a model actually gets made

1. **Fetch** raw OHLCV and index series into `data/raw/`, guarded: any series returning far
   fewer rows than its peers is re-requested individually and warned about loudly.
2. **Build** three panels. `panel_market` is one row per trading day; `panel_sector` is one row
   per (day, ETF); `panel_pooled` stacks all 91 series with `is_index`/`is_sector` tags. All
   candidate factor families are merged into all three, so any study can test any family.
3. **Split honestly.** `models/dataset.py` is the only place train/test boundaries, the
   embargo and the regime thresholds are defined. A 10-day label spans 10 days, so any row
   whose outcome window crosses a boundary is *purged*, not leaked.
4. **Tune** capacity on a validation slice inside the training period (2015–2018), never on
   the test window.
5. **Walk forward.** For each year from 2012, fit on all prior embargoed data and predict that
   year alone. Concatenating the years gives a prediction series in which every point came
   from a model trained strictly before it.
6. **Calibrate intervals.** Split conformal on the tail of each fold's training data, then
   adaptive conformal (Gibbs & Candès) which steers width from realized misses — with an
   explicit *h*-day lag so today's band never uses an outcome that has not been revealed.
7. **Train production** on all labelled history and write `snapshot.json`, including
   per-feature contributions in annualized-vol points.

### 4.5 Built with

Python 3.11, pandas, NumPy, scikit-learn, XGBoost, SciPy, pyarrow, yfinance, FastAPI +
uvicorn. React + Vite + Recharts for the dashboard. pytest for the correctness layer. CPU
only — no GPU anywhere in the pipeline.

### 4.6 The four rules the layout enforces

1. **Configuration lives in exactly one file.** Split boundaries, horizons, tickers and the
   selected feature families are all in `config.py`, so the no-lookahead discipline is
   auditable from one place.
2. **The honesty layer is isolated and tested alone.** Splitting, embargoing and regime
   thresholds are three small functions whose bugs are silent — an off-by-one does not crash,
   it just quietly inflates every number in this file.
3. **The API computes nothing.** It reads `data/reports/` and reshapes for display, so the
   dashboard physically cannot show a figure that disagrees with the backtest. A missing
   report returns 503 naming the command to run, never an improvised default.
4. **Selection never sees the reporting window.** Hyperparameters are tuned inside the
   training period; feature families are chosen on folds strictly before the test start.

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
metric definitions: [`docs/METRICS.md`](docs/METRICS.md)

---

## 5. Running it locally

### Prerequisites

Python 3.11+, Node 20+ (only for the dashboard), and a network connection for the first fetch.
No API keys, no GPU.

### Setup

```bash
git clone <repo-url> && cd quant-nlp-gen

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Fastest path to a running dashboard

```bash
python -m volradar.cli fetch       # ~2 min, network
python -m volradar.cli build       # ~2 min
python -m volradar.cli oos         # ~15 min — the out-of-sample prediction series
python -m volradar.cli evaluate    # ~20 min
python -m volradar.cli train       # ~2 min — production fit + snapshot

cd frontend && npm install && npm run build && cd ..
python -m volradar.cli serve       # http://127.0.0.1:8000
```

### The full pipeline

```bash
python -m volradar.cli all         # every stage below, in dependency order
```

Budget several hours. `all` genuinely runs everything; it previously skipped four stages while
the docs claimed otherwise, which meant a reader following the quickstart never generated four
of the report files cited here.

| command | what it does |
|---|---|
| `fetch` | download raw series (network) |
| `build` | assemble market + sector + pooled panels |
| `select` | which factor families earn their place (decided pre-test) |
| `oos` | walk-forward out-of-sample prediction series |
| `evaluate` | fixed split, pooled OOS, walk-forward, per-sector |
| `ablation` | per-family walk-forward R² — descriptive only, see §6 |
| `methods` | methodology bake-off + Model Confidence Set |
| `settle` | decide open model questions on pre-test folds |
| `events` | did it flag real volatility spikes? |
| `alarm` | can a dedicated stress head beat the mean forecast? |
| `decision` | would acting on it have helped? |
| `simulate` | validate the method on known-truth synthetic markets |
| `intraday` | is intraday data worth paying for? |
| `benchmark` | vs the literature, on the literature's own data |
| `train` | production fit + dashboard snapshot |
| `news` | score news-channel headlines onto the radar (network); `--lexicon` skips DistilBERT |
| `refresh` | `fetch` + `build` + `train` + `news`, all forced — the daily update |
| `serve` | API + dashboard |

Useful flags: `--force` rebuilds a stage from scratch; `--market-only` skips the sector panel
in `ablation`, which is by far the most expensive stage.

To refresh the dashboard to the latest close without re-running the backtest:

```bash
python -m volradar.cli refresh     # fetch + build + train + news, ~1 minute
```

### Tests

```bash
pytest -q                          # 108 tests, ~10 seconds
python scripts/ui_check.py         # Playwright, against a running `serve`
```

Covering causality (including candidate feature blocks, not only shipped ones), the embargo,
split boundaries, metric correctness against hand-computed values, the paired adoption test
and the Model Confidence Set, conformal-interval causality, the download layer's coverage
guard and CSV parsing, the panel/feature-family contract, the synthetic-market generator, and
the API contract.

CI runs the test suite and builds the dashboard on every push. It does not run the pipeline —
that needs network access and a long compute budget.

---

## 6. Results

### 6.1 Accuracy

Pooled out-of-sample, 2012-01-03 → 2026-08-04. A model is retrained at the start of each year
on data available up to that point, then scored on the year that follows.

| | market (SPY) | sectors (11 ETFs) |
|---|:---:|:---:|
| out-of-sample days | 3,662 | 37,558 |
| **R² (log realized vol)** | **0.421** | **0.481** |
| persistence baseline R² | 0.111 | 0.208 |
| **error removed vs baseline** | **35%** | **35%** |
| rise/fall accuracy | **70.5%** (base 53.5%) | **70.9%** (base 51.0%) |
| rise/fall MCC | 0.409 | 0.417 |
| regime accuracy (3-class) | 64.7% (base 47.8%) | 66.3% (base 51.2%) |
| 80% interval, realized coverage | 78.0% | 78.5% |

At the 5-day horizon: market R² **0.446**, MCC **0.464**; sectors R² **0.426**, MCC **0.453**.

**In plain units.** Typical realized volatility over the window was 11.5% annualized.

| | median miss | 75th pct | 90th pct |
|---|---:|---:|---:|
| **model** | **2.71 vol points** | 4.87 | 8.29 |
| persistence baseline | 3.51 | 6.56 | 10.55 |

The model is closer than persistence on **60.8%** of the 3,662 days.

### 6.2 Stability — the number that matters most

Skill versus the persistence baseline, scored year by year so both sides face the same window:

| panel · horizon | mean ± std | worst year | years positive |
|---|:---:|:---:|:---:|
| market · 5d | +0.403 ± 0.123 | +0.184 | **15 / 15** |
| market · 10d | +0.354 ± 0.186 | +0.009 | **15 / 15** |
| sector · 5d | +0.407 ± 0.077 | +0.160 | **15 / 15** |
| sector · 10d | +0.358 ± 0.125 | −0.008 | 14 / 15 |

Every sector individually beats persistence — skill 0.290–0.406, direction accuracy 69–73%
across all eleven. No sector is carried by the others.

### 6.3 The model ladder

Fixed out-of-sample split at 10 days, so every result is quoted against a real benchmark
rather than against nothing:

| model | market R² | sector R² |
|---|---:|---:|
| persistence | 0.149 | 0.216 |
| HAR | 0.305 | 0.373 |
| HARQ | 0.303 | 0.373 |
| SHAR | 0.333 | 0.385 |
| ridge | **0.428** | 0.457 |
| gbm *(shipped)* | 0.412 | **0.463** |
| mlp | 0.361 | 0.447 |
| ensemble | 0.394 | 0.432 |

### 6.4 Tested against the literature, on the literature's data

21 international equity indices, 2000–2016, volatility measured from **5-minute intraday
returns**, at the horizons the HAR literature reports. Reference model: **Log-HAR**.

| model | beats Log-HAR, h=1 | h=5 | h=22 | R² (h=5) |
|---|---:|---:|---:|---:|
| **ridge** *(our features, linear)* | **20 / 21** | **19 / 21** | 13 / 21 | **0.413** |
| HAR-RS | 15 / 21 | 17 / 21 | 14 / 21 | 0.402 |
| Log-HAR *(reference)* | — | — | — | 0.392 |
| HARQ | 12 / 21 | 9 / 21 | 4 / 21 | 0.392 |
| HAR-J | 10 / 21 | 9 / 21 | 7 / 21 | 0.392 |
| gbm *(shipped)* | 13 / 21 | 8 / 21 | 4 / 21 | 0.345 |
| persistence | 0 / 21 | 3 / 21 | 2 / 21 | 0.282 |

**The feature set is the edge, not the tree model.** Linear ridge on our features beats the
field's reference on 20 of 21 indices at one day and 19 of 21 at one week — the strongest
externally-validated claim this project can make. The gradient-boosted model that ships does
*not*: competitive at one day, losing at longer horizons, and excluded outright by the Model
Confidence Set at one month.

The likely cause is specific rather than damning. Against precise intraday data the target is
much cleaner and close to linear in the HAR components, so flexibility buys noise; on its own
problem — a 10-day average of noisy *daily-bar* variance — the internal bake-off finds the
opposite. The model is fitted to its problem rather than universally superior, and only an
external benchmark can tell those two apart.

*(Fairness note: the first run of this benchmark gave every linear model its natural
specification while leaving the GBM on untuned defaults, which is rigged whichever way it
lands. The GBM is now tuned per index and horizon on a slice ending before the first
evaluation year. It improved — and the verdict held.)*

### 6.5 Testing the test: simulated markets with a known answer

Every metric above is scored against *realized* volatility, a noisy estimate of the true
volatility that is never observed. So the pipeline is also run on synthetic markets where σₜ
is known exactly.

| measurement | value | what it means |
|---|---:|---|
| perfect forecaster vs the noisy proxy | **0.887** | the ceiling; read 0.421 against this, not 1.0 |
| our model vs the **true** volatility | **0.593** | roughly half the apparent error is noise in the yardstick |
| our model vs the proxy | 0.455 | captures about 51% of what was attainable |
| market with **no** forecastable volatility | **−0.187** | the harness correctly finds nothing |
| block-shuffled target | **−1.316** | worst fold −0.20; no leakage |

**A finding that changed how this file reports.** In the market built to have *no*
forecastable volatility, R² correctly reads −0.19 — but *skill versus persistence* still reads
**+0.43**. Both are correct: with constant volatility the persistence baseline is nothing but
measurement noise from a short window, so predicting the unconditional mean beats it without
forecasting anything. The consequence: **skill versus persistence cannot establish
predictability on its own.** R² is what establishes signal. The real results clear both bars,
but the two must be read together.

### 6.6 Would acting on it have paid?

Volatility targeting at a 15% risk budget, on the real out-of-sample series:

| strategy | ann. return | ann. vol | Sharpe | max drawdown |
|---|---:|---:|---:|---:|
| buy & hold | 14.9% | 16.6% | 0.90 | −33.7% |
| sized on current vol | 14.6% | 17.1% | 0.85 | −22.0% |
| **sized on forecast** | 15.3% | 16.9% | **0.91** | −26.5% |

The forecast arm beats the persistence arm by **+0.046** Sharpe — but block-bootstrapping the
paired daily returns (2,000 resamples, 21-day blocks) gives a 95% interval of **[−0.067,
+0.156]**. It spans zero. The model forecasts the volatility *level* far more accurately than
the baseline, and that accuracy did **not** convert into a reliable position-sizing edge.

Across 60 simulated markets, including an arm no backtest can run:

| sizing rule | mean Sharpe | gap vs measurement | wins |
|---|---:|---:|---:|
| buy & hold | 0.340 | −0.204 | 28% |
| sized on current vol | 0.544 | 0.000 | — |
| **sized on forecast** | 0.581 | **+0.038** | 67% |
| sized on **perfect foresight** | **1.513** | **+0.969** | **100%** |

The simulated gap (+0.038) matches the real backtest's (+0.046) almost exactly, so the
inconclusive result was *the expected outcome given the model's accuracy*, not bad luck. And
the oracle arm shows **the ceiling is the forecast, not the strategy**: volatility targeting
pays handsomely when the input is sharp enough. Ours captures a few percent of that.

### 6.7 What did not work

These are results, not caveats.

**No feature family earned its place.** Ten families were built and offered to the selection
study: the own-history blocks (`jump`, `semivar`, `range_ext` — HAR-J, HAR-RS, and the
Rogers-Satchell/Yang-Zhang estimators) and the external ones (the full Cboe implied-volatility
complex split into `vix` and `vix_short`; cross-sectional dispersion and correlation across 80
large caps; rates/credit/dollar; sector dispersion; sector-relative position). **Not one
clears the bar on either panel.**

**The VIX splits the two loss functions**, which is the finding a skill-only rule could not
have surfaced:

| panel · folds | t on skill | t on QLIKE |
|---|---:|---:|
| sector · annual | −1.48 | **+2.72** |
| sector · semi-annual | −1.33 | **+2.62** |
| market · annual | −0.21 | **+3.69** |
| market · semi-annual | −0.87 | +0.85 |

Implied volatility improves the forecast under QLIKE and not under squared error, consistently
across panels and slicings. QLIKE punishes under-forecasting risk far harder, so a family that
lifts QLIKE while flattening squared error is fixing the forecast in the upper tail — exactly
what implied volatility should do, and exactly what level accuracy cannot see. It does not
ship because the rule requires both, and the rule was fixed before the numbers were seen. But
"the VIX adds nothing" was too strong: **it adds nothing to level accuracy and something real
to tail-aware loss.**

**Nothing among 22 methods clears the bar either.** Pooled training, residual correction,
shrink-to-persistence, recency weighting, forecast combination, rolling-window refits, a
two-layer neural network, a full-ladder ensemble. The Model Confidence Set at α = 0.10:

| panel | in the set | ruled out |
|---|---:|---|
| market | 15 of 22 | every rolling-refit variant, plus HARQ and SHAR |
| sector | 12 of 22 | every rolling-refit variant, plus HARQ, SHAR and shrink-to-persistence |

The eliminations are coherent: on both panels the Audrino–Chassot rolling-window scheme is not
merely worse here, it is **statistically excluded**.

**The jump-feature result is the cautionary one, and it caught a bug in the harness.** On
annual folds it looks decisive — t = +3.26 on skill, +2.91 on QLIKE, 5/6 fold wins. Re-cut
into semi-annual folds over the same window it collapses to +0.29. The bug was that the
bake-off had only ever run annual folds, making it the one place in the project not applying
the rule the rest advertises, so it printed `ADOPT jumps` for a method already rejected twice
for exactly this reason. It now runs both and adopts only on the intersection.

**It is not a crash predictor.** 4 of 15 stress episodes flagged (27%), median lead 42 trading
days, false-alarm rate 31%. There is a structural cause on top of exogeneity: a squared-error
model shrinks predictions toward the mean, so the forecast rarely reaches the stressed band
until a spike is already forming. Rather than re-picking a flattering threshold, the full
tradeoff curve is published:

| alarm at | detected | median lead | days on alarm | false alarms |
|---:|---:|---:|---:|---:|
| 70th pct | 87% | 39d | 30% | 60% |
| 75th pct | 80% | 41d | 25% | 56% |
| 85th pct | 47% | 51d | 15% | 42% |
| 90th pct | 33% | 60d | 10% | 33% |

**A dedicated alarm head was built to attack exactly that — promising, not established.** The
mean forecast was left alone and a separate head added with its own binary target and its own
metric (MCC on "will the next 10 days be stressed?", never R²). On sectors:

| rule | MCC | stress recall | precision | days on alarm |
|---|---:|---:|---:|---:|
| mean-threshold *(shipped)* | 0.201 | 0.147 | 0.398 | 3.2% |
| **quantile-85** | **0.294** | **0.534** | 0.261 | 16.9% |

3.6× the stress detection with a better balanced score — and it does not ship. The paired delta
is stable at about +0.09 MCC while the t-statistic climbs with fold count (+1.29 annual, +1.75
semi-annual, +2.04 quarterly): the signature of a real but small effect gaining power, and
exactly the pattern the multi-granularity rule exists to refuse. The market panel also
contradicts it (quarterly t = −2.66). Recorded as *not established*, which is a different
claim from *rejected*.

**The 21-day horizon is not shipped.** In walk-forward it beat persistence in only 12 of 15
years on the market panel, with a −1.29 miss in an ultra-calm year. The sector panel at 21 days
is stable, but the dashboard presents both together and the market half is the half that fails.

**Buying intraday data is not justified for this product.** As an *input* it is worth +0.061
skill (t = +1.68, ahead in 8 of 12 folds) — the largest effect any feature addition has
produced here, and still short of the bar. As a *target* it made things **worse** (R² 0.191 →
0.132), which was the opposite of the prior. The reason is quantitative: our target is a
10-day *average* of daily variances, and averaging ten noisy estimates already removes most of
the measurement noise. Intraday precision pays at a 1-day horizon, which is where the
literature reports it.

**The ablation table is descriptive, not decisive.** `cli ablation` reports each family's mean
walk-forward R², and that metric is too noisy to decide anything: the largest delta in the
table is 0.06 against a fold-to-fold spread of ~0.35, because each year's R² is measured
against that year's own variance. It agrees directionally with the selection study, and that
is all it is good for.

### 6.8 What the literature said, and what actually transferred

Three well-cited results were implemented and tested rather than cited and assumed.

**SHAR replicates.** Splitting realized variance into up-day and down-day components beat
plain HAR on both panels (market t = +2.55 with 6/6 fold wins). It also replicates externally,
beating Log-HAR on 15–17 of 21 indices. But adding the block to the *shipped* model changed
nothing — the GBM already carries a downside-vol term and range estimators. A replication and
a redundancy at once.

**HARQ does not transfer to daily bars, for a known reason.** It corrects attenuation bias from
measurement error in RV estimated from *intraday* returns, using realized quarticity to gauge
that error. On daily bars realized quarticity is barely estimable and the correction has
nothing to work with. On the external benchmark, where genuine intraday data is available, it
behaves as published. A non-replication with a cause, not a contradiction.

**The rolling-window result did not transfer.** Audrino & Chassot found HAR re-estimated in a
2.5–4 year rolling window exceptionally hard to beat. Tested at 2, 3 and 4-year windows with
monthly refits, every variant was worse (t = −2.31 to −2.45), and the MCS excludes them all.
Their setup forecasts one day ahead for single stocks; ours forecasts a ten-day average for an
index.

**The neural network does not replicate here.** A two-layer MLP is the best single model in
Christensen–Siggaard–Veliyev, and neural nets dominate on pooled cross-sections in
Zhang–Cucuringu et al. Here: market t = −0.91 on skill, **−3.07 on QLIKE**. The first attempt
to measure it was invalid — every market fit hit its 400-iteration cap with a convergence
warning, so a truncated network was scored against fully-fitted rivals, in the direction that
flattered the incumbent. Raised to 2,000 iterations with early stopping, the verdict survived
and the QLIKE gap narrowed.

**Uncertainty bands, and why they are adaptive.** Split conformal gives coverage guarantees
without distributional assumptions — but only under *exchangeability*, which financial time
series plainly violate. The violation was measurable: the static 80% band covered just **73%**
of outcomes. Adaptive conformal inference (Gibbs & Candès) steers width from realized misses,
restoring coverage to **78%**.

---

## Also here

**News semantics overlay** (`volradar/news/`, the *News radar* tab). Headlines from 24 free
RSS feeds — CNBC, MarketWatch, the FT, BBC, Nasdaq, Seeking Alpha, Investing.com, the Federal
Reserve, Google News and per-sector Yahoo Finance — are de-duplicated across channels, scored
Bearish / Bullish / Neutral by the DistilBERT classifier in `volradar/sentiment/` (88%
accuracy, 0.85 macro-F1), and tagged with sectors, risk themes and shock language. They are
aggregated into a recency- and breadth-weighted 0–100 *news pressure* per sector, shown beside
the forecast with a flag where the two rank a sector very differently.

It is **not a model input**, for the same reason as before: free feeds carry days of history,
not years, so news tone cannot be walk-forward validated over 2012–2026, and back-filling from
today's articles would leak the future. What changed is that the blocker is now being worked
off: every run appends each story, with its score frozen at first sight, to
`data/news/headlines.parquet`. `validation_status()` reports progress, and once the archive
spans `NEWS_MIN_DAYS_FOR_TEST` days it runs a first screen against the out-of-sample forecast
error. Adoption would still go through `metrics.paired_test` like every other family.

The prior research phase — stock *direction* prediction (near-random, MCC ≤ 0.09) and a
single-stock Tesla deep dive — is preserved at tag `v1-research-archive`.

---

Educational research project. **Not investment advice.** Forecasts are of volatility (how much
prices move), never of direction (which way).
