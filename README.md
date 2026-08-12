# Volatility Radar

Forecasts how much the **market** and each of the **eleven S&P sectors** will move over the
next 5–10 trading days, and ships the track record that qualifies every number.

Not direction. Not returns. **Volatility** — because short-horizon direction is very close
to unpredictable, while volatility is strongly persistent and genuinely forecastable.

```bash
pip install -r requirements.txt
python -m volradar.cli all        # fetch, build, backtest, train  (~25 min, CPU only)
cd frontend && npm install && npm run build
python -m volradar.cli serve      # dashboard at http://127.0.0.1:8000
```

---

## Headline results

Pooled out-of-sample, **2012-01-03 → 2026-07-28**. A model is retrained at the start of each
year on data available up to that point, then scored on the year that follows. No model was
ever shown the data it is graded on.

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
Every forecast ships with a **calibrated uncertainty band**, and the coverage row above is
what that band actually delivered out-of-sample — not what it nominally promised.

**Stability — the number that matters most.** Skill versus the persistence baseline, scored
year by year so both sides face the same window:

| panel · horizon | mean ± std | worst year | years positive |
|---|:---:|:---:|:---:|
| market · 5d | +0.403 ± 0.123 | +0.184 | **15 / 15** |
| market · 10d | +0.354 ± 0.186 | +0.009 | **15 / 15** |
| sector · 5d | +0.407 ± 0.077 | +0.160 | **15 / 15** |
| sector · 10d | +0.358 ± 0.125 | −0.008 | 14 / 15 |

Every sector individually beats persistence — skill 0.29–0.41, direction accuracy 69–73%
across all eleven. No sector is carried by the others.

One honest note on that table: pooled training raised the sector 10-day mean (0.351 → 0.358)
and improved nine of eleven sectors, but one walk-forward year slipped to −0.008 where the
previous model had been positive in all fifteen. That is a rounding error against a baseline,
not a regime the model fails in — but the row says 14/15 rather than 15/15, so it is written
as 14/15.

---

## Testing the test: simulated markets with a known answer

Every metric above is scored against *realized* volatility — a noisy estimate of the true
volatility, which is never observed. So the pipeline is also run on synthetic markets from a
written-down stochastic-volatility process (persistence, leverage, jumps), where σₜ is known
exactly. Three things fall out that real data cannot give:

**1. The accuracy ceiling.** A forecaster with *perfect* knowledge of future volatility
scores **R² = 0.887** against the noisy realized proxy — not 1.0. That is the number our
real-data R² should be read against. Scored against the true target instead of the proxy,
the same model goes from R² 0.455 to **0.593**: roughly half the apparent error is noise in
the yardstick, not failure of the model. It captures about **51%** of what was attainable.

**2. Our primary metric can lie on its own — and that changed how this README reports.**
In a market built to have *no* forecastable volatility, R² correctly reads **−0.19**, but
*skill versus persistence* still reads **+0.43**. Both are correct: with constant
volatility, the persistence baseline is nothing but measurement noise from a short window,
so predicting the unconditional mean beats it without forecasting anything. The consequence:
**skill versus persistence cannot establish predictability on its own** — it measures "did we
beat a noisy estimator", a lower bar than "is there signal here". R² is what establishes
signal. The real results clear both bars (market R² 0.421 *and* skill 0.349), so the
conclusions stand — but the two must be read together, and that was not obvious before
simulating a world where one of them misleads.

**3. A real leakage test.** With the target block-shuffled to destroy any link to the
features, the harness reports R² **−1.32** (worst fold −0.20). It finds nothing when there is
nothing, which is what makes the positive results worth reading.

### Monte Carlo: would acting on it have paid?

The real backtest is one sample. Running the same volatility-targeting strategy across 60
simulated markets from the same process — including an arm no backtest can run, sizing on
**perfect foresight**:

| sizing rule | mean Sharpe | gap vs measurement | 95% range | wins |
|---|---:|---:|---:|---:|
| buy & hold | 0.340 | −0.204 | — | 28% |
| sized on current vol | 0.544 | 0.000 | — | — |
| **sized on forecast** | 0.581 | **+0.038** | [−0.32, +0.29] | 67% |
| sized on **perfect foresight** | **1.513** | **+0.969** | [+0.65, +1.31] | **100%** |

This resolves the open question from the real-data decision study. The forecast arm's
simulated gap (+0.038, spans zero) matches the real backtest's (+0.046, spans zero) almost
exactly — so that inconclusive result was *the expected outcome given the model's accuracy*,
not bad luck. And the oracle arm shows **the ceiling is the forecast, not the strategy**:
volatility targeting pays handsomely (+0.97 Sharpe, wins 100%) when the input is sharp
enough. Ours captures only a few percent of that. That is a headroom result — the economic
payoff exists, and it is gated on forecast sharpness, not on the trading rule.

---

## The three open questions, now settled

All three were left explicitly open above. Each was decided under the rules already in
force: pre-test folds only, paired against the incumbent, checked at two fold granularities.

**1. Ridge vs GBM on the market panel — the GBM stays.** Ridge had beaten it in the fixed
split and across the full walk-forward (t=+1.45), which looked like a problem. Re-tested on
*pre-test* folds — the window selection is actually allowed to use — ridge is t=+0.68
(annual) and t=+0.56 (semi-annual) on skill, and **worse** on QLIKE (t=−0.84, −2.23). The
earlier signal came from measuring on the reporting window, which is the wrong window for
choosing a model. Question closed, incumbent retained.

**2. Training on QLIKE or a high quantile — neither adopted.** The motivation was real:
squared error is symmetric, and the forecast's known failure modes (late alarms, flat
exposure signal) are asymmetric. The first QLIKE implementation *diverged* to a loss of
5×10⁸ — a genuine bug, since the Hessian decays to zero as the forecast rises above the
truth and boosting's step is gradient/Hessian. With the Hessian floored and the base score
started at the target mean, it trains properly and delivers a QLIKE gain of 0.003
(t=+0.13) while costing 0.10 of skill. A 60th-percentile fit does clear the QLIKE bar on the
sector panel (t=+2.04 and +2.22, both granularities) — but tested on the weakness it was
built to fix, it detects **exactly the same 27%** of stress episodes, buying its longer lead
purely with more alarm days (11.4% vs 9.8%) and more false alarms. That trade is already
available by moving the alarm threshold on the existing model. **Mean-shrinkage is not the
binding constraint on event detection; the exogeneity of shocks is.**

**3. Intraday data — helps as an input, does not help as a target.** See below.

## Does intraday data pay? A scoped study on 2000–2016 S&P 500

Run on the **Oxford-Man Realized Library**: 5-minute and 10-minute realized variance,
bipower variation, realized semivariance, realized kernels and median-truncated RV for 21
indices, computed from tick data. One 16MB file, cached once, no API key and no rate limits.
The Institute shut the library down and its domain no longer resolves, so this reads an
archived copy from a public GitHub repository — a 2016 vintage. That is enough to answer the
question but not to serve live forecasts, so it feeds a scoped study with its own
walk-forward and never the production pipeline.

| | R² | skill | direction |
|---|---:|---:|---:|
| daily features only | 0.123 | +0.169 | — |
| **+ intraday features** | **0.191** | **+0.230** | — |

Paired delta **+0.061, t=+1.68, ahead in 8 of 12 folds**. That is the largest effect any
feature addition has produced in this project — larger than every external factor family
combined — but it still does not clear the t>2 bar, and the bar does not move.

**As a target, precision made things worse, which was the opposite of the prior:**

| target | R² | direction |
|---|---:|---:|
| daily squared returns (close-to-close) | 0.191 | 0.717 |
| 5-min RV + overnight (close-to-close) | 0.132 | 0.663 |
| 5-min RV alone (open-to-close) | 0.139 | 0.658 |

The comparison is like-for-like: 5-minute RV is an *open-to-close* measure and cannot see
overnight gaps, so the overnight return was reconstructed from the open and prior close and
added back before comparing.

The explanation is quantitative and the simulation confirms it independently. Our target is
a **10-day average** of daily variances, and averaging ten noisy estimates already removes
most of the measurement noise — the simulated perfect forecaster scores 0.887 against it, so
it is ~89% clean before any intraday data is involved. Intraday precision pays at a **1-day**
horizon, which is where the literature reports it. At ten days, temporal averaging has
already done that job.

**Practical conclusion: buying intraday data is not justified for this product.** It would
help modestly as an input and not at all as a target, and the headline horizon is the wrong
one for it to matter. That is a €0 answer to the project's most expensive open question.

---

## Methodology: what was tried, what survived

Eight modelling methods were tested against the incumbent on identical pre-test folds. The
comparison is **paired** — fold by fold, not average against average — because judging two
separate means lets an *unstable* incumbent inflate its own replacement bar, which is
exactly backwards.

| method | idea | verdict |
|---|---|---|
| **pooled training** | fit on all 91 series (79 stocks + 11 sectors + SPY) instead of one panel | **adopted (sectors)** — t=+2.15, wins 9/12 folds |
| jump features | bipower variation / HAR-CJ continuous-vs-jump split | rejected — *see below* |
| pooled + residual correction | pooled fit plus a panel-specific correction | no gain over plain pooling |
| shrink to persistence | optimal blend with the baseline | worse (t=−1.35) |
| recency weighting | 3-year half-life on training samples | no gain |
| HAR + GBM average | equal-weight forecast combination | worse at 10d (t=−2.63 on sectors) |
| **adaptive conformal** | online-corrected prediction intervals | **adopted (both panels)** |
| **QLIKE** | volatility-native loss, robust to a noisy proxy | **adopted as a reported metric** |
| HARQ | realized-quarticity correction (Bollerslev/Patton/Quaedvlieg 2016) | no gain here — *see below* |
| SHAR | signed semivariance (Patton/Sheppard 2015) | replicates vs HAR, redundant in ours |
| rolling window + frequent refit | Audrino/Chassot 2025, "HARd to Beat" | did not transfer |

### What the literature said, and what actually transferred

Three well-cited results were implemented and tested properly rather than cited and assumed.

**SHAR replicates.** Splitting realized variance into up-day and down-day components beat
plain HAR on both panels — market **t=+2.55 with 6/6 fold wins**, sector t=+2.30. The
published effect is real in our data. But adding the same block to the *shipped* model
changed nothing (t=+0.36 at 6 folds, −0.62 at 12): the GBM already carries a downside-vol
term and range-based estimators, so the information was there. A replication and a
redundancy at once.

**HARQ does not, and the reason is structural.** It was worth t=+0.48 on the market panel and
−0.45 on sectors. HARQ corrects attenuation bias caused by measurement error in RV estimated
from *intraday* returns, using realized quarticity to gauge that error. We have daily bars,
so realized quarticity is barely estimable and the correction has nothing to work with. This
is a non-replication with a known cause, not a contradiction of the paper.

**The rolling-window result did not transfer.** Audrino & Chassot found across 1,445 stocks
that HAR re-estimated in a 2.5–4 year rolling window is exceptionally hard to beat, and that
the fitting scheme matters more than the model class. Our project used the scheme they call
overlooked — expanding window, annual re-estimation — so this looked like the single most
promising fix. Tested at 2, 3 and 4-year windows with monthly refits, every variant was
*worse* (t = −2.31 to −2.45), and holding the model and features fixed while changing only
the window (GBM rolling vs expanding) also lost (t=−1.17). Two plausible reasons: their
comparison is on RV and VIX inputs, while our edge comes from range-based estimators that
recover intraday information from daily bars; and they forecast one day ahead for single
stocks, where we forecast a ten-day average for an index. Different problem, different answer.

The lasting benefit is to the *benchmarks*: HARQ and SHAR now sit in the model ladder, so
every result is quoted against a stronger bar than plain HAR. On the fixed out-of-sample
split at 10 days:

| model | market R² | sector R² |
|---|---:|---:|
| persistence | 0.149 | 0.216 |
| HAR | 0.305 | 0.373 |
| HARQ | 0.303 | 0.373 |
| SHAR | 0.333 | 0.385 |
| ridge | **0.428** | 0.457 |
| gbm *(shipped)* | 0.412 | **0.463** |

On this single fixed window plain ridge outscores the shipped GBM on the market panel
(0.428 vs 0.412). That looked like a problem and was investigated properly — see **The three
open questions, now settled** above. On the pre-test folds that selection is actually
permitted to use, the gap is t=+0.68 and ridge is worse on QLIKE, so the GBM stays.

**Pooled training** was the one substantive win, and it fixed a real problem: the sector
model had 37k rows and now trains on 471k. Volatility dynamics are broadly shared across
equities and every feature is scale-free, so the cross-section supplies data one series
cannot. Nine of eleven sectors improved individually. It did **not** help the market panel
(t=−1.11) — plausibly because an index is genuinely a different object from its
constituents, and 79 single stocks drown out the one series that matters.

**The jump-feature result is the cautionary one.** On annual folds it looked decisive
(t=+3.26, 5/6 wins). Re-run on semi-annual folds over the same window it collapsed to
t=+0.29. A result that depends on how you slice the calendar is not a result, so it was
rejected — despite being the most theoretically motivated candidate in the list.

**Uncertainty bands, and why they are adaptive.** Split conformal prediction gives interval
coverage guarantees without distributional assumptions — but only under *exchangeability*,
which financial time series plainly violate. The violation was measurable: the static 80%
band covered just **73%** of outcomes on the market panel. Adaptive conformal inference
(Gibbs & Candès) steers the interval width from realized misses, restoring coverage to
**78%**. Bands widen after a run of misses and tighten once forecasts land again. The
dashboard shows the measured coverage next to the band rather than the nominal figure.

---

## What did not work

These are results, not caveats, and they are as load-bearing as the table above.

**No external factor family earned its place.** The VIX complex (level, term structure,
vol-of-vol, variance risk premium), cross-sectional dispersion and average correlation
across 80 large caps, rates/credit/dollar, and sector dispersion were all built and tested.
Every combination landed **within one standard error** of using the series' own price
history alone, across six pre-test selection folds — so the one-standard-error rule keeps
the simplest model. Realized volatility turns out to be a remarkably complete summary of a
market's own risk state, and the signals that look independent are largely reflections of
it. All four families remain implemented and are re-tested on every run.

The two ends of that spectrum are worth naming. **The VIX was the closest call**: in the
leave-one-out ablation it was worth **+0.075** R² on the market panel, easily the best of
the four — it simply did not clear the one-standard-error bar on the selection metric, and
that bar was set before the results were seen. **Macro was the clearest failure**, costing
0.05–0.07 R² wherever it appeared; its levels are non-stationary, so the model learns
regime-specific values that do not carry forward.

**The 21-day horizon is not shipped.** It was built and measured, but in walk-forward it beat
persistence in only **12 of 15** years for the market panel, with a −1.29 miss in an
ultra-calm year. Every model tested — including plain HAR — extrapolates a volatility floor
that very quiet years fall through. The measurement is kept in the reports; the number is not
served.

**It is not a crash predictor.** At the pre-registered alarm threshold it warned ahead of only
**4 of 15** stress episodes (27%), median lead 42 trading days, false-alarm rate 31%. COVID-19
and the 2024 yen carry unwind were exogenous — nothing in market data foreshadowed them. There
is also a structural cause: a squared-error model shrinks predictions toward the mean, so the
forecast rarely reaches the stressed band until a spike is already forming. Rather than
re-picking a flattering threshold, the **full tradeoff curve** is published:

| alarm at | detected | median lead | days on alarm | false alarms |
|---:|---:|---:|---:|---:|
| 70th pct | 87% | 39d | 30% | 60% |
| 75th pct | 80% | 41d | 25% | 56% |
| 85th pct | 47% | 51d | 15% | 42% |
| 90th pct | 33% | 60d | 10% | 33% |

**The trading benefit is not statistically established.** Volatility targeting at a 15% risk
level, 1bp costs, out-of-sample:

| sizing rule | return | vol | Sharpe | max drawdown |
|---|---:|---:|---:|---:|
| buy & hold | 14.9% | 16.6% | 0.90 | −33.7% |
| sized on current vol | 14.6% | 17.1% | 0.85 | **−22.0%** |
| **sized on forecast** | 15.3% | 16.8% | **0.91** | −26.5% |

The forecast arm beats the persistence arm by **+0.046** Sharpe — but block-bootstrapping the
paired daily returns (2,000 resamples, 21-day blocks to preserve volatility clustering) gives
a 95% interval of **[−0.067, +0.156]**. It spans zero. So: the model forecasts the volatility
*level* far more accurately than the baseline, and that accuracy did **not** convert into a
reliable position-sizing edge on this evidence. Both facts are true and both are reported.

---

## What it's for

Read it as a **gauge of current and near-term risk conditions**, which it measures well:

- *Is the market about to get choppier?* — 70% accurate, out-of-sample, 15 years running.
- *Which sectors are structurally calm right now?* — the ranking recovers the classic
  defensive/cyclical ordering from data alone, with per-sector accuracy attached to each row.
- *How much movement should I plan for?* — an annualized volatility number with frozen
  calm/normal/stressed bands.

Do **not** read it as a crash alarm or a trading signal. The evidence above says it is not
reliably either.

---

## How it stays honest

- **Chronological only** — train on the past, test on the future; days are never shuffled.
- **No lookahead** — enforced behaviourally: a test perturbs a *future* bar and asserts no
  *past* feature moves; a mirror test asserts the target *does* move, so a frozen target
  can't masquerade as clean hygiene.
- **Boundary embargo** — a 10-day label spans 10 days, so rows whose outcome window crosses a
  split boundary are purged rather than leaked.
- **Walk-forward** — 15 independent yearly out-of-sample windows, reported mean ± spread.
- **Baselines everywhere** — persistence for levels, majority for classes. No score is
  reported without the thing it must beat.
- **Frozen regime bands** — calm/normal/stressed cut points come from training data only.
- **Selection never sees the reporting window** — hyperparameters tuned on a validation slice
  inside the training period; factor families chosen on folds strictly before the test start.
- **Coverage guard** — a provider once silently returned 13 rows instead of 5,400 for a macro
  series, nulling two features without raising. The fetcher now checks every series against
  its peers, retries short ones, and warns loudly.

Full definitions: [`docs/METRICS.md`](docs/METRICS.md). Layout and design rationale:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Dashboard

Three tabs. **Dashboard** — market forecast with its regime meter, the exact per-feature
contributions driving it (in annualized-vol points, not unitless scores), the sector
stability ranking, and forecast-vs-realized history. **Track record** — everything on this
page, including the failures. **Method** — what is being predicted and what is not.

The API reads generated artifacts and computes nothing, so the dashboard physically cannot
display a figure that disagrees with the backtest. A missing report returns a 503 telling you
which command to run, never an improvised default.

---

## Pipeline

```
python -m volradar.cli fetch      # download raw series (network)
python -m volradar.cli build      # assemble market + sector panels
python -m volradar.cli evaluate   # fixed split, pooled OOS, walk-forward, per-sector
python -m volradar.cli ablation   # what each factor family is worth
python -m volradar.cli methods    # methodology bake-off (paired, pre-test folds)
python -m volradar.cli select     # which factor families ship
python -m volradar.cli events     # did it flag real volatility spikes?
python -m volradar.cli decision   # would acting on it have helped?
python -m volradar.cli simulate   # validate the method on known-truth synthetic markets
python -m volradar.cli settle     # decide open model questions on pre-test folds
python -m volradar.cli intraday   # is intraday data worth paying for?
python -m volradar.cli train      # production fit + dashboard snapshot
python -m volradar.cli all        # all of the above
python -m volradar.cli serve      # API + dashboard
```

Every stage caches and is idempotent; `--force` rebuilds one from scratch. To refresh the
dashboard to the latest close without re-running the backtest:
`fetch --force && build --force && train`.

**Tests:** `pytest -q` — 51 tests covering causality, embargo, split boundaries, metric
correctness against hand-computed values, conformal-interval causality (its width is derived
from realized errors, so it is one edit from being the subtlest lookahead bug here), the
synthetic-market generator, and the API contract.

**Data:** Yahoo Finance via `yfinance` — free, no API key. Prices 2005 → today.

---

## Also here

`volradar/sentiment/` holds a DistilBERT financial-sentiment classifier (88% accuracy,
0.85 macro-F1) fine-tuned in an earlier phase of this project. It is **not** a model input,
and the reason is documented in `headlines.py`: free headline sources provide days of
history, not years, so news tone cannot be walk-forward validated over a 2012–2026 window.
Back-filling it from today's articles would leak the future into the past. It ships as an
optional descriptive overlay, clearly labelled unvalidated.

The prior research phase — stock *direction* prediction (near-random, MCC ≤ 0.09) and a
single-stock Tesla deep dive — is preserved at tag `v1-research-archive`.

---

Educational research project. **Not investment advice.** Forecasts are of volatility, never
of direction.
