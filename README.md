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
| **R² (log realized vol)** | **0.421** | **0.479** |
| persistence baseline R² | 0.111 | 0.208 |
| **error removed vs baseline** | **35%** | **34%** |
| rise/fall accuracy | **70.5%** (base 53.5%) | **70.4%** (base 51.0%) |
| rise/fall MCC | 0.409 | 0.408 |
| regime accuracy (3-class) | 64.7% (base 47.8%) | 66.4% (base 51.2%) |

At the 5-day horizon: market R² **0.446**, MCC **0.464**; sectors R² **0.427**, MCC **0.450**.

**Stability — the number that matters most.** Skill versus the persistence baseline, scored
year by year so both sides face the same window:

| panel · horizon | mean ± std | worst year | years positive |
|---|:---:|:---:|:---:|
| market · 5d | +0.403 ± 0.123 | +0.184 | **15 / 15** |
| market · 10d | +0.354 ± 0.186 | +0.009 | **15 / 15** |
| sector · 5d | +0.408 ± 0.058 | +0.231 | **15 / 15** |
| sector · 10d | +0.351 ± 0.107 | +0.055 | **15 / 15** |

Every sector individually beats persistence — skill 0.29–0.39, direction accuracy 68–72%
across all eleven. No sector is carried by the others.

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
python -m volradar.cli events     # did it flag real volatility spikes?
python -m volradar.cli decision   # would acting on it have helped?
python -m volradar.cli train      # production fit + dashboard snapshot
python -m volradar.cli all        # all of the above
python -m volradar.cli serve      # API + dashboard
```

Every stage caches and is idempotent; `--force` rebuilds one from scratch. To refresh the
dashboard to the latest close without re-running the backtest:
`fetch --force && build --force && train`.

**Tests:** `pytest -q` — 27 tests covering causality, embargo, split boundaries, metric
correctness against hand-computed values, and the API contract.

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
