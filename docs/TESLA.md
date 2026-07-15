# Tesla (TSLA) single-stock, intraday-enriched deep dive

A focused companion to the multi-stock study. Same honest methodology
(chronological splits, overlapping-window embargo, baseline-vs-MCC for direction
and baseline-vs-R² for volatility), applied to **one** deliberately volatile,
retail/news-driven name, and enriched with **genuine intraday data** the daily
multi-stock project couldn't afford.

> **The one question:** does *deeper, single-stock, intraday-enriched* data buy a
> **stronger or more reliable** edge than *broad, shallow, multi-stock* data?
> **Answer, up front and honestly: no — not on the headline metrics.** Deeper on
> one stock lost more (the cross-sectional signal) than intraday granularity added.

---

## Why Tesla, and why this is a fair test

TSLA was chosen *before* seeing any numbers, for reasons that don't depend on the
result: it is large, liquid, genuinely volatile, and has a sustained (not one-off)
retail-and-news-driven narrative — Elon Musk's tweets are the canonical driver, so
it's the natural place to test whether single-stock + text + intraday depth helps.
It is **not** cherry-picked for good-looking metrics; the honest expectations were
written down in advance (single stock strips the cross-sectional signal and leaves
far fewer rows, so direction should be near-random and volatility R² should fall).

## Data (no API key)

The original plan named Alpha Vantage's free tier; with no key available we used
**yfinance** instead (already a dependency, no key), which changes nothing about
the method. Sources, all free:

| source | what | span | role |
|---|---|---|---|
| yfinance daily | TSLA OHLCV | 2015–2026 (2.8k days) | technical + HAR + range features, close-to-close RV target |
| yfinance **1-hour** | intraday bars (~7/session) | ~2015→2026 hourly retained **2023-08 → 2026-07** (729 sessions) | **intraday realized-vol** target + features |
| yfinance 5-min / 1-min | recent bars | 60 / 7 days | *validate* that hourly RV tracks finer RV |
| Elon Musk tweets (public, GitHub-mirrored Kaggle set) | 9,044 tweets | 2015–2020 | DistilBERT sentiment factor + the public baseline we sanity-check against |

**Honest limitation — no free 2-year 1-minute history.** Yahoo caps 1-min to ~7
days and hourly to ~2 years, so the "intraday" layer is **hourly** realized vol.
It is a real RV estimator (many within-day returns vs one daily squared return),
but coarse: over 60 overlapping sessions, hourly RV correlates **+0.47** with
5-min RV and under-measures it by ~26%. The HAR aggregates smooth most of that
daily noise. A paid 1-minute feed would mainly reduce this bias, not change the
conclusions below.

**Honest limitation — era mismatch.** The free intraday window (2023–2026) does
not overlap the free tweet window (2015–2020). So sentiment enriches the *daily*
models over 2015–2020, and intraday enriches the *volatility* model over 2023–2026;
they are reported separately rather than pretended to be one enriched dataset.

**Single-stock structural limitation.** There is **no cross-sectional feature**
here (the universe run's per-day vol rank / relative-vol, its single strongest
extra signal, is undefined for one ticker). Losing it is part of what we measure.

---

## Results

### Direction — no reliable edge (as expected, and noisier than the 88-stock run)

Walk-forward MCC (mean ± std across 10 yearly folds, 2017–2026). MCC ≈ 0 and the
std swamps the mean at **every** horizon and feature set:

| horizon | best TSLA MCC | StockNet-88 MCC | verdict |
|--------:|:-------------:|:---------------:|---------|
| 5 day  | +0.00 ± 0.11 | +0.068 | ❌ no edge (88-stock had a small real one) |
| 10 day | +0.02 ± 0.13 | +0.088 | ❌ indistinguishable from 0 |
| 20 day | +0.06 ± 0.22 | +0.079 | ❌ huge CI, no edge |

Adding sentiment doesn't help. Accuracy sits *below* the majority baseline
(the balanced classifier trades accuracy to predict both classes; MCC is the fair
metric, and it's ~0). One stock has too few, too noisy samples to beat chance.

### Volatility magnitude (log-RV R²) — real but ~half the universe's

Fixed OOS split (train < 2022, test 2022–2026), XGBoost right-sized for ~1,700
single-stock rows (500-tree universe settings overfit here — verified):

| horizon | persistence | HAR-OLS | XGBoost | universe-80 XGBoost |
|--------:|:-----------:|:-------:|:-------:|:-------------------:|
| 5 day  | −0.77 | 0.05 | **0.14** | 0.37 |
| 10 day | −0.20 | 0.13 | **0.21** | 0.42 |
| 21 day | −0.19 | 0.17 | **0.17** | 0.41 |

Positive and real, but **roughly half** the universe's R². **Why:** the universe's
0.42 was overwhelmingly **cross-sectional** — most of it is explaining *which* stock
is volatile on a given day, not *when* a given stock will be. For one stock that
part is gone; only the harder time-variation remains, and a single ticker's log-vol
is a nearly-constant-mean, noisy series (current-vs-forward daily-RV correlation is
just 0.31). Persistence even goes **negative** out-of-sample because TSLA's vol
level shifted structurally after 2021 — a regime shift that averages out across 80
stocks but not across one. The yearly walk-forward confirms the fragility: mean R²
is negative with a wide spread (some years the single-stock model simply fails).

### Volatility direction ("will vol rise?") — this part *does* transfer

| metric | TSLA (h=5) | universe-80 (h=5) |
|---|:---:|:---:|
| accuracy | 0.727 | 0.72 |
| MCC | +0.466 | +0.45 |

Volatility clustering is a **per-stock** property, so the *classification* edge
survives on one name even though the *magnitude* R² does not. A clean example of
why choosing the right target matters more than data depth.

### The intraday lever — modest, and not from where you'd expect

On the hourly window (chronological 70/30 split, ~200 test sessions), we crossed
the **target** (close-to-close `cc` vs intraday `iv` realized vol) with the
**feature set** (daily vs daily+intraday). Two honest findings:

- The **intraday-measured target is more forecastable** than the noisy
  close-to-close target, especially at longer horizons (h=21 XGBoost R²: iv −0.51
  vs cc −2.74). A cleaner *measurement* of the thing you predict helps.
- Adding intraday *features* to the daily HAR set **does not help** (often slightly
  worse) — hourly RV is too coarse (r ≈ 0.47 vs 5-min) and ~200 test rows too few
  to exploit. All absolute R² on this thin, single-regime split are negative; the
  experiment is **underpowered**, and we say so rather than over-reading it.

### vs public Tesla-prediction projects

Public notebooks routinely report **R² > 0.95 / "90%+ accuracy."** Those are
artifacts, not skill: regressing tomorrow's **price level** on today's (a near
random walk → R² ≈ 1, meaningless for returns), **shuffling time** so the test set
leaks into training, or feeding **same-day** High/Low to "predict" the same day's
Close. Measured honestly — walk-forward, embargoed, no leakage — public TSLA
direction studies land where we and the efficient-market literature do: **~50–55%**
next-day accuracy. Our *lower* numbers are the *correct* ones.

---

## Verdict

**Deeper, single-stock, intraday-enriched data did not buy a stronger or more
reliable edge than broad, shallow, multi-stock data.**

- Direction stayed near-random, with *wider* confidence intervals than the 88-stock run.
- Volatility magnitude R² *fell* to ~half — the universe's headline number was
  mostly cross-sectional, which one stock cannot provide.
- Only the "will vol rise?" classification held up (clustering is per-stock).
- Intraday granularity helped the *cleanliness of the target* at long horizons but
  hourly intraday *features* didn't beat daily HAR on this sample size.

This mirrors the parent project's own lesson in reverse: **breadth of data beat
depth of data.** Reported straight — a negative result, kept, exactly as the
original project keeps its negative results (VIX, calibration, the Reddit set).

---

## Reproduce

```bash
python tesla/fetch_prices.py        # yfinance daily + intraday, RV resolution check
python tesla/kaggle_baseline.py     # fetch/loads the public tweets (no-auth mirror)
python tesla/sentiment_features.py  # DistilBERT-score tweets -> daily sentiment
python tesla/features.py            # assemble the feature+label table
python tesla/direction.py           # direction walk-forward (MCC)
python tesla/volatility.py          # volatility R² + intraday lever + classification
python tesla/compare.py             # the three-way honest comparison + verdict
pytest -q tests/test_tesla.py       # leakage / RV construction / label geometry
```

*Educational project — not investment advice. Datasets keep their own licenses.*
