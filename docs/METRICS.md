# Metrics & Concepts — a learning guide

Everything this project measures, in plain language, with the math, why it's used here, what a "good"
value looks like **on this data**, and where you see it in the code. Read top to bottom once; use the
[cheat sheet](#cheat-sheet) after that.

The tasks are both **binary classification** — every row gets a 0 or 1 prediction:
- **Task A (direction):** 1 = price goes up over the next *h* days, 0 = down.
- **Task B (volatility):** 1 = volatility rises vs now, 0 = falls.

So all the classification metrics below apply to both.

---

## Part 0 — the foundation: the confusion matrix

Every classification metric is built from four counts. Compare each prediction to the truth:

|                     | actually **1** (up) | actually **0** (down) |
|---------------------|:-------------------:|:---------------------:|
| **predicted 1**     | TP (true positive)  | FP (false positive)   |
| **predicted 0**     | FN (false negative) | TN (true negative)    |

- **TP** — said up, was up ✅
- **TN** — said down, was down ✅
- **FP** — said up, was down ❌ ("false alarm")
- **FN** — said down, was up ❌ ("miss")

Worked example we'll reuse (5 predictions):

| truth | 1 | 1 | 1 | 0 | 0 |
|-------|---|---|---|---|---|
| pred  | 1 | 1 | 0 | 1 | 0 |

→ TP=2, FN=1, FP=1, TN=1.

---

## Part 1 — the classification metrics

### Accuracy
**What:** the fraction of predictions that were correct.
**Formula:** `(TP + TN) / everything`. Example: `(2+1)/5 = 0.60` → 60%.
**Good value here:** for **direction**, anything meaningfully above ~50% is hard (markets are near-random);
mid-50s% is realistic. For **volatility**, ~70% is achievable.
**The trap — imbalance:** if 90% of days are "up", a model that *always* says "up" scores 90% accuracy
while learning nothing. That's why we never report accuracy alone.
**In code:** `sklearn.metrics.accuracy_score`, in `predictor/evaluate.py::score`.

### Base rate & the majority baseline
**Base rate** = the fraction of rows that are class 1 (e.g. "52% of test days had volatility rise").
**Majority baseline** = the accuracy you'd get by always predicting the more common class = `max(base,
1−base)`. **Any accuracy claim must beat this**, or the model has shown no skill. In Task B we print the
baseline in every table (`predictor/volatility.py`) precisely so a high-but-meaningless accuracy can't
fool us. *This is the single most important habit in this guide.*

### Precision & Recall (the ingredients of F1)
- **Precision** = of the days you *predicted* up, how many were up? `TP / (TP + FP)`. Example: `2/3 =
  0.67`. "When it says up, how often is it right?"
- **Recall** = of the days that *were* up, how many did you catch? `TP / (TP + FN)`. Example: `2/3 =
  0.67`. "Of all the ups, how many did it find?"

There's a tension: predict "up" for everything → recall 100% but precision poor. F1 balances them.

### F1 score / macro-F1
**F1** = the harmonic mean of precision and recall: `2·P·R / (P + R)`. It's high only when *both* are
high. Ranges 0–1.
**macro-F1** = compute F1 for class 1 *and* for class 0 (treating each as "the positive class"), then
**average the two equally**. This is what we report. Why macro? It refuses to ignore the minority class:
a model that nails "up" but is useless on "down" gets a mediocre macro-F1, even if accuracy looks fine.
**Good value here:** tracks accuracy; ~0.50–0.55 for direction, ~0.70 for volatility.
**In code:** `f1_score(..., average="macro")`.

### MCC — Matthews Correlation Coefficient ⭐ (the metric we trust)
**What:** a single correlation between predictions and truth, from **−1 to +1**:
- **+1** = perfect, **0** = no better than random guessing, **−1** = perfectly wrong.

**Formula:**
```
        TP·TN − FP·FN
MCC = ─────────────────────────────────────
      √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
Example: `(2·1 − 1·1) / √(3·3·2·2) = 1/6 ≈ +0.167`.

**Why it's the star of this project:** MCC only rises above 0 when the model beats chance across **both**
classes at once. The always-predict-majority trick that fools accuracy scores **MCC = 0**. So MCC is
"skill after removing the free points you get from class imbalance." It's also the metric the StockNet
paper uses, so our numbers are comparable to theirs.

**Good value here — calibrate your expectations:** MCC on hard financial tasks is *small*.
- Direction: **+0.04 to +0.12** is a real, publishable edge. (Yes, 0.1 is "good" here.)
- Volatility: **~+0.40** — genuinely strong, because the task is genuinely predictable.
- Negative MCC = the model is anti-correlated with truth on that slice (usually noise on a small sample).

**In code:** `matthews_corrcoef`, in `score`. Tested against hand-computed values in
`tests/test_evaluate.py`.

> **Rule of thumb for this repo:** *accuracy tells you how often you're right; MCC tells you whether that
> was skill or just the base rate.* Read them together, always next to the baseline.

---

## Part 2 — probability & confidence metrics

The models don't just output 0/1; they output a **probability** `p` that the answer is 1 (e.g. `p = 0.63`
→ "63% chance it goes up"). Two ideas use that number.

### Confidence & the conviction curve
**Confidence** = how far `p` is from the 0.5 fence: `|p − 0.5|`. `p = 0.95` is confident; `p = 0.51` is a
shrug. The **conviction curve** (`predictor/analysis.py`) keeps only the most confident X% of days
("coverage") and measures accuracy on just those. If the model has real skill, its confident calls are
more accurate than its average calls — so accuracy should *rise* as coverage shrinks. In our 10-day
direction result, restricting to the top-10% most confident days lifted MCC from ~0.01 to +0.074.
**Coverage** = what fraction of days you chose to act on (1.0 = all, 0.1 = only the top 10%). It's the
"how selective are you?" knob; there's a trade-off between being selective (higher accuracy) and having
enough opportunities (coverage).

### Regression metrics: R² and RMSE (Task B magnitude)
Task B also has a **regression** version — predict the *actual* volatility level (a number), not just
up/down. Regression needs different metrics:
- **RMSE** (root mean squared error) = `√(mean((prediction − truth)²))` — the typical error, in the same
  units as the target. **Lower is better.** Good for "how far off am I," bad for comparing across
  different targets (units differ).
- **R²** (coefficient of determination) = the fraction of the target's variance the model explains.
  `R² = 1 − (model's squared error) / (squared error of just predicting the mean)`.
  - **R² = 1** perfect · **R² = 0** no better than always guessing the average · **R² < 0** *worse* than
    guessing the average (yes, it can go negative — that's how we caught the persistence baseline failing).
  - **Good value here:** volatility R² of **0.15–0.27** with daily data is a real, useful signal; 0.4–0.6
    is achievable with intraday data. (Context matters — 0.2 is respectable for noisy financial targets.)
- **Persistence baseline** (regression's version of the majority baseline): predict "next = current."
  The model must beat it to show skill — and at short horizons ours *doesn't* just copy the present, it
  genuinely improves on it. **In code:** `r2_score`, `mean_squared_error` in `predictor/volatility.py`.

### Brier score & calibration
**Calibration** asks: when the model says "70% chance", does it actually happen ~70% of the time?
A well-calibrated model's probabilities mean what they say.
**Brier score** measures this: the mean squared error between the probability and the outcome,
`mean((p − y)²)`. **Lower is better** (0 = perfect; 0.25 = the score of always guessing 0.5).
In our project, naive calibration on the dev split actually made the test Brier *worse* (0.25 → 0.32) —
an honest finding that the dev and test periods were different market regimes, so a calibration fit on the
past didn't transfer to the future.
**In code:** `brier_score_loss` + `CalibratedClassifierCV` in `conviction_curve`.

---

## Part 3 — evaluation methodology (how we keep the numbers honest)

Metrics are only as trustworthy as *how* you computed them. These concepts matter as much as the metrics.

### Train / dev / test split
- **Train** — the data the model learns from.
- **Dev** (validation) — used to *choose* settings (e.g. XGBoost hyperparameters). Looked at many times.
- **Test** — touched **once**, at the very end, to report the final number. If you tune on test, your
  number is optimistic fiction.

### Chronological split (no shuffling)
In finance you must split by **time**: train on the past, test on the future — never shuffle days
randomly. We use StockNet's own dates (train 2014→Aug 2015, dev, then test Oct–Dec 2015). Random
shuffling would let the model "see the future," inflating every metric.

### Lookahead bias
Using any information that wouldn't have been available at prediction time. A feature for day *t* must use
only data up to *t*. We enforce this with a test that perturbs a *future* price and asserts no past
feature changes (`tests/test_build_features.py`). It's the cardinal sin of backtesting because it makes
fake models look brilliant.

### The overlapping-window embargo (purging)
Because a horizon-*h* label spans days *t*+1…*t*+*h*, neighbouring rows' outcomes **overlap**, which
leaks information across the train/test boundary and inflates scores. The **embargo** keeps a train/dev
row only if its outcome window stays inside its own split. It's why our honest numbers are a touch lower
than the naive ones. (`predictor/train_predictor.py::embargoed_frame`.)

### Walk-forward evaluation & "mean ± std" ⭐
One test window gives **one** number that might be luck. **Walk-forward** rolls several consecutive
out-of-sample windows through time and reports the **mean ± standard deviation** across them.
- **mean** = the typical performance.
- **std (standard deviation)** = how much it wobbles window to window = your **confidence interval**.

How to read it: **`+0.088 ± 0.030`** means "about 0.088, give or take 0.030" — since the mean is ~3× the
spread, the edge is real. But **`+0.079 ± 0.093`** means the wobble is *bigger* than the signal — you
**cannot** claim an edge; it's statistically indistinguishable from zero. This single idea is why we
trust the 5–10 day direction edge but *not* the flashy single-window 20-day number.
**In code:** `predictor/walkforward.py`, `predictor/volatility.py::walk_forward`.

### Non-overlapping cross-check
An independent, stricter re-run that samples rows *h* apart so labels never share a day at all — a second
opinion that the overlapping numbers weren't an autocorrelation illusion. (`analysis.py`.)

---

## Part 4 — a few model/feature terms you'll see

- **Standardization** (`StandardScaler`): rescales each feature to mean 0, std 1 so a linear model treats
  them fairly. Fit on **train only** (fitting on test would leak information).
- **class_weight="balanced"**: tells the model to care equally about both classes even if one is rarer —
  counters imbalance during *training* (complements MCC during *evaluation*).
- **scale_pos_weight** (XGBoost): the same idea for trees — up-weights the minority class.
- **Hyperparameter tuning**: trying many model settings and keeping the best — here chosen by **dev-set
  MCC** (`tune_xgboost`), never test.
- **LogisticRegression**: the simple, transparent linear baseline. Often competitive here.
- **XGBoost**: gradient-boosted decision trees; can capture nonlinear interactions; what quant desks reach
  for on tabular data.
- **√h threshold scaling** (Task A): the up/down "dead band" grows with the square root of the horizon,
  because returns scale ~√time — keeps the drop-rate of ambiguous days constant so horizons are compared
  fairly.

---

## The sentiment model's metrics

The DistilBERT classifier is a **3-class** problem (Bearish / Bullish / Neutral), so:
- **accuracy** = fraction of tweets labelled correctly (**0.881**).
- **macro-F1** = average F1 across all three classes (**0.846**) — confirms it's good on every class, not
  just the common one.
Same metrics as above, just three classes instead of two.

---

## Cheat sheet

| metric | range | reading | good here | fooled by |
|--------|:-----:|---------|-----------|-----------|
| **Accuracy** | 0–1 | % correct | >baseline; ~0.54 (dir), ~0.72 (vol) | class imbalance |
| **Majority baseline** | 0–1 | always-predict-common accuracy | you must beat it | — |
| **Precision** | 0–1 | when it says 1, how often right | context-dependent | predicting 1 rarely |
| **Recall** | 0–1 | of real 1s, how many caught | context-dependent | predicting 1 always |
| **macro-F1** | 0–1 | balance of P & R, both classes | ~0.55 (dir), ~0.70 (vol) | less than accuracy is |
| **MCC** ⭐ | −1…+1 | skill beyond chance | +0.04–0.12 (dir), ~0.40 (vol) | almost nothing |
| **RMSE** | 0→ | typical regression error (lower=better) | small; compare within a target | scale differences |
| **R²** | ≤1 | variance explained (regression) | 0.15–0.27 (vol, daily data) | — |
| **Brier** | 0–1 | probability calibration (lower=better) | <0.25 | — |
| **Coverage** | 0–1 | fraction of days acted on | trade-off vs accuracy | — |
| **mean ± std** | — | value ± window-to-window wobble | mean ≫ std = real | one lucky window |

**The one-sentence version:** *accuracy = how often right; baseline = the free points; MCC = the real
skill; mean ± std = is it repeatable or luck.* If you can read a result table with those four in mind,
you understand this project's numbers.
