# Metrics — what every number means

Plain-language guide to every statistic this project reports, what a good value looks like
**on this data**, and how each one can mislead you if read alone.

---

## What is being predicted

**Realized volatility**: how much prices actually moved over a window, measured as the
root-mean-square of daily returns. Not direction. Not returns. Just magnitude of movement.

Two framings of the same forecast:

| framing | question | metric family |
|---|---|---|
| **regression** | *how volatile* will the next 10 days be? | R², RMSE, skill vs persistence |
| **classification** | will volatility **rise or fall** vs now? | accuracy, MCC |
| **regime** | calm, normal or stressed? | 3-class accuracy, stressed recall |

All three come from **one model**. The rise/fall arrow is just the predicted level compared
to the current level — there is no second classifier that could disagree with the number
next to it.

### Why logs

The model predicts **log** realized volatility. Volatility is roughly log-normal and far
more persistent in log space, which is why the HAR literature reports much higher R² there.
Every number shown to a user is converted back to **annualized percent** — multiply the
daily figure by √252.

---

## Regression metrics

### R² — variance explained
`R² = 1 − (model's squared error) / (squared error of always predicting the mean)`

- **1.0** perfect · **0** no better than guessing the average · **negative** worse than that.
- **Good here:** ~0.45 pooled out-of-sample is strong for a volatility forecast.

**The trap — the denominator moves.** R² is relative to the variance *of the window you
score on*. In this data, within-year variance of log-vol ranges from 0.07 (calm 2022) to
0.42 (2020), against 0.275 for the whole period. So the *same forecast* scores very
differently depending on whether you slice by year or pool. A model can forecast well and
still post a negative per-year R², because beating a single year's average requires having
known that average in advance.

This project reports **pooled out-of-sample R²** as the headline (every prediction still
comes from a model that saw nothing after its own year's cutoff — only the scoring window
changed) and shows the per-year spread alongside so the instability is visible.

### Skill vs persistence ⭐ — the metric that can't be gamed
`skill = 1 − (model's squared error) / (persistence's squared error)`

Persistence means "next 10 days will be as volatile as the last 10." It is free, it is
strong, and it is the honest thing to beat. Because **both sides are scored on the same
window**, this metric is immune to the moving-denominator problem above — which makes it
the right stability check.

- **0** = no better than assuming nothing changes · **0.35** = removed 35% of the
  baseline's squared error.
- **Good here:** 0.30–0.43 at 5 and 10 days, positive in **13 of 13** walk-forward years.

### RMSE / MAE
Typical error size in log units. Useful for comparing models on the same target, useless
across different targets. Lower is better.

---

## Classification metrics

### Accuracy, and the baseline that must sit next to it
The fraction of correct rise/fall calls. **Never meaningful alone.** If 80% of days see
volatility rise, always saying "rise" scores 80% while knowing nothing. Every accuracy in
this project is printed beside its **majority baseline**, and the gap is what matters.

### MCC — Matthews Correlation Coefficient ⭐
A single correlation between predictions and truth, from **−1 to +1**.

```
        TP·TN − FP·FN
MCC = ─────────────────────────────
      √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

MCC rises above 0 only when the model beats chance on **both** classes. The
always-predict-majority trick that fools accuracy scores exactly **0**. That is why it is
the metric to trust.

- **Good here:** ~0.42–0.47 for volatility rise/fall — genuinely strong, because the task
  is genuinely predictable. (For comparison, stock *direction* tops out near 0.1, which is
  why this project does not forecast direction.)

### Stressed recall
Of the periods that really were stressed, how many did the model label stressed? Reported
separately because the stressed class is small and the one that matters — an overall regime
accuracy can look fine while missing every crisis.

---

## Evaluation methodology

### Chronological splits
Train on the past, test on the future. Days are never shuffled. Random shuffling lets a
model see the future and inflates every metric.

### Lookahead bias
Using information unavailable at prediction time. Enforced behaviourally: a test perturbs a
**future** bar and asserts that no feature on any **earlier** row changes
(`tests/test_causality.py`). A mirror test asserts the *target* does move, so a bug that
froze the target could not pass as good hygiene.

### The embargo (purging)
A 10-day label spans days t+1…t+10, so a row dated late December has an outcome reaching
into January. Training on it while testing on January leaks the answer. Rows whose outcome
window crosses a split boundary are **dropped**. This lowers the reported numbers — that is
the point.

### Walk-forward ⭐
One test window gives one number that might be luck. Walk-forward retrains at the start of
each year on all prior data and scores the following year — **13 independent out-of-sample
windows**. Report the **mean and the spread**:

- `+0.355 ± 0.204` — mean well clear of the spread, and positive in every year. Real.
- `−0.268 ± 1.423` — spread dwarfs the mean. **No edge.** (This is the 21-day horizon, which
  is why it is not shipped.)

### Frozen regime thresholds
Calm/normal/stressed cut points come from the **training period only**. Computing them on
the full sample would let a quiet decade silently redefine "stressed" using future
information.

---

## Event-study metrics

- **Detection rate** — share of realized stress episodes where the model raised an alarm in
  the 60 sessions before onset.
- **Lead time** — trading days between first alarm and onset.
- **False-alarm rate** — share of alarm days not followed by stress.

**Always read these three together.** A model that alarms every day detects 100% of events
and is worthless. This is why the full threshold sweep is published rather than one chosen
operating point.

---

## Decision-utility metrics

From the volatility-targeting simulation:

- **Sharpe** — return per unit of risk. Higher is better.
- **Max drawdown** — worst peak-to-trough loss.
- **Target miss** — how far the strategy's realized volatility landed from its 15% target.
  This is the metric a volatility forecast most directly controls, so it is the fairest test
  of whether the forecast worked.
- **Turnover** — how much trading the rule required. A signal that only pays off before
  costs has not paid off.

The comparison that counts is **model-sized vs persistence-sized**, not vs buy & hold.
Persistence-sized already uses a volatility estimate — it just measures instead of
forecasting. Beating buy & hold shows volatility targeting works; beating persistence shows
the *forecast* works.

---

## Cheat sheet

| metric | range | good here | fooled by |
|---|:---:|---|---|
| **R² (pooled OOS)** | ≤1 | ~0.45 | the scoring window's own variance |
| **Skill vs persistence** ⭐ | ≤1 | 0.30–0.43, positive every year | almost nothing |
| **Accuracy** | 0–1 | ~0.70 vs ~0.51 baseline | class imbalance |
| **MCC** ⭐ | −1…1 | 0.42–0.47 | almost nothing |
| **Regime accuracy** | 0–1 | must beat majority baseline | a dominant class |
| **Detection rate** | 0–1 | read with false-alarm rate | alarming constantly |
| **Sharpe** | — | vs persistence arm, not buy & hold | ignoring costs |
| **mean ± std** | — | mean > 2× std | one lucky window |

**One sentence:** *accuracy says how often you were right; the baseline says how many of
those were free; skill vs persistence says whether forecasting beat merely measuring; and
mean ± std says whether any of it repeats.*
