# Financial Sentiment & Market Signal Analysis

**How far ahead can you actually predict a stock's direction — and how much does Twitter sentiment
help?** This project answers that honestly, on a real published benchmark, entirely on a laptop CPU.

It predicts whether a stock moves **up or down** over several horizons — **1, 5, 10, 20, and 60
trading days** — by combining:

- **Price factors** — momentum, moving-average ratios, rolling volatility, volume change, **RSI(14)**, **MACD(12,26,9)**.
- **A sentiment factor** — a fine-tuned **DistilBERT** reads financial tweets and scores them bullish / bearish / neutral (a *minor* supporting input).

Everything is evaluated on the **StockNet** benchmark (Xu & Cohen, ACL 2018) using its **original
chronological split**, plus a **walk-forward** re-evaluation for confidence intervals. No GPU, no cloud.

> **This is a study in honest measurement, not a get-rich model.** Short-horizon stock direction is
> near-random; no legitimate model reaches 0.80 here. The value is a defensible answer, with every
> number guarded against the ways backtests usually lie.

---

## TL;DR — what we found

- **A small but real directional edge exists at short-to-medium horizons (~1–10 trading days).** It
  survives a walk-forward across 5 out-of-sample windows *and* a stricter non-overlapping cross-check:
  MCC ≈ **+0.04 to +0.09**, accuracy ≈ **51.6% → 54.4%**.
- **The edge does *not* reliably extend past ~10 days.** By a month the confidence interval swallows the
  mean; by a quarter (60d) the edge is gone (**MCC −0.02 ± 0.07**). Single-window numbers at 20/60d
  looked great but were mirages — walk-forward exposes them.
- **Predictability is uneven across sectors** — Financials are the most predictable (MCC +0.15), Services
  the least (−0.25).
- **Acting only on the most confident days helps** — restricting to the top-10% highest-conviction
  10-day predictions lifts MCC to +0.074.
- **Sentiment is a genuine but minor factor.** The fine-tuned classifier is strong on its own
  (**88.1% accuracy**), but adding it to price features moves the needle only slightly.

The headline is the **walk-forward horizon curve** below. Everything else supports or stress-tests it.

---

## Contents
- [The honest headline: walk-forward horizon curve](#the-honest-headline-walk-forward-horizon-curve)
- [Architecture](#architecture)
- [All results](#all-results)
- [How these numbers stay honest](#how-these-numbers-stay-honest)
- [Quickstart](#quickstart)
- [Repo layout](#repo-layout)
- [Limitations](#limitations) · [Future work](#future-work)
- [The sentiment model](#the-sentiment-model) · [Citation & license](#citation--license)

---

## The honest headline: walk-forward horizon curve

Instead of trusting one fixed test window, we roll **5 consecutive quarterly out-of-sample windows**
through 2014–2016 and report the **mean ± standard deviation** of accuracy and MCC per horizon
(LogisticRegression on price + sentiment). This is the single biggest reason to trust the result.

| horizon | accuracy (mean ± std) | **MCC (mean ± std)** | verdict |
|--------:|:---------------------:|:--------------------:|---------|
| **1 day**  | 0.516 ± 0.010 | **+0.037 ± 0.020** | small, fairly stable edge |
| **5 day**  | 0.531 ± 0.004 | **+0.068 ± 0.021** | ✅ strongest reliable edge (mean > 3× std) |
| **10 day** | 0.544 ± 0.011 | **+0.088 ± 0.030** | ✅ edge holds, variance growing |
| **20 day** | 0.538 ± 0.046 | +0.079 ± 0.093 | ⚠️ mean ≈ 1× std — not distinguishable from 0 |
| **60 day** | 0.488 ± 0.036 | −0.023 ± 0.074 | ❌ no edge |

**Read it like this:** at **5 and 10 days** the mean MCC is comfortably larger than its own spread —
a real signal. At **20 days** the spread (±0.093) is bigger than the mean (+0.079), so we *cannot*
claim an edge. At **60 days** it's gone. Predictability rises, peaks around a week or two, then drowns
in noise. That is the honest answer to "how far ahead can we predict?"

MCC = Matthews Correlation Coefficient (−1…1): on a ~50/50 target, accuracy can be fooled by always
predicting the majority direction, while MCC only rises when the model beats chance across **both**
directions. It's the metric StockNet uses, so our numbers are comparable to the paper.

---

## Architecture

```
Twitter Financial News Sentiment (11,932 pre-labeled tweets)
        │  fine-tune DistilBERT (CPU)
        ▼
  sentiment model (88% acc) ──────────────┐
                                          │ score every StockNet tweet
StockNet (88 tickers, 9 sectors,          │ aggregate -> daily sentiment factor
 tweets + prices 2014–2016)               ▼
        │                        sentiment factors (incl. 3-day trailing)
        ├── price factors ────────────────┤
        │   momentum · MA · vol · volume  │
        │   · RSI · MACD                  │
        ▼                                 ▼
   join per ticker-day  (ONLY past data — no lookahead)
        │  chronological split  +  overlapping-window EMBARGO
        ▼
   for each horizon h ∈ {1,5,10,20,60}:
        price-only / sentiment-only / combined
        × LogisticRegression + XGBoost (tuned on dev by MCC)
        ▼
   ├─ fixed-window comparison table        (predictor/evaluate.py)
   ├─ walk-forward mean ± std  ◀ headline  (predictor/walkforward.py)
   └─ non-overlap · conviction · sector    (predictor/analysis.py)
```

---

## All results

### 1. Fixed-window comparison (StockNet's own Oct–Dec 2015 test split)

Best variant per horizon on the single official test window. Useful, but **treat the long horizons
with suspicion** — the walk-forward above shows why.

| horizon | best variant | accuracy | macro-F1 | MCC | note |
|--------:|--------------|:--------:|:--------:|:---:|------|
| 1  | logreg / sentiment-only | 0.531 | 0.520 | +0.068 | |
| 5  | logreg / combined       | 0.537 | 0.536 | +0.074 | |
| 10 | xgboost / combined      | 0.564 | 0.562 | +0.124 | strongest single-window |
| 20 | logreg / sentiment-only | 0.607 | 0.574 | +0.151 | ⚠️ unstable (price-only MCC is −0.12 here) |
| 60 | xgboost / price-only    | 0.376 | 0.319 | +0.075 | ❌ unreliable — accuracy < 0.5; dev window too short to tune |

<details><summary>Full table — every variant × horizon</summary>

Generated by `python predictor/evaluate.py` → `data/results_comparison.csv`. The 60-day XGBoost rows
use default hyperparameters because a 60-day label overruns the 2-month dev window, so the embargo
leaves **zero** dev rows to tune on — itself an honest finding about long-horizon evaluation on this
dataset.

| h | model | acc | F1 | MCC |
|--:|-------|:--:|:--:|:---:|
| 1 | logreg/price-only | 0.509 | 0.507 | +0.019 |
| 1 | xgboost/price-only | 0.491 | 0.487 | −0.020 |
| 1 | logreg/sentiment-only | 0.531 | 0.520 | +0.068 |
| 1 | xgboost/sentiment-only | 0.512 | 0.501 | +0.027 |
| 1 | logreg/combined | 0.518 | 0.516 | +0.037 |
| 1 | xgboost/combined | 0.491 | 0.488 | −0.020 |
| 5 | logreg/price-only | 0.526 | 0.526 | +0.052 |
| 5 | xgboost/price-only | 0.518 | 0.517 | +0.035 |
| 5 | logreg/sentiment-only | 0.534 | 0.524 | +0.069 |
| 5 | xgboost/sentiment-only | 0.512 | 0.502 | +0.025 |
| 5 | logreg/combined | 0.537 | 0.536 | +0.074 |
| 5 | xgboost/combined | 0.528 | 0.527 | +0.055 |
| 10 | logreg/price-only | 0.508 | 0.504 | +0.009 |
| 10 | xgboost/price-only | 0.556 | 0.553 | +0.107 |
| 10 | logreg/sentiment-only | 0.551 | 0.531 | +0.075 |
| 10 | xgboost/sentiment-only | 0.436 | 0.348 | −0.081 |
| 10 | logreg/combined | 0.517 | 0.514 | +0.028 |
| 10 | xgboost/combined | 0.564 | 0.562 | +0.124 |
| 20 | logreg/price-only | 0.416 | 0.413 | −0.120 |
| 20 | xgboost/price-only | 0.422 | 0.418 | −0.115 |
| 20 | logreg/sentiment-only | 0.607 | 0.574 | +0.151 |
| 20 | xgboost/sentiment-only | 0.337 | 0.257 | −0.087 |
| 20 | logreg/combined | 0.428 | 0.425 | −0.093 |
| 20 | xgboost/combined | 0.450 | 0.445 | −0.063 |
| 60 | logreg/price-only | 0.530 | 0.503 | +0.013 |
| 60 | xgboost/price-only | 0.376 | 0.319 | +0.075 |
| 60 | logreg/sentiment-only | 0.517 | 0.491 | −0.011 |
| 60 | xgboost/sentiment-only | 0.344 | 0.256 | +0.012 |
| 60 | logreg/combined | 0.527 | 0.497 | −0.002 |
| 60 | xgboost/combined | 0.377 | 0.335 | +0.019 |

</details>

### 2. Non-overlapping cross-check (`predictor/analysis.py`)

The fixed/walk-forward numbers use *overlapping* labels (every day gets its next-h-day return, so
neighbouring rows share days). Here we instead keep only **non-overlapping** windows (rows h apart, so
labels share *no* days) — the strictest test that the edge isn't an autocorrelation artefact.

| horizon | n_test | accuracy | MCC |
|--------:|:------:|:--------:|:---:|
| 1  | 3744 | 0.518 | +0.037 |
| 5  |  749 | 0.521 | +0.041 |
| 10 |  383 | 0.512 | +0.022 |
| 20 |  186 | 0.441 | −0.092 |
| 60 |   62 | 0.500 | +0.015 |

The edge is smaller here than in the overlapping tables — an honest sign the overlapping numbers were
somewhat optimistic — but it stays **positive at 1–10 days** and collapses beyond. Consistent with the
walk-forward story.

### 3. Conviction curve — trade only your most confident days (`predictor/analysis.py`)

If the model has any real skill, its high-confidence predictions should be more accurate than its
average ones. Restricting to the most confident fraction of **10-day** test days:

| coverage | days | accuracy | MCC |
|---------:|:----:|:--------:|:---:|
| 100% | 3813 | 0.549 | +0.012 |
| 50%  | 1906 | 0.559 | +0.028 |
| 25%  |  953 | 0.511 | +0.057 |
| 10%  |  381 | 0.517 | **+0.074** |

Confidence-ranking does concentrate signal (MCC climbs as coverage shrinks). One honest caveat: naive
**probability calibration** (sigmoid, fit on dev) *worsened* the Brier score on test (0.251 → 0.324),
because the dev and test periods are different market regimes — so we rank by raw confidence rather
than trust the recalibrated probabilities. Calibrating across regime shift is itself hard.

### 4. Per-sector models (`predictor/analysis.py`, horizon = 10d)

A separate model per StockNet sector. Predictability is clearly uneven (small per-sector samples, so
read with care):

| sector | n_test | accuracy | MCC |
|--------|:------:|:--------:|:---:|
| Financial | 422 | 0.576 | **+0.154** |
| Utilities | 376 | 0.545 | +0.090 |
| Consumer Goods | 433 | 0.517 | +0.056 |
| Healthcare | 432 | 0.512 | +0.045 |
| Technology | 408 | 0.515 | +0.040 |
| Basic Materials | 519 | 0.511 | +0.026 |
| Industrial Goods | 415 | 0.511 | +0.023 |
| Conglomerates | 353 | 0.473 | +0.018 |
| Services | 455 | 0.378 | −0.245 |

### 5. The sentiment model (held-out validation split)

| metric | value |
|--------|-------|
| accuracy | **0.881** |
| macro-F1 | **0.846** |

A strong standalone financial-tweet classifier (DistilBERT, CPU, 3 epochs) — used here as one factor
feeding the movement models. See [The sentiment model](#the-sentiment-model).

---

## How these numbers stay honest

Short-horizon direction is close to a coin flip, so the easy failure mode is a backtest that *looks*
brilliant because it quietly cheated. Every guard below exists to prevent that:

1. **No-lookahead features.** Every feature for day *t* uses only data up to *t* (rolling/EWM windows,
   causal RSI/MACD). Enforced by a test that perturbs a *future* price and asserts no past feature —
   including RSI/MACD — changes (`tests/test_build_features.py`).
2. **Overlapping-window embargo (purging).** Labeling every day with its next-*h*-day return makes
   neighbouring rows' outcomes overlap, which inflates scores and leaks across split boundaries. We keep
   a train/dev row only if its outcome day (*t+h*) lands in the **same split** as *t*; test rows may run
   their outcome past the window using real later prices (not contamination, since nothing trains on
   test). Tested in `tests/test_evaluate.py`.
3. **Fair thresholds across horizons.** The up/down dead-band is scaled by **√h**, so a higher number at
   10 days reflects real signal, not a looser bar on bigger moves.
4. **Tune on dev, touch test once.** XGBoost hyperparameters are selected by **dev-split MCC**; the test
   split is scored a single time.
5. **Walk-forward for confidence intervals.** Five out-of-sample windows turn a single point estimate
   into a mean ± std — the difference between "10-day MCC is 0.12" and "it's 0.09 ± 0.03, and 20-day is
   indistinguishable from zero."
6. **Non-overlapping cross-check.** An independent, stricter re-run with zero label overlap.

**No result here approaches 0.80, and that's the point.** Numbers like that on next-day/next-week
direction come from leakage or overfitting — the exact things these guards rule out.

---

## Quickstart

```bash
# 1. Environment (Python 3.11, CPU)
py -3.11 -m venv .venv && .venv/Scripts/activate      # Windows
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Get StockNet (tweets + prices)
git clone https://github.com/yumoxu/stocknet-dataset data/stocknet-dataset

# 3. Fine-tune the sentiment model (~30 min CPU; downloads the tweet dataset automatically)
python sentiment_model/train_sentiment_model.py

# 4. Build features (scores every tweet, then price + sentiment factors + multi-horizon labels)
python features/build_features.py

# 5. Evaluate
python predictor/evaluate.py        # fixed-window horizon comparison table
python predictor/walkforward.py     # walk-forward mean ± std  (the headline)
python predictor/analysis.py        # non-overlap + conviction + per-sector studies

# 6. (optional) interactive demo — pick a ticker + horizon
streamlit run app.py

# tests
pytest -q
```

Already built the features once? Re-running steps 4–5 is cheap; the expensive tweet-scoring is cached,
so `python features/build_features.py --reuse-sentiment-cache` skips it.

---

## Repo layout
```
config.py                       # paths, seed, horizons, √h thresholds, feature lists, date split
sentiment_model/
  train_sentiment_model.py      # fine-tune DistilBERT (bullish/bearish/neutral)
  model_card.md                 # Hugging Face model card
features/build_features.py      # tweet scoring + price factors (RSI/MACD) + trailing sentiment
                                #   + multi-horizon labels + embargo columns + sector map
predictor/
  train_predictor.py            # LogReg + dev-tuned XGBoost, 3 variants, per horizon, embargoed
  evaluate.py                   # fixed-window accuracy / F1 / MCC horizon table
  walkforward.py                # rolling out-of-sample windows -> mean ± std   ◀ headline
  analysis.py                   # non-overlap cross-check · conviction curve · per-sector models
app.py                          # Streamlit demo (ticker + horizon selector)
tests/                          # leakage, RSI/MACD, horizon labels, embargo, metrics, non-overlap, sectors
.github/workflows/ci.yml        # runs the test suite on every push
```

---

## Limitations
- **One dataset, one 2014–2016 regime.** StockNet's tweet coverage is 2014–2016; results may not
  generalize to other periods or to less-covered tickers.
- **Short dev/test windows** make long-horizon evaluation fragile — the 60-day horizon can't even be
  tuned (its labels overrun the 2-month dev window under the embargo).
- **Tweet sentiment is coarse and dated.** Three classes, pre-tokenized text with placeholder tokens,
  and a labeling era that may not match every ticker's chatter.
- **No transaction costs, slippage, or position sizing.** MCC/accuracy measure *directional* skill, not
  a tradeable strategy.

## Future work
- **Non-overlapping weekly bars as the primary label** (not just a cross-check) for a cleaner long-horizon read.
- **Stronger sentiment factor** — fine-tune `roberta-base`/FinBERT, or aggregate sentiment over longer trailing windows.
- **Probability calibration that survives regime shift** (e.g. calibrate on a rolling recent window).
- **Confidence-thresholded strategy** with realistic costs, to see where a directional edge becomes usable.
- **More/finer horizons and more walk-forward folds** for tighter confidence intervals.

---

## The sentiment model

`distilbert-base-uncased` fine-tuned on
[`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
(≈9.5k train / 2.4k val; labels 0=Bearish, 1=Bullish, 2=Neutral) — **88.1% accuracy, 0.846 macro-F1**,
trained on CPU in ~30 min. Full details, intended use, and limitations in
[`sentiment_model/model_card.md`](sentiment_model/model_card.md).

**Hugging Face Hub:** _upload pending_ — `huggingface-cli login`, then push `data/sentiment_model/` with
the model card (link filled in here after upload).

```python
from transformers import pipeline
clf = pipeline("text-classification", model="<your-hf-username>/distilbert-financial-tweet-sentiment")
clf("$AAPL breaking out to new highs, strong buy")   # -> Bullish
```

---

## Citation & license

> Yumo Xu and Shay B. Cohen. 2018. *Stock Movement Prediction from Tweets and Historical Prices.*
> Proceedings of ACL 2018.

Educational project — **not investment advice.** Datasets retain their own licenses (Twitter Financial
News Sentiment: MIT; StockNet: see its repository).
