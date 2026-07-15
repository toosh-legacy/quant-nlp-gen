# Financial Sentiment & Market Signal Analysis

A CPU-only **quant + NLP** study on the **StockNet** benchmark (Xu & Cohen, ACL 2018) and a large
2005–2024 stock universe. It asks two questions honestly — *can you predict a stock's **direction**?*
and *can you predict its **volatility**?* — and is rigorous about the difference.

> **The whole point is honest measurement.** Stock *direction* is near-random (~54%); *volatility* is
> genuinely predictable (R² ≈ 0.42). Knowing which is which — and **proving** it — is the deliverable.

---

## TL;DR

- 📉 **Direction is near-random.** A real but tiny edge at 1–10 days (MCC ≈ +0.07–0.09, ~52–54% accuracy), gone within a month. Nothing hits the mythical 0.80 on daily/weekly direction — that's leakage, not skill.
- 📊 **Volatility is predictable.** Forecasting log realized volatility reaches **R² ≈ 0.42** out-of-sample (positive in all 10 walk-forward years); "will volatility rise?" hits **72% accuracy / MCC 0.45**.
- 💬 **Sentiment model:** DistilBERT fine-tuned on financial tweets → **88% accuracy** — a strong standalone NLP module (only a minor factor for prediction).
- 🔎 **Better data > fancier models.** The volatility R² roughly *doubled* by improving the **data and features** (bigger universe, log target, cross-sectional + range-based estimators), not the algorithm.

---

## What we built

| task | what it does | result |
|---|---|---|
| **Sentiment (NLP)** | Fine-tune DistilBERT to tag tweets bullish / bearish / neutral | **88.1%** acc · 0.85 macro-F1 |
| **Direction** | Predict up/down over 1–60 days from price + sentiment factors | honest ~54%, MCC ≤ 0.09 |
| **Volatility** | Predict next-week vol regime + magnitude on 80 stocks, 2005–2024 | **R² ≈ 0.42**, 72% up/down |

---

## Results

### 📊 Volatility — the strong result · `predictor/volatility_universe.py`
Predicting **log realized volatility** on ~80 large-caps, 2005–2024 (~380k ticker-days), with
cross-sectional and range-based (**Garman-Klass**) features. R² = fraction of variance explained.

| horizon | persistence (baseline) | HAR-OLS | **XGBoost (tuned)** |
|--------:|:----------------------:|:-------:|:-------------------:|
| 5 day  | −0.08 | 0.30 | **0.37** |
| 10 day |  0.08 | 0.35 | **0.42** |
| 21 day |  0.13 | 0.35 | **0.41** |

*Out-of-sample (test 2018–2024). Walk-forward (10 yearly folds): R² **0.29 / 0.34 / 0.36**, positive every
year. "Will vol rise?" classification: **72% accuracy, MCC 0.45**.*

### 📉 Direction — the honest result · `predictor/walkforward.py`
Walk-forward across 5 windows; MCC (−1…1) by horizon. Read *mean vs its own std*:

| horizon | MCC (mean ± std) | verdict |
|--------:|:----------------:|---------|
| 5 day  | +0.068 ± 0.021 | ✅ small, real edge |
| 10 day | +0.088 ± 0.030 | ✅ edge holds |
| 20 day | +0.079 ± 0.093 | ⚠️ indistinguishable from 0 |
| 60 day | −0.023 ± 0.074 | ❌ no edge |

*Predictability rises to ~2 weeks, then drowns in noise. ~52–54% accuracy — exactly where the
efficient-market literature says it should be.*

### 💬 Sentiment classifier · `sentiment_model/`
DistilBERT fine-tuned on `zeroshot/twitter-financial-news-sentiment`: **88.1% accuracy, 0.846 macro-F1**
on held-out validation. (Hugging Face upload optional — see the model card.)

---

## How we kept it honest (methodology)

- **Chronological splits** — always train on the past, test on the future; never shuffle days.
- **No lookahead** — every feature uses only data up to its own day; enforced by a test that perturbs a *future* price and asserts no past feature changes.
- **Overlapping-window embargo** — drop rows whose multi-day outcome window crosses a split boundary (purging), so a training label never overlaps the test period.
- **Walk-forward confidence intervals** — report mean ± std across windows, so a real edge is distinguishable from a lucky one.
- **Honest baselines + MCC** — every number sits next to a majority/persistence baseline, and we report MCC (not just accuracy) so class imbalance can't fake skill.
- **Negative results kept** — VIX added nothing, calibration failed under regime shift, the Reddit-news dataset was skipped. All documented, not hidden.

New to any of these terms? **[`docs/METRICS.md`](docs/METRICS.md)** explains every metric in plain language.

---

## Repo guide (what's where)

```
config.py                                  # single source of truth: paths, horizons, features, splits
sentiment_model/train_sentiment_model.py   # fine-tune DistilBERT   ← the NLP module
features/
  build_features.py                        # StockNet: price/technical + sentiment + labels (Tasks A & B)
  universe_data.py                         # download 80-stock 2005-2024 universe + vol features (Task B pro)
predictor/
  train_predictor.py · evaluate.py         # Task A: direction, fixed-window horizon table
  walkforward.py                           # Task A: walk-forward CIs        ← direction headline
  analysis.py                              # Task A: non-overlap · conviction · per-sector cross-checks
  volatility.py                            # Task B: volatility regime on StockNet
  volatility_universe.py                   # Task B pro: best vol forecaster ← volatility headline
app.py                                     # Streamlit demo (pick a ticker + horizon)
docs/METRICS.md                            # plain-language guide to every metric
tests/                                     # leakage · labels · embargo · metric correctness (19 tests)
.github/workflows/ci.yml                   # runs the tests on every push
```

---

## Quickstart

```bash
# Python 3.11, CPU only
py -3.11 -m venv .venv && .venv/Scripts/activate          # Windows
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
git clone https://github.com/yumoxu/stocknet-dataset data/stocknet-dataset

python sentiment_model/train_sentiment_model.py   # fine-tune DistilBERT (~30 min CPU)
python features/build_features.py                 # build StockNet features
python predictor/walkforward.py                   # Task A: direction (honest)
python predictor/volatility_universe.py           # Task B: volatility (the strong result)
streamlit run app.py                              # optional interactive demo
pytest -q                                          # 19 tests
```

---

## Limitations & disclaimer
One benchmark era; **daily** data (intraday would push volatility R² higher); no transaction costs.
Educational project — **not investment advice.** Datasets keep their own licenses (Twitter Financial
News Sentiment: MIT; StockNet: see its repo).

> Yumo Xu and Shay B. Cohen. 2018. *Stock Movement Prediction from Tweets and Historical Prices.* ACL 2018.
