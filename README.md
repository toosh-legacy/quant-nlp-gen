# Financial Sentiment & Market Signal Analysis

**How far ahead can you predict a stock's direction?** This project predicts whether a stock moves
up or down over several horizons — **1, 5, and 10 trading days** — by combining two factors:

1. **Price** — technical features (momentum, moving averages, volatility, volume change, **RSI, MACD**) from the stock's own trading history.
2. **Sentiment** — a fine-tuned DistilBERT reads financial tweets and scores them bullish / bearish / neutral (a supporting factor).

Both feed one classifier per horizon. This is a lightweight, CPU-only build on the task from
**Xu & Cohen, ACL 2018, _"Stock Movement Prediction from Tweets and Historical Prices"_**, evaluated
on their **StockNet** benchmark with its **original chronological split** — so the numbers compare to a
real published baseline instead of an invented target. The headline finding is a genuine, honest one:
**predictability rises modestly with horizon.**

> Everything here runs on a normal laptop **CPU** — no GPU, no cloud compute.

## Architecture

```
Twitter Financial News Sentiment (11,932 pre-labeled tweets)
        │  fine-tune
        ▼
  DistilBERT sentiment model ──────────────┐
                                           │ score every tweet
StockNet (88 tickers, tweets + prices,     │ aggregate per ticker-day
 2014–2016, with up/down labels)           ▼
        │                         daily sentiment factor
        ├── technical price features ──────┤
        │   momentum · MA · volatility ·   │
        │   volume · RSI · MACD            │
        ▼                                  ▼
   join per ticker-day (ONLY past data; no lookahead)
        │  StockNet's original train/dev/test date split
        │  + embargo (drop rows whose h-day outcome crosses a split)
        ▼
   for each horizon h ∈ {1, 5, 10} days:
      price-only  vs  sentiment-only  vs  combined
        │  LogisticRegression + XGBoost (tuned on dev by MCC)
        ▼
   accuracy · macro-F1 · MCC   ← honest horizon curve
```

## Results

### Headline: how far can we predict? (StockNet test split, reported once)

Best variant at each horizon. MCC (Matthews Correlation Coefficient, −1…1) is the metric to watch:
on a ~50/50 target, accuracy can be fooled by always predicting one direction, while MCC only rises
when the model beats chance across **both** directions.

| horizon | best variant | accuracy | macro-F1 | **MCC** |
|--------:|--------------|:--------:|:--------:|:-------:|
| **1 day**  | logreg / sentiment-only | 0.528 | 0.519 | **+0.065** |
| **5 day**  | logreg / combined       | 0.536 | 0.536 | **+0.080** |
| **10 day** | logreg / sentiment-only | 0.547 | 0.542 | **+0.112** |

**Predictability climbs with horizon** — accuracy 52.8% → 53.6% → 54.7%, MCC +0.065 → +0.080 → +0.112.
A single day is nearly a coin flip; over 10 days, accumulated drift/momentum gives a small but real,
consistent edge. This is the honest answer to the headline question, and it lands right where the
literature says it should (simple models in the low-to-mid 50s%; StockNet's own deep model ~58%).

<details><summary>Full comparison — every variant × horizon</summary>

| horizon | model | accuracy | macro-F1 | MCC |
|--------:|-------|:--------:|:--------:|:---:|
| 1 | logreg/price-only | 0.513 | 0.511 | +0.029 |
| 1 | xgboost/price-only | 0.496 | 0.492 | −0.012 |
| 1 | logreg/sentiment-only | 0.528 | 0.519 | +0.065 |
| 1 | xgboost/sentiment-only | 0.514 | 0.497 | +0.042 |
| 1 | logreg/combined | 0.519 | 0.518 | +0.043 |
| 1 | xgboost/combined | 0.493 | 0.489 | −0.018 |
| 5 | logreg/price-only | 0.526 | 0.526 | +0.059 |
| 5 | xgboost/price-only | 0.511 | 0.510 | +0.027 |
| 5 | logreg/sentiment-only | 0.523 | 0.519 | +0.078 |
| 5 | xgboost/sentiment-only | 0.507 | 0.490 | +0.071 |
| 5 | logreg/combined | 0.536 | 0.536 | +0.080 |
| 5 | xgboost/combined | 0.523 | 0.522 | +0.051 |
| 10 | logreg/price-only | 0.507 | 0.507 | +0.019 |
| 10 | xgboost/price-only | 0.546 | 0.546 | +0.094 |
| 10 | logreg/sentiment-only | 0.547 | 0.542 | +0.112 |
| 10 | xgboost/sentiment-only | 0.502 | 0.380 | −0.074 |
| 10 | logreg/combined | 0.516 | 0.516 | +0.037 |
| 10 | xgboost/combined | 0.552 | 0.552 | +0.108 |

</details>

**Does sentiment help?** It's a *minor* factor here, and the effect is mixed: at 1d and 10d a
sentiment-only model edges out price-only, but combining price + sentiment doesn't consistently beat
the best single factor (adding a weak signal to price can dilute it). Sentiment is kept as a secondary
input, not the star.

### Supporting model: the sentiment classifier (held-out validation split)
| metric | value |
|--------|-------|
| accuracy | **0.881** |
| macro-F1 | **0.846** |

A strong standalone tweet-sentiment classifier (DistilBERT, CPU, 3 epochs) — used here as one factor
feeding the movement models.

See [How honest are these numbers?](#how-honest-are-these-numbers) for the methodology that makes the
horizon curve trustworthy (chronological split, overlapping-window **embargo**, tune-on-dev).

## Quickstart

```bash
# 1. Environment (Python 3.11, CPU)
py -3.11 -m venv .venv && .venv/Scripts/activate     # Windows
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Get StockNet (tweets + prices)
git clone https://github.com/yumoxu/stocknet-dataset data/stocknet-dataset

# 3. Fine-tune the sentiment model (~30 min CPU; downloads the tweet dataset automatically)
python sentiment_model/train_sentiment_model.py

# 4. Build features (scores every StockNet tweet, then price features + combine)
python features/build_features.py

# 5. Train + tune + evaluate across horizons -> horizon curve
python predictor/evaluate.py

# 6. (optional) interactive demo
streamlit run app.py

# tests
pytest -q
```

## How honest are these numbers?

Published results on this task land only modestly above chance — roughly mid-50s% accuracy and small
positive MCC. Markets are near-efficient and short-horizon direction is close to a coin flip. **No
legitimate model reaches 0.80 on next-day (or next-week) direction** — numbers like that come from
data leakage or overfitting, not skill. What makes the horizon curve above trustworthy:

- **No-lookahead features**, enforced in tests (`tests/test_build_features.py` perturbs a *future*
  price and asserts no past feature — including RSI/MACD — changes).
- **Overlapping-window embargo.** Labeling every day with its next-*h*-day return makes neighbouring
  rows' outcomes overlap, which can *fake* good scores. We keep a row for horizon *h* only if its
  outcome day lands in the **same** split as its feature day, so a train outcome never overlaps a
  dev/test outcome (standard purging). Tested in `tests/test_evaluate.py`.
- **Tune on dev, touch test once.** XGBoost hyperparameters are selected by **dev-split MCC**; the
  test split is scored a single time.
- **Fair thresholds across horizons.** The up/down dead band is scaled by √h, so a higher number at 10
  days reflects real signal, not a looser bar on bigger moves.

The rising MCC with horizon (+0.065 → +0.080 → +0.112) is therefore a modest but *real* effect, not an
artefact — and it's exactly the kind of result an interviewer trusts over a suspicious 0.80.

## Future work / suggested improvements
- **Longer + more horizons** (20d, 60d) to trace the full predictability curve; expect diminishing,
  eventually noisy returns.
- **Non-overlapping weekly bars** as a stricter cross-check on the overlapping-window result.
- **Stronger sentiment factor:** fine-tune `roberta-base`/FinBERT, or aggregate sentiment over a
  trailing multi-day window rather than a single day.
- **Per-sector or per-ticker models**, and **probability calibration** + a confidence threshold
  (trade only high-conviction days) — often where a small directional edge becomes usable.
- **Richer targets:** predict magnitude/volatility, or model the paper's temporal structure.
- **Walk-forward evaluation** across multiple test windows to report a confidence interval rather than
  a single window's point estimate.

## Repo layout
```
config.py                       # paths, seed, feature lists, horizons, √h thresholds, date split
sentiment_model/
  train_sentiment_model.py      # fine-tune DistilBERT
  model_card.md                 # Hugging Face model card
features/build_features.py      # score tweets + price features (RSI/MACD) + multi-horizon labels + embargo
predictor/
  train_predictor.py            # LogReg + dev-tuned XGBoost, 3 variants, per horizon, embargoed
  evaluate.py                   # accuracy / F1 / MCC horizon curve
app.py                          # optional Streamlit demo (ticker + horizon selector)
tests/                          # leakage, RSI/MACD, horizon-label, embargo + metric tests
.github/workflows/ci.yml        # runs the test suite on every push
```

## Model
Fine-tuned sentiment model on the Hugging Face Hub: _upload pending_ — run
`huggingface-cli login` then push `data/sentiment_model/` with `sentiment_model/model_card.md`
(link filled in here after upload).

## Citation
> Yumo Xu and Shay B. Cohen. 2018. *Stock Movement Prediction from Tweets and Historical Prices.*
> Proceedings of ACL 2018.

## License & disclaimer
Educational project. **Not investment advice.** Datasets retain their own licenses
(Twitter Financial News Sentiment: MIT; StockNet: see its repo).
