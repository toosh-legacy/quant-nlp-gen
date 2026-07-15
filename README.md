# Financial Sentiment & Market Signal Analysis

Predict **next-day stock direction** by combining two factors:

1. **Sentiment** — a fine-tuned DistilBERT reads financial tweets and scores them bullish / bearish / neutral.
2. **Price** — standard technical features (momentum, moving averages, volatility, volume change) from the stock's own trading history.

Both feed one classifier. The whole point is an **honest** measurement of how much the sentiment
factor actually adds on top of price alone — which, on this task, is modest by design. This is a
lightweight, CPU-only reimplementation of the task from **Xu & Cohen, ACL 2018,
_"Stock Movement Prediction from Tweets and Historical Prices"_**, evaluated on their
**StockNet** benchmark using its **original chronological split**, so the numbers are comparable to a
real published baseline instead of an invented target.

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
        │   (momentum, MA, volatility)     │
        ▼                                  ▼
   join per ticker-day (ONLY past data; no lookahead)
        │  StockNet's original train/dev/test date split
        ▼
   price-only  vs  sentiment-only  vs  combined
        │  Logistic Regression + XGBoost
        ▼
   accuracy · macro-F1 · MCC   ← honest comparison table
```

## Results

### Sentiment model (held-out validation split)
| metric | value |
|--------|-------|
| accuracy | **0.881** |
| macro-F1 | **0.846** |

Trained on CPU, 3 epochs. A strong standalone tweet-sentiment classifier.

### Movement prediction — StockNet test split (reported once)
_Accuracy / macro-F1 / MCC. MCC (Matthews Correlation Coefficient, −1…1) is the headline metric:
on a ~50/50 up/down task it stays honest where accuracy can be fooled by always predicting one direction.
Test split up-rate: 51.3%._

| model | accuracy | macro-F1 | MCC |
|-------|----------|----------|-----|
| logreg / price-only | 0.482 | 0.474 | −0.042 |
| logreg / sentiment-only | 0.499 | 0.394 | −0.037 |
| logreg / combined | 0.503 | 0.497 | +0.001 |
| xgboost / price-only | 0.482 | 0.482 | −0.037 |
| **xgboost / sentiment-only** | **0.521** | **0.509** | **+0.053** |
| xgboost / combined | 0.478 | 0.477 | −0.045 |

**Discussion.** On this particular test window (Oct–Dec 2015) the **price-only** models land
right around — even slightly below — chance (MCC ≈ −0.04). The single best variant is
**sentiment-only XGBoost** (accuracy 52.1%, **MCC +0.053**), and *combining* the two factors did
**not** reliably beat the best single factor: logreg/combined nudged MCC to ~0, while
xgboost/combined actually did worse than either input alone. In other words, the sentiment factor
carried what little signal there was, and naively concatenating it with near-noise price features
diluted rather than helped — a textbook reminder that "more features" ≠ "better model." All of this
sits squarely in the range the original StockNet paper primes you to expect (simple models hover
around 50–55% accuracy with small MCC); the paper's own deep generative model reaches ~58% / MCC
~0.08 with far more machinery. See [How honest are these numbers?](#how-honest-are-these-numbers).

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

# 5. Train + evaluate the three variants -> comparison table
python predictor/evaluate.py

# 6. (optional) interactive demo
streamlit run app.py

# tests
pytest -q
```

## How honest are these numbers?

Published results on this exact task land only modestly above chance — roughly mid-50s%
accuracy and small positive MCC. That's the nature of the problem: markets are near-efficient and
daily direction is close to a coin flip. So:

- A **small** improvement (or none) from adding sentiment is the **expected, normal** outcome here —
  not a failed project. The value is in measuring it honestly on a proper chronological split.
- The **no-lookahead discipline** is enforced in code and tests (`tests/test_build_features.py`
  perturbs a *future* price and asserts no past feature changes). Without that guard, it's easy to
  post impressive-but-fake backtest numbers.

## Repo layout
```
config.py                       # paths, seed, feature lists, the paper's date split
sentiment_model/
  train_sentiment_model.py      # fine-tune DistilBERT
  model_card.md                 # Hugging Face model card
features/build_features.py      # score tweets + price features + combine (no lookahead)
predictor/
  train_predictor.py            # LogReg + XGBoost, three feature variants
  evaluate.py                   # accuracy / F1 / MCC comparison table
app.py                          # optional Streamlit demo
tests/                          # leakage + metric correctness tests
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
