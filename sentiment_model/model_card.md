---
license: apache-2.0
language:
  - en
base_model: distilbert-base-uncased
tags:
  - financial-sentiment
  - text-classification
  - distilbert
datasets:
  - zeroshot/twitter-financial-news-sentiment
pipeline_tag: text-classification
---

# DistilBERT Financial-Tweet Sentiment

Fine-tuned `distilbert-base-uncased` that classifies a financial tweet as **Bearish**,
**Bullish**, or **Neutral**. Trained entirely on a laptop CPU as the sentiment factor for a
combined sentiment + price stock-movement model
([GitHub repo](https://github.com/toosh-legacy/quant-nlp-gen)).

## Labels
| id | label |
|----|---------|
| 0  | Bearish |
| 1  | Bullish |
| 2  | Neutral |

## Training data
[`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
— 11,932 finance-related tweets, pre-labeled, MIT licensed. Used with its own splits:
~9,543 train / ~2,388 validation. Tweets are short, informal, and cashtag-heavy
(`$AAPL`), with URLs/mentions replaced by placeholder tokens.

## Training procedure
- Base model: `distilbert-base-uncased` (66M params)
- Max sequence length: 128 tokens
- Batch size: 16, learning rate: 2e-5, epochs: 3
- Optimizer/loss: HuggingFace `Trainer` defaults (AdamW, cross-entropy)
- Hardware: **CPU only** (no GPU)
- Seed: 42

## Evaluation
On the dataset's own validation split:

| metric | value |
|--------|-------|
| accuracy | **0.8807** |
| macro-F1 | **0.8464** |

(From `sentiment_metrics.txt`, 3 epochs on CPU.)

## Intended use & limitations
- **Intended:** research/education — scoring short financial-tweet text for coarse
  bull/bear/neutral tone, e.g. as one factor in a movement-prediction model.
- **Not intended:** trading decisions, non-English text, long-form news articles, or any
  high-stakes use. Sentiment ≠ price direction; on the downstream StockNet task, adding this
  factor moves accuracy only modestly above chance (see the repo's comparison table), which is
  consistent with the published literature.
- **Biases:** reflects the training tweets' era (financial Twitter, ~2020s labeling) and
  domain; cashtag/ticker conventions matter. May misread sarcasm or mixed-signal posts.

## How to use
```python
from transformers import pipeline
clf = pipeline("text-classification", model="<your-hf-username>/distilbert-financial-tweet-sentiment")
clf("$AAPL breaking out to new highs, strong buy")
# -> [{'label': 'Bullish', 'score': ...}]
```

## Citation
Downstream task and dataset:
> Yumo Xu and Shay B. Cohen. 2018. *Stock Movement Prediction from Tweets and Historical
> Prices.* ACL 2018.
