# Financial Sentiment & Market Signal Analysis

## What this is
This project asks a simple question: if you combine what people are saying about a stock with how that stock has actually been trading, can you predict whether it goes up or down the next day — even a little better than chance? It's not text analysis alone (just reading tweets) and it's not price analysis alone (just charting momentum) — it's both, combined into one model, which is what "quantitative" actually means in finance: multiple factors, weighed together.

Two pieces feed into one prediction:
1. A **sentiment factor** — a small language model reads financial tweets and scores how bullish or bearish they sound
2. A **price factor** — standard technical features (momentum, moving averages, volatility) computed from the stock's own trading history

Both get combined into one classifier that predicts next-day stock movement, and the whole point of the evaluation is being honest about how much the sentiment factor actually adds on top of price alone — which, per the real academic literature on this exact task, is usually modest. That honesty is the strongest part of the project, not a weakness.

Everything here runs on a normal laptop CPU. No GPU, no Colab, no cloud compute account needed anywhere in this build.

## No manual labeling — this uses real, already-labeled datasets
Two public, freely available, pre-labeled datasets do all the labeling work:

1. **Twitter Financial News Sentiment** (`zeroshot/twitter-financial-news-sentiment` on Hugging Face) — 11,932 finance-related tweets, each already labeled Bullish, Bearish, or Neutral. MIT licensed. This is what teaches the sentiment model what bullish/bearish language sounds like.
2. **StockNet** (Xu & Cohen, ACL 2018 — `yumoxu/stocknet-dataset` on GitHub, also mirrored on Hugging Face) — a well-known academic benchmark pairing two years (2014–2016) of tweets with historical daily price data for 88 stock tickers across 9 industries, and it already includes the up/down movement label for each trading day. This is the actual academic dataset the "combine text and price history" task comes from — you're not inventing the task, you're building a lightweight, CPU-friendly version of a real published benchmark, which means you have a paper to cite when someone asks "why this approach?"

Nothing in this project requires you to sit down and hand-label anything. The only labels used anywhere are the ones that came with these two datasets.

## Scope — decided, don't re-litigate this
- **Compute:** CPU only, on a normal laptop. No GPU, no Google Colab, no Kaggle notebooks, no cloud compute account.
- **Sentiment model:** fine-tune `distilbert-base-uncased` (66 million parameters — small enough to fine-tune on a CPU in well under an hour on the 11,932-tweet dataset above). If even that feels slow on your machine, the fallback is to skip fine-tuning entirely and use `ProsusAI/finbert`, an already-fine-tuned financial sentiment model, directly for inference — note in the learning guide which path was taken.
- **Combined predictor:** Logistic Regression as the simple baseline, XGBoost as the main model — both train in seconds to minutes on a CPU, no exceptions needed.
- **Data universe:** the StockNet dataset's own 88 tickers and its own 2014–2016 date range. Use the **train/dev/test date split that comes with the original StockNet paper** rather than inventing your own — this keeps the split chronologically correct (no lookahead bias) and makes your results directly comparable to a real published baseline.
- **Where it ends up:** the fine-tuned sentiment model gets pushed to the **Hugging Face Hub** with a model card. The full combined system — feature building, the predictor, evaluation, and an optional demo — gets published as a clean **GitHub repo**.

## Why this is a real project, not just a resume line
The task itself — predicting stock movement from tweets plus historical prices — is the exact task from Xu & Cohen's 2018 ACL paper "Stock Movement Prediction from Tweets and Historical Prices," which introduced the StockNet dataset this project uses. Published results on this task tend to land only modestly above chance (roughly mid-50s% accuracy, small positive correlation scores) — which sets an honest bar. If your combined model also lands in that range, that's a legitimate, citable result, not a disappointing one. If it doesn't, that's worth investigating and writing up too. Either way, you have real published numbers to compare against instead of an invented target.

## How the pieces fit together

```
Twitter Financial News Sentiment dataset (pre-labeled, 11,932 tweets)
                  │
                  ▼
     fine-tune DistilBERT to classify bullish / bearish / neutral
                  │
                  ▼
        [ this becomes your sentiment-scoring model ]
                  │
                  ▼
StockNet dataset (tweets + daily prices, 88 tickers, already has
      the up/down movement label for each trading day)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  run each day's tweets   compute price-based features from
  for each ticker through  the included daily price history:
  the sentiment model,     momentum, moving averages,
  aggregate into a daily   volatility, volume change
  sentiment score
        │                   │
        └─────────┬─────────┘
                  ▼
      combine both feature sets into one row per
      ticker-day, using only information available
      up to that day (never anything from the future)
                  │
                  ▼
      train three versions of the predictor and compare:
      price-only, sentiment-only, and combined —
      using the dataset's original chronological
      train/dev/test split
                  │
                  ▼
      report accuracy, F1, and MCC — short for Matthews
      Correlation Coefficient, a single score from -1 to 1
      that the original paper uses because it stays honest
      even when the up/down labels aren't perfectly balanced —
      for all three model variants, honestly
```

## Tools, and why each one is there
| Piece | Tool | Why this one |
|---|---|---|
| Sentiment fine-tuning | `distilbert-base-uncased` via Hugging Face `transformers` | 66M parameters — genuinely fine-tunable on a CPU in a reasonable amount of time, unlike a multi-billion-parameter model |
| Sentiment fallback | `ProsusAI/finbert` | Already fine-tuned for financial sentiment — a zero-training option if fine-tuning DistilBERT is still too slow on your machine |
| Combined predictor (baseline) | Logistic Regression (`scikit-learn`) | Trains instantly on a CPU, and it's the right first thing to try before anything fancier |
| Combined predictor (main) | XGBoost | Still CPU-native and fast, and it's what most real quant/finance teams actually reach for on tabular, multi-factor data like this |
| Price features | plain `pandas` on the price data already included in StockNet | Momentum, moving averages, volatility, volume change — no exotic feature engineering needed |
| Storage | CSV / Parquet files, no database | At this data size (88 tickers, ~2 years), a database is unnecessary overhead — flat files are simpler and just as fast |
| Demo (optional) | Streamlit | One file, lets you type in a ticker and a date and see the model's factors and prediction |
| Keeping tests running | GitHub Actions | One workflow file, runs the test suite on every push — no deployment step needed since this isn't a live service |

## Environment & setup — so nothing is left to guess
- **Python 3.11**, one virtual environment for the whole project.
- **Core dependencies:** `transformers`, `datasets`, `torch` (CPU build), `scikit-learn`, `xgboost`, `pandas`, `numpy`. Optional: `streamlit` if the demo gets built. Pin versions in a `requirements.txt` the first time the environment is set up, so the build is reproducible later.
- **Sentiment model training defaults** (a reasonable starting point, not a requirement to hit exactly): batch size 16, 2–3 epochs, learning rate 2e-5, max sequence length 128 tokens (tweets are short). On a modern laptop CPU this should train in well under an hour on the ~9,500-row training split; if it's taking much longer than that, cut to 1 epoch or reduce the training set size before abandoning the CPU-only approach.
- **Use the dataset's own split.** The Twitter Financial News Sentiment dataset already ships with a train split (~9,540 rows) and a validation split (~2,390 rows) on Hugging Face — use those directly as train/test rather than making a new split.

## Getting the StockNet dataset
StockNet is distributed as a GitHub repository (`yumoxu/stocknet-dataset`), not a single downloadable file — `git clone` it directly. It contains historical price data and tweets organized by ticker, already split into the price/tweet pairs the original paper used, along with the movement labels. **Read the dataset repo's own README once it's cloned** before writing the loading code — dataset repo layouts occasionally get reorganized after a paper is published, so treat this spec's description of "tweets and prices per ticker" as the shape of the data, and confirm the exact file/folder names against what's actually there rather than assuming. If the GitHub version is awkward to parse, check Hugging Face first (search "stocknet") for a pre-converted Parquet/CSV mirror, which is usually far less work to load.

## Repo layout — kept deliberately small
```
financial-sentiment-market-signal/
├── requirements.txt
├── .gitignore                     # see the Gitignore section below
├── .github/workflows/ci.yml       # runs the test suite on every push — this is what makes "testing" actually count
├── config.py                      # file paths, random seed, feature column list — decided once
├── data/                          # downloaded datasets live here, not committed
├── sentiment_model/
│   ├── train_sentiment_model.py   # fine-tunes DistilBERT on the labeled tweet dataset
│   └── model_card.md              # goes to Hugging Face with the model
├── features/
│   └── build_features.py          # sentiment scores + price features, combined per ticker-day
├── predictor/
│   ├── train_predictor.py         # logistic regression + XGBoost, all three variants
│   └── evaluate.py                # accuracy / F1 / MCC table, price-only vs sentiment-only vs combined
├── app.py                         # optional Streamlit demo
├── tests/
│   ├── test_sentiment_model.py    # sanity checks on the fine-tuned model's predictions
│   ├── test_build_features.py     # confirms no future data ever leaks into a feature row
│   └── test_evaluate.py           # confirms accuracy/F1/MCC are computed correctly
├── LEARNING_GUIDE.md               # personal notes, gitignored — see below
└── README.md
```
That's the whole thing. No microservices, no orchestration layer, no message queue — this is a data science project, not a distributed system, and it should look like one.

## Building it, one piece at a time
1. **Download both datasets and look at them before writing any code.** Get a feel for the tweet sentiment dataset's label balance, and for what a row of StockNet's price data actually looks like.
2. **Fine-tune the sentiment model.** Train `distilbert-base-uncased` on the labeled tweet dataset to classify bullish/bearish/neutral. Hold out a portion of that dataset as a test set and report accuracy/F1 on it before moving on — this is a complete, standalone result on its own.
3. **Score the StockNet tweets.** Run every tweet in the StockNet dataset through the fine-tuned sentiment model, then aggregate to one sentiment score per ticker per day (e.g. average score, tweet count, ratio of bullish to bearish).
4. **Build the price features.** From the daily price data already in StockNet, compute a small set of standard technical features per ticker-day: recent return/momentum, a moving average or two, rolling volatility, volume change. Nothing exotic — a handful of well-understood features beats twenty confusing ones.
5. **Combine them, respecting time.** Join the sentiment features and price features into one row per ticker-day, using only data available up through that day — never anything from a later date. This is the same lookahead-bias discipline that matters in any real financial ML project, and it's worth a code comment explaining it.
6. **Use the dataset's own date split.** Train on the original StockNet train period, tune on the dev period, and only touch the test period once, at the very end, to report final numbers.
7. **Train and compare three models.** Price-only, sentiment-only, and combined — using both Logistic Regression and XGBoost. Report accuracy, F1, and MCC for all of them side by side.
8. **Set up CI.** A minimal GitHub Actions workflow that installs dependencies and runs the test suite on every push — a data project without automated tests running is easy to quietly break; this is what keeps it honest going forward.
9. **Write it up honestly.** State plainly whether the combined model actually beat price-only, by how much, and whether that's consistent with what the original StockNet paper found.

## What "done" looks like
- **A sentiment model results section**: accuracy/F1 on the held-out portion of the tweet sentiment dataset.
- **A comparison table**: price-only vs. sentiment-only vs. combined, each with accuracy, F1, and MCC, on the dataset's proper held-out test period.
- **A model card**, published with the Hugging Face upload — what the sentiment model was trained on, how, and its limitations.
- **An honest discussion section** comparing your combined-model result to what's reported in the original StockNet paper, and explaining, in your own words, why a small improvement (or no improvement) from adding sentiment is a completely normal, expected outcome in this literature — not a failure of the project.
- **A limitations section** — what you'd try with more data, more compute, or more time.

## Publishing it
**Hugging Face (the model):** push the fine-tuned DistilBERT sentiment classifier to a public Hugging Face model repo with a real model card — what it does, what it was trained on, how to load it, its limitations, and a link back to the GitHub repo. This step needs a free Hugging Face account and an access token (`huggingface-cli login`, or a `HUGGINGFACE_TOKEN` environment variable) — keep that token out of the repo the same way any other credential would be.

**GitHub (the system):** the README should read like a short, clear explanation — what the project does, why the task is a real published benchmark and not an invented one, an architecture diagram, the comparison table, a quickstart, and a link to the live Hugging Face model. Keep the jargon light in the README itself.

## Resume bullet — fill this in only once you've actually measured the numbers
```
Built a combined sentiment + price-history model to predict daily stock
movement, fine-tuning DistilBERT on a labeled financial-tweet sentiment
dataset and combining its output with technical price features (momentum,
volatility, moving averages) in an XGBoost classifier; evaluated on the
StockNet benchmark (Xu & Cohen, ACL 2018) using its original chronological
train/test split; the combined model reached [X]% accuracy and an MCC of
[Y], [an improvement of Z points over / roughly in line with] a
price-only baseline, consistent with published results on this task.
Fine-tuned model deployed on Hugging Face; full pipeline on GitHub.
```
Only fill in bracketed values with numbers you personally computed and can explain.

## If there's time left over (optional, do this last)
- Add a few more technical indicators (RSI, MACD) to the price-feature set
- Try the larger `bert-base-uncased` instead of DistilBERT for the sentiment model, if CPU time allows
- Extend the sentiment model to score financial news headlines in addition to tweets
- Build the optional Streamlit demo if you skipped it during the core build

---

## How to build this
The instructions below apply to whoever (or whatever) is building this project — including Claude Code, if that's what's driving the build.

**Keep it simple — top priority.** This is scoped so one undergrad, on a laptop CPU, can build and fully explain every part of it. Don't add complexity that isn't described above — no databases, no orchestration tools, no services beyond what's listed. Choose the simple, well-understood approach over the sophisticated one every time there's a choice. Everything in "If there's time left over" is optional and out of scope until the core build is fully done and tested.

**Work autonomously.** Build straight through the numbered build steps in order, without pausing between them to ask for direction. The Scope section exists to remove ambiguity — use those decisions instead of asking. Only stop and ask if something is genuinely unresolvable — a real contradiction, or a hard blocker with no free alternative (there shouldn't be any, since everything here is free and CPU-only). Otherwise make the reasonable call, note it and the reasoning in `LEARNING_GUIDE.md`, and keep going.

**Verify before moving on.** After each numbered step, run it, run its tests, and confirm it actually does what it's supposed to before starting the next one.

**Annotate as you go.** Every non-trivial file should have comments explaining *why*, not just what — especially the parts of the feature-building step that guard against using future information.

**Write real tests as you build**, not bolted on at the end. Tests here don't need to be complicated — checking that the feature-building step never includes a future date's data, and that the evaluation script computes accuracy/F1/MCC correctly, covers the most important risks.

**Keep a learning guide.** Maintain `LEARNING_GUIDE.md` at the repo root — a personal study reference, not for public consumption, excluded from git (see Gitignore below). After each build step, append: what was built, in plain language; any new concept introduced, explained from first principles (e.g. what MCC measures and why it's used instead of plain accuracy, what fine-tuning a small transformer actually changes versus using it zero-shot); why this approach was chosen over the obvious alternative; two or three interview talking points; and pitfalls this implementation avoids. Add a final summary section once the whole project is done.

**Commit discipline.** Commit at the end of each meaningful sub-step. Write commit messages that describe *why* (e.g. `feat: use StockNet's original date split to keep results comparable to published baselines`). Never commit `LEARNING_GUIDE.md`, `.env` files, or credentials, and never commit the raw downloaded datasets themselves (see Gitignore).

**Definition of done.** The project isn't finished until the comparison table exists and is honestly reported, including the discussion of how it compares to the original paper — that comparison is what makes this credible in an interview rather than just a demo.

### Gitignore
Create a `.gitignore` in the repo root with at least the following:
```
LEARNING_GUIDE.md

__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

.env
.env.*
!.env.example
*.pem
credentials.json

.DS_Store
Thumbs.db
.vscode/
.idea/

data/
*.parquet
*.csv
*.pt
*.bin
```
(Project 1 doesn't call any paid API, so a `.env`/`.env.example` pair isn't strictly required — but keep the pattern above for consistency if one gets added later, e.g. for a Hugging Face upload token.)
