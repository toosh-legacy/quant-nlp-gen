"""Step 7 — optional Streamlit demo.

Pick a ticker and a date; see the price factors and the sentiment factor the model uses,
its up/down prediction with confidence, and (since these are historical dates) what
actually happened next. It's a window into the model's inputs, not a live trading tool.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from predictor.train_predictor import FEATURE_SETS, load_combined, train_all  # noqa: E402

st.set_page_config(page_title="Financial Sentiment & Market Signal", page_icon="📈")


@st.cache_data(show_spinner="Loading feature table...")
def _load() -> pd.DataFrame:
    return load_combined()


@st.cache_resource(show_spinner="Training predictors...")
def _models(_df: pd.DataFrame, horizon: int):
    # Trained once per horizon and cached for the session. XGBoost is tuned on the dev split
    # inside train_all.
    trained, _frame = train_all(_df, horizon)
    return {m.name: m for m in trained}


st.title("📈 Financial Sentiment & Market Signal")
st.caption(
    "Predicts stock-price direction over a chosen horizon on the StockNet benchmark "
    "(Xu & Cohen, ACL 2018), combining technical price factors with a fine-tuned DistilBERT "
    "sentiment factor. Educational demo — not investment advice."
)

if not config.COMBINED_FEATURES_PATH.exists():
    st.error(
        "Feature table not found. Run `python features/build_features.py` first to generate "
        f"{config.COMBINED_FEATURES_PATH.name}."
    )
    st.stop()

df = _load()

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.selectbox("Ticker", sorted(df["ticker"].unique()))
with col2:
    horizon = st.selectbox(
        "Horizon (trading days ahead)", config.HORIZONS,
        index=config.HORIZONS.index(config.DEFAULT_HORIZON),
    )

models = _models(df, horizon)
mv_col = config.movement_col(horizon)

with col3:
    model_name = st.selectbox(
        "Model", list(models.keys()),
        index=list(models.keys()).index("xgboost/combined") if "xgboost/combined" in models else 0,
    )

ticker_df = df[df["ticker"] == ticker].sort_values("date")
if ticker_df.empty:
    st.warning("No usable rows for this ticker.")
    st.stop()

date = st.select_slider(
    "Date",
    options=list(ticker_df["date"].dt.strftime("%Y-%m-%d")),
    value=ticker_df["date"].dt.strftime("%Y-%m-%d").iloc[-1],
)
row = ticker_df[ticker_df["date"].dt.strftime("%Y-%m-%d") == date].iloc[0]

# --- Factors ---
st.subheader("Factors used for this ticker-day")
fcol1, fcol2 = st.columns(2)
with fcol1:
    st.markdown("**Price (technical) factors**")
    st.dataframe(
        pd.DataFrame({"value": [round(float(row[c]), 4) for c in config.PRICE_FEATURE_COLS]},
                     index=config.PRICE_FEATURE_COLS)
    )
with fcol2:
    st.markdown("**Sentiment factors**")
    st.dataframe(
        pd.DataFrame({"value": [round(float(row[c]), 4) for c in config.SENTIMENT_FEATURE_COLS]},
                     index=config.SENTIMENT_FEATURE_COLS)
    )

# --- Prediction ---
model = models[model_name]
cols = model.columns
X = row[cols].to_numpy(dtype=float).reshape(1, -1)
if model.scaler is not None:
    X = model.scaler.transform(X)
pred = int(model.estimator.predict(X)[0])
try:
    proba = float(model.estimator.predict_proba(X)[0, 1])
except Exception:
    proba = None

st.subheader("Prediction")
direction = "⬆️ UP" if pred == 1 else "⬇️ DOWN"
pcol1, pcol2, pcol3 = st.columns(3)
pcol1.metric(f"Predicted direction ({horizon}d ahead)", direction)
if proba is not None:
    pcol2.metric("P(up)", f"{proba:.1%}")

# The realized label can be NaN when the move fell inside the ambiguous dead band (those
# rows are excluded from training/scoring) or when the horizon runs past the data end.
actual_val = row[mv_col]
if pd.isna(actual_val):
    pcol3.metric("Actual (realized)", "— (ambiguous / n.a.)")
    st.info("This day's realized move fell in the neutral dead band, so it's excluded from the scored set.")
else:
    actual_int = int(actual_val)
    pcol3.metric("Actual (realized)", "⬆️ UP" if actual_int == 1 else "⬇️ DOWN")
    hit = pred == actual_int
    st.success("Model was correct on this day ✅") if hit else st.info("Model missed on this day ❌")

st.caption(
    f"Split: **{row['split']}**  ·  Features used by `{model_name}`: {', '.join(cols)}. "
    "Remember: even the best variant lands only modestly above 50% — predictability rises a "
    "little with horizon. See the README horizon table."
)
