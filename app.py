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
def _models(_df: pd.DataFrame):
    # Trained once and cached for the session. We surface the XGBoost/combined model as the
    # headline predictor in the demo.
    trained = train_all(_df)
    return {m.name: m for m in trained}


st.title("📈 Financial Sentiment & Market Signal")
st.caption(
    "Combines a fine-tuned DistilBERT sentiment factor with technical price factors to "
    "predict next-day stock direction on the StockNet benchmark (Xu & Cohen, ACL 2018). "
    "Educational demo — not investment advice."
)

if not config.COMBINED_FEATURES_PATH.exists():
    st.error(
        "Feature table not found. Run `python features/build_features.py` first to generate "
        f"{config.COMBINED_FEATURES_PATH.name}."
    )
    st.stop()

df = _load()
models = _models(df)

col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Ticker", sorted(df["ticker"].unique()))
with col2:
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
pcol1.metric("Predicted next-day direction", direction)
if proba is not None:
    pcol2.metric("P(up)", f"{proba:.1%}")
actual = "⬆️ UP" if int(row[config.TARGET_COL]) == 1 else "⬇️ DOWN"
pcol3.metric("Actual (realized)", actual)

hit = pred == int(row[config.TARGET_COL])
st.success("Model was correct on this day ✅") if hit else st.info("Model missed on this day ❌")

st.caption(
    f"Split: **{row['split']}**  ·  Features used by `{model_name}`: {', '.join(cols)}. "
    "Remember: on this task even good models land only modestly above 50% — see the README "
    "comparison table."
)
