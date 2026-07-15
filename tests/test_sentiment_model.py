"""Sanity checks on the fine-tuned sentiment model.

These are intentionally lightweight and self-skipping: if the model has not been trained
yet (e.g. in CI, where we never download data or train), the test is skipped rather than
failed. When the model IS present locally, it confirms the scoring pipeline runs and that
obviously-bullish vs obviously-bearish text produce different signed sentiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.SENTIMENT_MODEL_DIR.exists(),
    reason="Fine-tuned sentiment model not present (run train_sentiment_model.py first).",
)


def _pipeline():
    from features.build_features import _load_sentiment_pipeline
    return _load_sentiment_pipeline()


def test_scores_have_expected_shape():
    from features.build_features import score_texts

    tokenizer, model = _pipeline()
    probs = score_texts(tokenizer, model, ["$ aapl to the moon", "market crash incoming"])
    assert probs.shape == (2, config.NUM_LABELS)
    # Each row is a valid probability distribution.
    assert probs.sum(axis=1).round(3).tolist() == [1.0, 1.0]


def test_bullish_text_more_bullish_than_bearish_text():
    from features.build_features import score_texts

    tokenizer, model = _pipeline()
    bull_idx = config.NAME_TO_LABEL["Bullish"]
    bear_idx = config.NAME_TO_LABEL["Bearish"]

    bullish = "$ aapl strong buy huge upside rally breakout new highs"
    bearish = "$ aapl plunges crash sell off huge losses bankruptcy fear"
    probs = score_texts(tokenizer, model, [bullish, bearish])

    bullish_signed = probs[0, bull_idx] - probs[0, bear_idx]
    bearish_signed = probs[1, bull_idx] - probs[1, bear_idx]
    # The bullish sentence should be more bullish than the bearish one.
    assert bullish_signed > bearish_signed
