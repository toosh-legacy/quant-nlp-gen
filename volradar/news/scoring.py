"""Headline tone: Bearish / Bullish / Neutral.

Primary scorer is the DistilBERT classifier fine-tuned in `volradar/sentiment/` (88.1%
accuracy, 0.846 macro-F1 on its validation split of financial news tweets — short texts
close in register to headlines). If the optional `transformers` extras or the trained
weights are absent, a small finance lexicon takes over so the overlay still works. The
scorer in use is always reported alongside the reading; the two are never mixed in a run.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from volradar import config

TONE = {"Bearish": -1, "Neutral": 0, "Bullish": 1}

_POS = ["beat", "beats", "gain", "gains", "rally", "rallies", "rise", "rises", "rose",
        "surge", "surges", "soar", "soars", "record high", "upgrade", "upgrades", "strong",
        "stronger", "growth", "boost", "boosts", "optimism", "rebound", "rebounds", "jump",
        "jumps", "tops", "higher", "bullish", "recovery", "raises guidance", "outperform",
        "cools", "eases", "deal"]
_NEG = ["miss", "misses", "fall", "falls", "fell", "drop", "drops", "decline", "declines",
        "downgrade", "downgrades", "weak", "weaker", "loss", "losses", "plunge", "plunges",
        "slump", "slumps", "fear", "fears", "concern", "concerns", "worry", "worries",
        "warns", "layoffs", "recession", "lawsuit", "probe", "tumble", "tumbles", "sink",
        "sinks", "lower", "bearish", "cuts guidance", "selloff", "sell-off", "crash",
        "default", "bankruptcy", "slowdown", "tariffs", "underperform", "rout"]
_POS_RX = re.compile(r"\b(" + "|".join(map(re.escape, _POS)) + r")\b", re.I)
_NEG_RX = re.compile(r"\b(" + "|".join(map(re.escape, _NEG)) + r")\b", re.I)


def _lexicon(titles: list[str]) -> list[tuple[str, float]]:
    out = []
    for t in titles:
        s = len(_POS_RX.findall(t)) - len(_NEG_RX.findall(t))
        label = "Bullish" if s > 0 else "Bearish" if s < 0 else "Neutral"
        out.append((label, min(1.0, 0.5 + 0.2 * abs(s))))
    return out


@lru_cache(maxsize=1)
def _pipeline():
    model_dir = config.SENTIMENT_MODEL_DIR
    if not (model_dir / "model.safetensors").exists():
        return None, "lexicon (no fine-tuned model at data/sentiment_model)"
    try:
        from transformers import pipeline
    except ImportError:
        return None, "lexicon (transformers not installed)"
    clf = pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir),
                   truncation=True, max_length=config.SENT_MAX_LENGTH, device=-1)
    return clf, "distilbert-financial-sentiment"


def _label_name(raw: str) -> str:
    if raw.startswith("LABEL_"):
        return config.LABEL_NAMES.get(int(raw.split("_")[-1]), raw)
    return raw


def score(df: pd.DataFrame, force_lexicon: bool = False) -> tuple[pd.DataFrame, str]:
    """Add `label`, `confidence`, `tone` (-1/0/+1) columns. Returns (frame, scorer name)."""
    if df.empty:
        return df.assign(label=[], confidence=[], tone=[]), "none"
    titles = df["title"].tolist()
    clf, name = (None, "lexicon (forced)") if force_lexicon else _pipeline()
    if clf is None:
        pairs = _lexicon(titles)
    else:
        preds = clf(titles, batch_size=32)
        pairs = [(_label_name(p["label"]), float(p["score"])) for p in preds]
    labels, conf = zip(*pairs)
    out = df.assign(label=list(labels), confidence=list(conf))
    out["tone"] = out["label"].map(TONE).fillna(0).astype(int)
    return out, name
