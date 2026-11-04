"""News semantics layer — what the news channels are saying, mapped onto the radar.

    sources.py    pull headlines from free RSS feeds of financial news channels
    semantics.py  tag each headline: which sectors, which risk themes, how much shock language
    scoring.py    tone (Bearish / Bullish / Neutral) from the fine-tuned DistilBERT classifier,
                  with a transparent lexicon fallback when the model is not installed
    radar.py      aggregate into a market + per-sector news reading, archive every headline,
                  and report how close the archive is to supporting an honest validation

The governing rule of the project still applies: this is a **descriptive overlay, not a
forecast input**. Free feeds carry days of history, not years, so news tone cannot be
walk-forward validated today. The archive exists so that it can be, later, by the same
paired-test rule every other candidate feature faces.

Run:
    python -m volradar.cli news
"""
