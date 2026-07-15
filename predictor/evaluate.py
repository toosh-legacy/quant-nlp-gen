"""Step 6 (evaluation half) — the honest comparison table.

Trains on train, selects on dev, and touches the test split exactly once to report the
final numbers for all three variants x two model families:

    accuracy | macro-F1 | MCC

Why MCC (Matthews Correlation Coefficient)? On a near-50/50 up/down task, plain accuracy
can look fine while the model just predicts the majority direction. MCC is a single number
in [-1, 1] that only moves away from 0 when the model is right across BOTH classes in a way
that beats chance, so it stays honest under mild imbalance. It is the metric the original
StockNet paper leans on, which also makes our numbers directly comparable to theirs.

Run:
    python predictor/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from predictor.train_predictor import (  # noqa: E402
    TrainedModel, load_combined, predict, split_frame, train_all,
)


def score(y_true, y_pred) -> dict[str, float]:
    """accuracy / macro-F1 / MCC for one set of predictions."""
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate_models(df: pd.DataFrame, models: list[TrainedModel], split: str) -> pd.DataFrame:
    """Evaluate every trained model on the requested split and return a tidy table."""
    part = split_frame(df, split)
    y_true = part[config.TARGET_COL].to_numpy()
    rows = []
    for m in models:
        metrics = score(y_true, predict(m, part))
        rows.append({"model": m.name, "family": m.family, "features": m.feature_set, **metrics})
    return pd.DataFrame(rows)


def _format_table(table: pd.DataFrame) -> str:
    show = table[["model", "accuracy", "f1_macro", "mcc"]].copy()
    for c in ("accuracy", "f1_macro", "mcc"):
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    return show.to_string(index=False)


def main() -> None:
    df = load_combined()
    counts = df.groupby("split")[config.TARGET_COL].agg(["count", "mean"])
    print("Split sizes and up-rate (mean of target):")
    print(counts.to_string(), "\n")

    print("Training all variants on the train split...")
    models = train_all(df)

    # Dev is where you'd tune; we report it for transparency but do not select on test.
    dev_table = evaluate_models(df, models, "dev")
    print("\n=== DEV split ===")
    print(_format_table(dev_table))

    # Test: touched once, at the very end.
    test_table = evaluate_models(df, models, "test")
    print("\n=== TEST split (final, reported once) ===")
    print(_format_table(test_table))

    # Persist the test table for the README/write-up.
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_table.to_csv(config.RESULTS_PATH, index=False)
    print(f"\nWrote comparison table -> {config.RESULTS_PATH}")

    # Headline: did adding sentiment help, on the metric the paper cares about?
    for family in ("logreg", "xgboost"):
        fam = test_table[test_table["family"] == family].set_index("features")
        if {"price_only", "combined"}.issubset(fam.index):
            delta = fam.loc["combined", "mcc"] - fam.loc["price_only", "mcc"]
            print(f"  [{family}] combined MCC - price-only MCC = {delta:+.4f}")


if __name__ == "__main__":
    main()
