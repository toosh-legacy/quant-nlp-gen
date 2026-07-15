"""The honest comparison — now across prediction horizons: "how far ahead can we predict?"

For each horizon h in config.HORIZONS (1, 5, 10 trading days) we train on train, tune
XGBoost on dev, and touch the test split exactly once to report accuracy / macro-F1 / MCC
for all three feature variants x two model families. Then a compact horizon-summary shows
the best MCC at each horizon, which is the answer to the headline question.

Why MCC? On a ~50/50 up/down target, plain accuracy can look fine while the model just
predicts the majority direction. MCC (Matthews Correlation Coefficient, -1..1) only moves
away from 0 when the model beats chance across BOTH classes, so it stays honest under mild
imbalance and is comparable to the StockNet paper.

Embargo note: each horizon's rows are filtered so a training example's h-day outcome window
never overlaps a dev/test example's — see predictor.train_predictor.embargoed_frame.

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


def evaluate_models(test: pd.DataFrame, models: list[TrainedModel]) -> pd.DataFrame:
    """Evaluate every trained model on the (embargoed) test split for its horizon."""
    y_true = test["y"].to_numpy()
    rows = []
    for m in models:
        metrics = score(y_true, predict(m, test))
        rows.append({
            "horizon": m.horizon, "model": m.name, "family": m.family,
            "features": m.feature_set, **metrics,
        })
    return pd.DataFrame(rows)


def _format_table(table: pd.DataFrame) -> str:
    show = table[["horizon", "model", "accuracy", "f1_macro", "mcc"]].copy()
    for c in ("accuracy", "f1_macro", "mcc"):
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    return show.to_string(index=False)


def main() -> None:
    df = load_combined()

    all_tables = []
    for h in config.HORIZONS:
        print(f"\n### Horizon = {h} trading day(s) — training + tuning on train/dev...")
        models, frame = train_all(df, h)
        test = split_frame(frame, "test")
        up_rate = test["y"].mean()
        print(f"  test rows (embargoed): {len(test)}   up-rate: {up_rate:.3f}")
        all_tables.append(evaluate_models(test, models))

    results = pd.concat(all_tables, ignore_index=True)

    print("\n=== TEST split — full comparison (reported once) ===")
    print(_format_table(results))

    # Horizon summary: the best MCC achieved at each horizon, and by which model. This is the
    # direct answer to "how far can we predict?"
    print("\n=== Horizon summary — best MCC per horizon ===")
    summary_rows = []
    for h in config.HORIZONS:
        sub = results[results["horizon"] == h]
        best = sub.loc[sub["mcc"].idxmax()]
        summary_rows.append({
            "horizon": h, "best_model": best["model"],
            "accuracy": best["accuracy"], "f1_macro": best["f1_macro"], "mcc": best["mcc"],
        })
    summary = pd.DataFrame(summary_rows)
    show = summary.copy()
    for c in ("accuracy", "f1_macro", "mcc"):
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    print(show.to_string(index=False))

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.RESULTS_PATH, index=False)
    print(f"\nWrote full comparison table -> {config.RESULTS_PATH}")

    # Does adding sentiment help the combined model, per horizon (on MCC)?
    for h in config.HORIZONS:
        sub = results[(results["horizon"] == h) & (results["family"] == "xgboost")]
        fam = sub.set_index("features")
        if {"price_only", "combined"}.issubset(fam.index):
            delta = fam.loc["combined", "mcc"] - fam.loc["price_only", "mcc"]
            print(f"  [xgboost h={h}] combined MCC - price-only MCC = {delta:+.4f}")


if __name__ == "__main__":
    main()
