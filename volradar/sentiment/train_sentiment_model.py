"""Step 2 — Fine-tune DistilBERT for financial-tweet sentiment.

Trains `distilbert-base-uncased` (66M params, CPU-friendly) on the pre-labeled
`zeroshot/twitter-financial-news-sentiment` dataset to classify each tweet as
Bearish / Bullish / Neutral, then reports accuracy + macro-F1 on the dataset's own
validation split.

Why fine-tune a small transformer instead of using a bag-of-words baseline?
A transformer already "understands" general English from pretraining; fine-tuning only
nudges it to map that understanding onto the three finance-sentiment classes, which is
why a few epochs on ~9.5k examples is enough. Why DistilBERT specifically? It keeps ~97%
of BERT's quality at ~60% of the size, so it actually fine-tunes on a laptop CPU in a
reasonable time — the whole point of this project's CPU-only constraint.

Run:
    python sentiment_model/train_sentiment_model.py            # full run (2-3 epochs)
    python sentiment_model/train_sentiment_model.py --epochs 1 # faster fallback
    python sentiment_model/train_sentiment_model.py --smoke    # tiny subset, wiring test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `import config` work regardless of where the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def compute_metrics(eval_pred):
    """Accuracy + macro-F1. Macro-F1 averages F1 across the three classes equally, so a
    model that ignores the minority class is penalized — more honest than plain accuracy
    on a mildly imbalanced dataset."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on financial-tweet sentiment.")
    parser.add_argument("--epochs", type=int, default=config.SENT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.SENT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.SENT_LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=config.SENT_MAX_LENGTH)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train on a tiny subset for a fast end-to-end wiring check (not a real result).",
    )
    parser.add_argument("--output-dir", type=str, default=str(config.SENTIMENT_MODEL_DIR))
    args = parser.parse_args()

    # Heavy imports done inside main so `--help` and unit-test imports stay fast.
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    print(f"Loading dataset: {config.SENTIMENT_DATASET}")
    ds = load_dataset(config.SENTIMENT_DATASET)
    # This dataset ships a 'train' (~9,540) and a 'validation' (~2,390) split. We use them
    # directly rather than re-splitting, per the project spec.
    print({k: len(v) for k, v in ds.items()})

    # Confirm the label set matches our assumed mapping (0=Bearish,1=Bullish,2=Neutral)
    # instead of assuming it — dataset cards occasionally change.
    label_values = sorted(set(ds["train"]["label"]))
    print(f"Label values present: {label_values}  (expected {sorted(config.LABEL_NAMES)})")
    assert label_values == sorted(config.LABEL_NAMES), "Unexpected label set; re-check dataset card."

    if args.smoke:
        ds["train"] = ds["train"].select(range(min(200, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(100, len(ds["validation"]))))
        args.epochs = 1
        print("SMOKE MODE: tiny subset, 1 epoch — wiring check only.")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_SENTIMENT_MODEL)

    def tokenize(batch):
        # Tweets are short; truncating/padding to 128 tokens is plenty and keeps CPU cost low.
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_SENTIMENT_MODEL,
        num_labels=config.NUM_LABELS,
        id2label=config.LABEL_NAMES,
        label2id=config.NAME_TO_LABEL,
    )

    # Dynamic padding per-batch (pad to the longest in the batch, not a fixed 128) — less
    # wasted compute on a CPU.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",          # we save the final model ourselves below
        logging_steps=50,
        seed=config.RANDOM_SEED,
        report_to="none",            # no W&B/experiment tracker
        use_cpu=True,                # force CPU per the project's compute constraint
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,  # transformers 5.x renamed `tokenizer` -> `processing_class`
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Training for {args.epochs} epoch(s) on CPU...")
    trainer.train()

    print("Final evaluation on the validation split:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved fine-tuned model + tokenizer to {out_dir}")

    # Persist the headline metrics next to the model so the README/model-card can quote
    # the exact numbers without re-running training.
    (out_dir / "sentiment_metrics.txt").write_text(
        f"eval_accuracy: {metrics.get('eval_accuracy', float('nan')):.4f}\n"
        f"eval_f1_macro: {metrics.get('eval_f1_macro', float('nan')):.4f}\n"
        f"epochs: {args.epochs}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
