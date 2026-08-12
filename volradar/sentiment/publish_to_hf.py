"""Publish the fine-tuned DistilBERT financial-sentiment model to the Hugging Face Hub.

Requires a HF *write* token on the machine (`hf auth login`). Uploads only the
inference artifacts (weights + tokenizer + config + metrics) and the model card as
README.md — the intermediate `checkpoints/` and `training_args.bin` are skipped.

Run:
    hf auth login                                  # once, interactive
    python sentiment_model/publish_to_hf.py        # push to the default repo id
    python sentiment_model/publish_to_hf.py --repo-id you/my-model --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

DEFAULT_REPO_ID = "tooshlegacy/distilbert-financial-tweet-sentiment"
MODEL_CARD = Path(__file__).resolve().parent / "model_card.md"
# Intermediate training artifacts that don't belong in an inference model repo.
IGNORE = ["checkpoints/*", "checkpoints/**", "training_args.bin", "*.tmp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    args = ap.parse_args()

    from huggingface_hub import HfApi, whoami

    who = whoami()["name"]
    print(f"Authenticated as: {who}")

    model_dir = config.SENTIMENT_MODEL_DIR
    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"No model weights at {model_dir} — train first.")

    api = HfApi()
    print(f"Creating repo {args.repo_id} (private={args.private}) ...")
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading model files from {model_dir} (skipping intermediates) ...")
    api.upload_folder(repo_id=args.repo_id, folder_path=str(model_dir),
                      ignore_patterns=IGNORE, commit_message="Add inference artifacts")

    if MODEL_CARD.exists():
        print("Uploading model card as README.md ...")
        api.upload_file(path_or_fileobj=str(MODEL_CARD), path_in_repo="README.md",
                        repo_id=args.repo_id, commit_message="Add model card")

    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
