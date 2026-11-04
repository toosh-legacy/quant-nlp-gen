"""Deploy the dashboard to a Hugging Face Space, or pull its state back down.

Alternative to the default hosting (static GitHub Pages, see .github/workflows/refresh.yml):
this ships the live FastAPI server as a Docker Space, which requires a Hugging Face PRO
account. The daily workflow no longer calls it.

The Space repo doubles as the artifact store. Reports (data/reports/) and the news archive
(data/news/) are gitignored in this repo, so the daily GitHub Actions job starts from an
empty checkout: it pulls the last published state from the Space, runs `cli refresh`, and
pushes the result back. The backtest reports therefore persist without being committed
here, and the news archive keeps growing day over day.

    python scripts/deploy_space.py push            # stage app + reports, upload to the Space
    python scripts/deploy_space.py pull            # restore data/reports + data/news locally
    python scripts/deploy_space.py push --dry-run  # stage only, print what would upload

Auth: HF_TOKEN in the environment (CI), or a prior `hf auth login` (local).
The Space id comes from --space or HF_SPACE, e.g. `your-name/volatility-radar`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy" / "space"

# What the Space needs to build and serve. The frontend is built inside the image, so its
# sources ship rather than dist/; node_modules never does.
APP_FILES = [
    ("deploy/space/Dockerfile", "Dockerfile"),
    ("deploy/space/README.md", "README.md"),
    ("requirements-serve.txt", "requirements-serve.txt"),
]
APP_DIRS = ["volradar", "frontend"]
STATE_DIRS = ["data/reports", "data/news"]       # pushed and pulled; never in this git repo
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules", "dist", ".vite")

REQUIRED_REPORTS = ["snapshot.json", "pooled_oos.csv", "per_sector.csv", "oos_market.parquet",
                    "oos_sector.parquet", "conformal_market.csv", "walkforward_folds.csv"]


def _space_id(arg: str | None) -> str:
    sid = arg or os.environ.get("HF_SPACE")
    if not sid:
        sys.exit("No Space id: pass --space owner/name or set HF_SPACE.")
    return sid


def stage(dst: Path) -> list[str]:
    """Copy everything the Space needs into `dst`; return the relative paths staged."""
    missing = [r for r in REQUIRED_REPORTS if not (ROOT / "data" / "reports" / r).exists()]
    if missing:
        sys.exit(f"Refusing to deploy an incomplete dashboard; missing reports: {missing}. "
                 "Run the pipeline (or `deploy_space.py pull`) first.")
    for src, name in APP_FILES:
        shutil.copy2(ROOT / src, dst / name)
    for d in APP_DIRS + STATE_DIRS:
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, dst / d, ignore=SKIP)
    return sorted(str(p.relative_to(dst)).replace("\\", "/")
                  for p in dst.rglob("*") if p.is_file())


def push(space: str, dry_run: bool = False, private: bool = False) -> None:
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp))
        size = sum((Path(tmp) / f).stat().st_size for f in staged) / 1e6
        print(f"Staged {len(staged)} files ({size:.1f} MB) for {space}")
        if dry_run:
            for f in staged[:40]:
                print("  ", f)
            print("   ..." if len(staged) > 40 else "")
            return
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(space, repo_type="space", space_sdk="docker",
                        private=private, exist_ok=True)
        # delete_patterns makes the Space mirror the staged tree for app code and reports,
        # so a renamed or removed file does not linger and get served stale.
        api.upload_folder(repo_id=space, repo_type="space", folder_path=tmp,
                          delete_patterns=["volradar/**", "frontend/**", "data/reports/**"],
                          commit_message="Deploy Volatility Radar")
    print(f"Deployed -> https://huggingface.co/spaces/{space}")


def pull(space: str) -> None:
    """Restore the published reports and news archive into ./data (CI starts empty)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=space, repo_type="space",
                             allow_patterns=[f"{d}/**" for d in STATE_DIRS],
                             token=os.environ.get("HF_TOKEN"))
    n = 0
    for d in STATE_DIRS:
        src = Path(path) / d
        if src.exists():
            shutil.copytree(src, ROOT / d, dirs_exist_ok=True)
            n += sum(1 for p in src.rglob("*") if p.is_file())
    print(f"Pulled {n} files from {space} into data/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--space", help="owner/name (default: $HF_SPACE)")
    ap.add_argument("--dry-run", action="store_true", help="push: stage and list only")
    ap.add_argument("--private", action="store_true", help="push: create the Space private")
    args = ap.parse_args()
    space = _space_id(args.space) if not args.dry_run else (args.space or "dry/run")
    push(space, args.dry_run, args.private) if args.action == "push" else pull(space)


if __name__ == "__main__":
    main()
