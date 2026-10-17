"""Prepare the BGE model cache used by both native Python and Docker full mode."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(Path(__file__).resolve().parents[1] / "models"))
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=args.model, cache_dir=args.cache_dir)
    print(f"Model cache ready: {path}")


if __name__ == "__main__":
    main()
