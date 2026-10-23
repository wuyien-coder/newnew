"""Bake real BGE corpus vectors into the online image, without any LLM key."""
from pathlib import Path

from mathkb.rag.chunking import load_directory
from mathkb.rag.embedding import BgeEmbedding


def warm(model_path="/opt/mathkb-model", cache_dir="/opt/mathkb-vector-cache"):
    root = Path(__file__).resolve().parents[1]
    embedder = BgeEmbedding(model_path, model_path,
                            vector_cache_dir=cache_dir, batch_size=16)
    chunks = load_directory(root / "data" / "full" / "biographies", chunk_size=500, chunk_overlap=100)
    batch, count = [], 0
    for chunk in chunks:
        batch.append(chunk.text)
        if len(batch) == 16:
            embedder.encode(batch)
            count += len(batch)
            batch = []
            if count % 512 == 0:
                print(f"BGE corpus vectors: {count}", flush=True)
    if batch:
        embedder.encode(batch)
        count += len(batch)
    if count != 9921:
        raise RuntimeError(f"Full corpus changed: expected 9921 chunks, found {count}")
    print(f"BGE full corpus ready: {count} chunks", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/opt/mathkb-model")
    parser.add_argument("--cache-dir", default="/opt/mathkb-vector-cache")
    args = parser.parse_args()
    warm(args.model_path, args.cache_dir)
