"""Start the full native app on a container host with one persistent volume."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def configure_environment() -> None:
    token = os.environ.get("MATHKB_SERVER__API_TOKEN", "").strip()
    if len(token) < 24 or not token.isascii() or any(c.isspace() for c in token):
        raise RuntimeError("Set a private MATHKB_SERVER__API_TOKEN of at least 24 ASCII characters before deployment.")
    os.environ["MATHKB_SERVER__API_TOKEN"] = token
    root = Path(os.environ.get("MATHKB_PERSIST_DIR", "/data"))
    root.mkdir(parents=True, exist_ok=True)
    corpus = root / "corpus"
    source = Path(__file__).resolve().parents[1] / "data" / "full"
    # Corpus lives on the same durable volume as uploads. An interrupted initial
    # copy is retried before the marker is written; later uploads stay intact.
    marker = root / "corpus-ready"
    if not marker.exists():
        shutil.copytree(source, corpus, dirs_exist_ok=True)
        marker.write_text("full\n", encoding="utf-8")
    baked_cache = Path("/opt/mathkb-vector-cache")
    if baked_cache.is_dir() and not (root / "embedding-cache-ready").exists():
        shutil.copytree(baked_cache, root / "index" / "embedding-cache", dirs_exist_ok=True)
        (root / "embedding-cache-ready").write_text("bge-full\n", encoding="utf-8")
    defaults = {
        "MATHKB_ENV": "prod", "MATHKB_LOG_JSON": "true",
        "MATHKB_CREATE_SCHEMA": "true",
        "MATHKB_DATABASE_URL": f"sqlite:///{root / 'mathkb.db'}",
        "MATHKB_DATA_DIR": str(corpus),
        "MATHKB_BIOGRAPHIES_DIR": str(corpus / "biographies"),
        "MATHKB_INDEX_DIR": str(root / "index"),
        "MATHKB_KEYWORD__PERSIST_DIR": str(root / "index" / "bm25"),
        "MATHKB_EMBEDDING__PROVIDER": "bge",
        "MATHKB_EMBEDDING__MODEL_NAME": "/opt/mathkb-model",
        "MATHKB_EMBEDDING__CACHE_DIR": "/opt/mathkb-model",
        "MATHKB_EMBEDDING__VECTOR_CACHE_DIR": str(root / "index" / "embedding-cache"),
        "MATHKB_EMBEDDING__BATCH_SIZE": "16",
        "MATHKB_VECTOR__PROVIDER": "memory",
        "MATHKB_VECTOR__PERSIST_DIR": str(root / "index" / "vectors"),
        "MATHKB_QUEUE__PROVIDER": "local",
        "MATHKB_LLM__PROVIDER": "stub",
        "MATHKB_SERVER__CORS_ORIGINS": "[]",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


if __name__ == "__main__":
    configure_environment()
    import uvicorn
    # One API process: local background ingestion and the live index share a
    # consistent snapshot. Original PostgreSQL/Redis/Celery Compose remains supported.
    uvicorn.run("mathkb.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), workers=1)
