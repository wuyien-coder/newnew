"""Explicit embedding providers and an atomic, content-addressed vector cache."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from ..core.config import ROOT, EmbeddingSettings
from ..core.logging import get_logger

log = get_logger(__name__)


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    @property
    def name(self) -> str:
        return self.__class__.__name__


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashEmbedding(EmbeddingProvider):
    """Explicit offline demonstration provider; never a BGE error fallback."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            norm_text = re.sub(r"\s+", "", text or "")
            for n in (2, 3):
                for i in range(len(norm_text) - n + 1):
                    gram = norm_text[i:i + n]
                    h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
                    vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            out.append(_l2(vec))
        return out


def text_digest(text: str) -> str:
    """Exact UTF-8 text identity: whitespace changes must invalidate embeddings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingDiskCache:
    """One SQLite database per model identity; atomic batches and concurrent readers.

    Only float32 vectors and hashes are stored. Transactions coordinate API/worker
    processes and prevent an interrupted write from exposing a partial cache batch.
    """

    def __init__(self, directory: str | Path, identity: dict, dimension: int):
        self.dimension = dimension
        self.identity = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        namespace = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{namespace}.sqlite3"
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS vectors (text_hash TEXT PRIMARY KEY, vector BLOB NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS cache_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO cache_meta VALUES ('identity', ?)", (self.identity,))
            conn.execute("INSERT OR IGNORE INTO cache_meta VALUES ('dimension', ?)", (str(dimension),))

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def get_many(self, texts: list[str]) -> dict[str, list[float]]:
        hashes = list(dict.fromkeys(text_digest(t) for t in texts))
        found = {}
        with self._lock, self._connect() as conn:
            for start in range(0, len(hashes), 500):
                batch = hashes[start:start + 500]
                marks = ",".join("?" for _ in batch)
                rows = conn.execute(f"SELECT text_hash, vector FROM vectors WHERE text_hash IN ({marks})", batch)
                for key, value in rows:
                    if len(value) != self.dimension * 4:
                        continue
                    vector = list(struct.unpack(f"<{self.dimension}f", value))
                    if all(math.isfinite(v) for v in vector):
                        found[key] = vector
        return found

    def put_many(self, texts: list[str], vectors: list[list[float]]) -> int:
        if len(texts) != len(vectors):
            raise ValueError("缓存文本数与向量数不一致")
        values = {}
        for text, vector in zip(texts, vectors):
            if len(vector) != self.dimension or not all(math.isfinite(float(v)) for v in vector):
                raise ValueError("缓存向量维度或数值无效")
            values[text_digest(text)] = struct.pack(f"<{self.dimension}f", *vector)
        with self._lock, self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", values.items())
        return len(values)

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT count(*) FROM vectors").fetchone()[0]


class BgeEmbedding(EmbeddingProvider):
    """Offline local BGE; an explicit BGE configuration loads the real model."""

    def __init__(self, model_name: str, cache_dir: str, fallback=None,
                 vector_cache_dir: str | Path | None = None, batch_size: int = 64):
        # The previous positional signature remains compatible, but never falls back.
        del fallback
        try:
            path = _resolve_snapshot(model_name, cache_dir)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            log.info("加载本地 BGE embedding 模型: %s", path)
            self.model = SentenceTransformer(path, cache_folder=cache_dir, local_files_only=True)
            self.dim = self.model.get_sentence_embedding_dimension()
            if not self.dim:
                raise ValueError("模型未返回有效向量维度")
        except Exception as exc:
            raise RuntimeError(
                f"BGE 模型加载失败：{model_name}。请确认本地模型目录与 ml 依赖完整；"
                "完整模式不会自动替换为 HashEmbedding。"
            ) from exc
        self.batch_size = batch_size
        self.model_path = path
        config_path = Path(path) / "config.json"
        revision = Path(path).name if Path(path).parent.name == "snapshots" else "local"
        self.identity = {
            "provider": "sentence-transformers", "model": model_name,
            "revision": revision, "dimension": self.dim,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "normalize_embeddings": True, "query_prefix": "",
            "max_seq_length": self.model.max_seq_length,
        }
        directory = (vector_cache_dir or os.environ.get("MATHKB_EMBEDDING__VECTOR_CACHE_DIR")
                     or ROOT / ".index" / "embedding-cache")
        self.cache = EmbeddingDiskCache(directory, self.identity, self.dim)
        self._encode_lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._encode_lock:
            found = self.cache.get_many(texts)
            missing = list(dict.fromkeys(t for t in texts if text_digest(t) not in found))
            self.cache_hits += len(texts) - sum(text_digest(t) not in found for t in texts)
            self.cache_misses += len(missing)
            if missing:
                vecs = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False,
                                         batch_size=self.batch_size)
                encoded = [list(map(float, v)) for v in vecs]
                self.cache.put_many(missing, encoded)
                found.update(self.cache.get_many(missing))
            return [found[text_digest(t)] for t in texts]


def _resolve_snapshot(model_name: str, cache_dir: str) -> str:
    """Resolve standard Hugging Face caches on Windows without a network call."""
    direct = Path(model_name).expanduser()
    if direct.is_dir() and (direct / "config.json").is_file():
        return str(direct.resolve())
    base = Path(cache_dir).expanduser()
    if (base / "config.json").is_file():
        return str(base.resolve())
    candidates = [base / ("models--" + model_name.replace("/", "--")),
                  base / "hub" / ("models--" + model_name.replace("/", "--")),
                  base / "models--" / model_name.replace("/", "--")]
    for repo in candidates:
        refs = repo / "refs" / "main"
        preferred = refs.read_text(encoding="utf-8").strip() if refs.is_file() else ""
        snapshots = repo / "snapshots"
        if not snapshots.is_dir():
            continue
        paths = sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.name)
        if preferred:
            paths.insert(0, snapshots / preferred)
        for path in paths:
            if (path / "config.json").is_file():
                return str(path.resolve())
    raise FileNotFoundError(f"本地模型未找到：{model_name}；缓存目录：{base}")


def build_embedding(cfg: EmbeddingSettings) -> EmbeddingProvider:
    if cfg.provider == "bge":
        return BgeEmbedding(cfg.model_name, cfg.cache_dir,
                            vector_cache_dir=getattr(cfg, "vector_cache_dir", None),
                            batch_size=cfg.batch_size)
    if cfg.provider != "hash":
        raise ValueError(f"未知 embedding provider：{cfg.provider}")
    log.info("使用显式离线 HashEmbedding(dim=%s)", cfg.dim)
    return HashEmbedding(dim=cfg.dim)
