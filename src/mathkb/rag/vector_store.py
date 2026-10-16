"""离线 numpy 向量检索与 PostgreSQL/pgvector 的统一接口。"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.logging import get_logger

log = get_logger(__name__)


@dataclass
class VectorRecord:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float          # 该路召回的原始分数（越大越相关）
    source: str           # "vector" | "keyword" | "graph"
    metadata: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    fused_score: float = 0.0
    rerank_score: Optional[float] = None
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = [self.source]


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> int: ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int = 10,
              person: Optional[str] = None, field: Optional[str] = None) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...

    def clear(self) -> None:
        raise NotImplementedError

    def replace_all(self, records: list[VectorRecord], signature: str = "") -> int:
        """完整替换一个集合，删除已从源数据库移除的旧分块。"""
        raise NotImplementedError


class MemoryVectorStore(VectorStore):
    """精确余弦检索，用于离线演示及单测；不声称具有 ANN 的扩展能力。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, VectorRecord] = {}
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[str] = []

    def upsert(self, records: list[VectorRecord]) -> int:
        for rec in records:
            if rec.embedding is None:
                raise ValueError(f"记录 {rec.doc_id} 缺少 embedding")
        with self._lock:
            for rec in records:
                self._records[rec.doc_id] = rec
            self._matrix = None
        return len(records)

    def _rebuild(self) -> None:
        self._ids = list(self._records.keys())
        mat = np.asarray([self._records[i].embedding for i in self._ids], dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.where(norms == 0, 1.0, norms)

    def query(self, vector, top_k=10, person=None, field=None):
        if top_k <= 0:
            return []
        with self._lock:
            if not self._records:
                return []
            if self._matrix is None:
                self._rebuild()
            vec = np.asarray(vector, dtype="float32")
            n = np.linalg.norm(vec)
            if n:
                vec = vec / n
            sims = self._matrix @ vec
            mask = np.ones(len(self._ids), dtype=bool)
            if person:
                mask &= [self._records[i].metadata.get("person_name") == person for i in self._ids]
            if field:
                mask &= [field in str(self._records[i].metadata.get("research_field", ""))
                         for i in self._ids]
            sims = np.where(mask, sims, -np.inf)
            k = min(top_k, int(mask.sum()))
            if k <= 0:
                return []
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            return [
                Hit(doc_id=self._ids[i], text=self._records[self._ids[i]].text,
                    score=float(sims[i]), source="vector",
                    metadata=self._records[self._ids[i]].metadata)
                for i in idx if sims[i] > -np.inf
            ]

    def count(self) -> int:
        return len(self._records)

    def get(self, doc_id: str) -> Optional[VectorRecord]:
        return self._records.get(doc_id)

    def all_records(self) -> list[VectorRecord]:
        return list(self._records.values())

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._matrix = None

    def replace_all(self, records: list[VectorRecord], signature: str = "") -> int:
        for rec in records:
            if rec.embedding is None:
                raise ValueError(f"记录 {rec.doc_id} 缺少 embedding")
        with self._lock:
            self._records = {r.doc_id: r for r in records}
            self._matrix = None
            self._ids = []
        return len(records)


def build_vector_store(cfg, database_url: Optional[str] = None,
                       dimension: int = 512, create_schema: bool = True) -> VectorStore:
    """生产配置错误立即报错，禁止把 pgvector 故障静默变成进程内数据。"""
    if cfg.provider == "memory":
        return MemoryVectorStore()
    if cfg.provider == "pgvector":
        from .pgvector_store import PgVectorStore

        if not database_url:
            raise ValueError("pgvector 需要配置 PostgreSQL database_url")
        return PgVectorStore(database_url, collection=cfg.collection,
                             dimension=dimension, create_schema=create_schema)
    raise ValueError(f"不支持的向量库 provider: {cfg.provider}")
