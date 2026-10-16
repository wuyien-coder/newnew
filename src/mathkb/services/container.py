"""应用容器：集中装配 provider，供 API / CLI / 测试复用（等价于 FastAPI 依赖注入根）。"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

from sqlalchemy import select

from ..core.config import Settings, get_settings
from ..core.exceptions import IndexNotReadyError
from ..core.logging import get_logger
from ..db.base import Database
from ..db.models import Chunk, Person, PersonField, Relation
from ..db.seed import seed_knowledgebase
from ..llm.base import LLMClient
from ..llm.factory import build_llm
from ..llm.usage import UsageTracker
from ..rag.embedding import EmbeddingProvider, build_embedding
from ..rag.entity_linker import EntityLinker
from ..rag.keyword_store import BM25Store, records_signature
from ..rag.kg_store import GraphPerson, GraphStore
from ..rag.rerank import RerankProvider, build_rerank
from ..rag.retriever import HybridRetriever, RetrievalConfig
from ..rag.vector_store import MemoryVectorStore, VectorRecord, VectorStore, build_vector_store

log = get_logger(__name__)


@dataclass
class AppContext:
    settings: Settings
    db: Database
    llm: LLMClient
    tracker: UsageTracker
    embedder: EmbeddingProvider
    vectors: VectorStore
    keywords: BM25Store
    reranker: RerankProvider
    graph: Optional[GraphStore] = None
    linker: Optional[EntityLinker] = None
    retriever: Optional[HybridRetriever] = None
    index_stats: dict = field(default_factory=dict)
    _index_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _seen_revision: int = field(default=-1, repr=False)
    _vector_signature: str = field(default="", repr=False)
    _closed: bool = field(default=False, repr=False)

    @classmethod
    def create(cls, settings: Optional[Settings] = None, seed: bool = True) -> "AppContext":
        settings = settings or get_settings()
        if settings.kg.provider not in {"relational", "sqlite"}:
            raise ValueError("图谱当前由关系表实现，请设 kg.provider=relational")
        db = Database(settings.database_url)
        if settings.create_schema:
            db.create_all()
        if seed:
            with db.session() as session:
                changes = seed_knowledgebase(session, settings)
            if any(changes.get(key, 0) for key in ("documents", "chunks", "persons", "relations")):
                db.bump_index_revision()
        embedder = build_embedding(settings.embedding)
        vectors = build_vector_store(settings.vector, database_url=settings.database_url,
                                     dimension=embedder.dim, create_schema=settings.create_schema)
        keywords = BM25Store(cache_dir=settings.keyword.persist_dir)
        reranker = build_rerank(settings.rerank, settings.embedding.cache_dir)
        tracker = UsageTracker(settings.llm.price_input, settings.llm.price_output)
        return cls(settings=settings, db=db, llm=build_llm(settings.llm), tracker=tracker,
                   embedder=embedder, vectors=vectors, keywords=keywords, reranker=reranker)

    # ---- 索引构建 ----
    def build_index(self, force: bool = False) -> dict:
        """按内容签名重建，进程内串行切换完整快照；PG 多进程也串行构建。"""
        with self._index_lock, self.db.index_build_lock(self.settings.vector.collection):
            return self._build_index(force)

    def _build_index(self, force: bool = False) -> dict:
        t0 = time.perf_counter()
        cfg = self.settings.retrieval
        use_rerank = self.settings.rerank.enabled
        revision = self.db.index_revision()
        with self.db.session() as session:
            rows = session.scalars(select(Chunk).order_by(Chunk.id)).all()
            if not rows:
                self.vectors.clear()
                self.keywords = BM25Store(cache_dir=self.settings.keyword.persist_dir)
                self.graph = None
                self.linker = None
                self.retriever = None
                self.index_stats = {"chunks": 0, "revision": revision, "signature": ""}
                self._seen_revision = revision
                raise IndexNotReadyError("知识库为空，请先执行 seed/ingest 再构建索引")
            records = [
                VectorRecord(doc_id=c.id, text=c.text,
                             metadata={**dict(c.metadata_ or {}), "person_name": c.person_name,
                                       "section_title": c.section_title})
                for c in rows
            ]
            persons = session.scalars(select(Person).order_by(Person.id)).all()
            fields_by_person: dict[int, list[str]] = {}
            for pf in session.scalars(select(PersonField).order_by(PersonField.id)).all():
                fields_by_person.setdefault(pf.person_id, []).append(pf.field)
            graph_persons = [
                GraphPerson(name=p.name, hometown=p.hometown, birth_year=p.birth_year,
                            death_year=p.death_year, achievements=p.achievements,
                            institutions=list(p.institutions or []),
                            fields=fields_by_person.get(p.id, []))
                for p in persons
            ]
            relations = [(r.from_name, r.to_name, r.rel_type)
                         for r in session.scalars(select(Relation).order_by(Relation.id)).all()]

        corpus_signature = records_signature(records)
        embedding_identity = {"provider": self.embedder.name, "dim": self.embedder.dim,
                              "configured": self.settings.embedding.model_dump()}
        vector_signature = self._digest({"corpus": corpus_signature, "embedding": embedding_identity})
        signature = self._digest({"vectors": vector_signature,
                                  "persons": [asdict(p) for p in graph_persons], "relations": relations})

        vectors_store = self.vectors
        existing_signature = (vectors_store.signature() if hasattr(vectors_store, "signature")
                              else self._vector_signature)
        replace_vectors = force or existing_signature != vector_signature or vectors_store.count() != len(records)
        if replace_vectors:
            for i in range(0, len(records), self.settings.embedding.batch_size):
                batch = records[i:i + self.settings.embedding.batch_size]
                encoded = self.embedder.encode([r.text for r in batch])
                if len(encoded) != len(batch):
                    raise ValueError("Embedding 返回的向量数与分块数不一致")
                for rec, vec in zip(batch, encoded):
                    rec.embedding = vec
            if isinstance(vectors_store, MemoryVectorStore):
                # 正在运行的请求保留旧实例，切换时不会读到部分更新的 numpy/BM25 索引。
                vectors_store = MemoryVectorStore()

        keywords = self.keywords
        if force or keywords.signature != corpus_signature or keywords.count() != len(records):
            keywords = BM25Store(cache_dir=self.settings.keyword.persist_dir)
            if force or not keywords.load(expected_signature=corpus_signature):
                keywords.build(records)

        graph = GraphStore(graph_persons, relations)
        linker = EntityLinker(graph.names)
        # 先准备好全部进程内结构，再提交持久向量替换，降低构建中途失败的影响。
        if replace_vectors:
            vectors_store.replace_all(records, signature=vector_signature)
        retriever = HybridRetriever(
            vectors=vectors_store, keywords=keywords, reranker=self.reranker,
            embedder=self.embedder, graph=graph, linker=linker,
            cfg=RetrievalConfig.from_settings(cfg, use_rerank=use_rerank),
        )
        stats = {
            "chunks": vectors_store.count(),
            "persons": len(graph_persons),
            "relations": len(relations),
            "embedder": self.embedder.name,
            "vector_store": vectors_store.__class__.__name__,
            "graph_provider": "relational",
            "revision": revision,
            "signature": signature,
            "reranker": self.reranker.name,
            "build_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        self.vectors, self.keywords = vectors_store, keywords
        self.graph, self.linker = graph, linker
        self.retriever, self.index_stats = retriever, stats
        self._vector_signature, self._seen_revision = vector_signature, revision
        log.info("索引就绪: %s", stats)
        return dict(stats)

    @staticmethod
    def _digest(value: dict) -> str:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    def bump_revision(self) -> int:
        """摄取事务提交后调用；供另一个 API/worker 进程识别本地索引过期。"""
        return self.db.bump_index_revision()

    def ensure_fresh(self) -> dict:
        """请求开始时查询轻量版本号，只在必要时重新加载 BM25/图谱/向量。"""
        with self._index_lock:
            revision = self.db.index_revision()
            if not self.ready or revision != self._seen_revision:
                return self.build_index()
            return dict(self.index_stats)

    def snapshot(self) -> "AppContext":
        """在同一锁内刷新并抓取请求视图，保证检索器、图谱和签名成套对应。"""
        with self._index_lock:
            self.ensure_fresh()
            return replace(self)

    def close(self) -> None:
        """任务结束/应用退出时释放两个独立连接池及上游 HTTP 客户端。"""
        with self._index_lock:
            if self._closed:
                return
            self._closed = True
            engines = [getattr(self.vectors, "engine", None), self.db.engine]
            disposed: set[int] = set()
            for engine in engines:
                if engine is not None and id(engine) not in disposed:
                    disposed.add(id(engine))
                    try:
                        engine.dispose()
                    except Exception as exc:
                        log.warning("关闭数据库连接池失败: %s", type(exc).__name__)
            client = getattr(self.llm, "client", None)
            if callable(getattr(client, "close", None)):
                try:
                    client.close()
                except Exception as exc:
                    log.warning("关闭模型客户端失败: %s", type(exc).__name__)

    @property
    def ready(self) -> bool:
        return self.retriever is not None
