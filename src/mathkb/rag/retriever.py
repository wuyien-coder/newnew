"""混合检索编排：稠密 + 稀疏 → 融合 → 精排 → 图谱增强。

每一路都可单独关闭（`RetrievalConfig`），这是论文消融实验在工程侧的等价物：
评测脚本用同一份代码跑不同开关组合，保证「一次只改一个变量」。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..core.config import RetrievalSettings
from ..core.logging import get_logger
from ..core.metrics import metrics
from .embedding import EmbeddingProvider
from .entity_linker import EntityLinker
from .fusion import rrf, weighted
from .keyword_store import BM25Store
from .kg_store import GraphStore
from .rerank import RerankProvider
from .vector_store import Hit, VectorStore

log = get_logger(__name__)


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_multiplier: int = 4
    max_candidates: int = 60
    rrf_k: int = 60
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    use_graph: bool = True
    use_entity_filter: bool = True
    fusion: str = "weighted"  # rrf | weighted
    # 去名化集（488 条）实测：等权 RRF 把稠密弱路（R@1 0.35）抬进前排，R@1 只有 0.506；
    # 2:8 加权融合到 0.980，追平 BM25 单路。风险：该集词面重合偏高，换人工改写集需重新校准。
    weights: list[float] = field(default_factory=lambda: [0.2, 0.8])

    @classmethod
    def from_settings(cls, s: RetrievalSettings, use_rerank: bool) -> "RetrievalConfig":
        return cls(top_k=s.top_k, candidate_multiplier=s.candidate_multiplier,
                   max_candidates=s.max_candidates, rrf_k=s.rrf_k, use_rerank=use_rerank)


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    confidence: str = "low"
    confidence_basis: str = "none"   # rerank | dense | lexical_only | none
    entities: list[str] = field(default_factory=list)
    graph_lines: list[str] = field(default_factory=list)
    person_filter: Optional[str] = None
    pool_size: int = 0
    config: Optional[RetrievalConfig] = None

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)


class HybridRetriever:
    def __init__(self, vectors: VectorStore, keywords: BM25Store, reranker: RerankProvider,
                 embedder: EmbeddingProvider, cfg: RetrievalConfig,
                 graph: Optional[GraphStore] = None,
                 linker: Optional[EntityLinker] = None):
        self.vectors = vectors
        self.keywords = keywords
        self.reranker = reranker
        self.embedder = embedder
        self.cfg = cfg
        self.graph = graph
        self.linker = linker

    def _candidate_k(self) -> int:
        return min(max(self.cfg.top_k * self.cfg.candidate_multiplier, 20), self.cfg.max_candidates)

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 person: Optional[str] = None) -> RetrievalResult:
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        pool_k = self._candidate_k()
        timings: dict[str, float] = {}

        # 实体链接：命中唯一人物时把检索限定在其传记范围内
        # 关掉它（use_entity_filter=False）即可测「纯排序质量」，否则 gold 会被过滤到饱和
        entities: list[str] = []
        if cfg.use_entity_filter:
            if person is None and self.linker is not None:
                t0 = time.perf_counter()
                entities = self.linker.link(query)
                timings["entity_link"] = round((time.perf_counter() - t0) * 1000, 2)
                if len(entities) == 1:
                    person = entities[0]

        routes: list[list[Hit]] = []
        if cfg.use_dense:
            t0 = time.perf_counter()
            vec = self.embedder.encode_one(query)
            dense_hits = self.vectors.query(vec, top_k=pool_k, person=person)
            for h in dense_hits:
                h.dense_score = h.score
            routes.append(dense_hits)
            timings["dense"] = round((time.perf_counter() - t0) * 1000, 2)

        if cfg.use_bm25:
            t0 = time.perf_counter()
            sparse_hits = self.keywords.search(query, top_k=pool_k, person=person)
            for h in sparse_hits:
                h.lexical_score = h.score
            routes.append(sparse_hits)
            timings["sparse"] = round((time.perf_counter() - t0) * 1000, 2)

        if not routes:
            raise ValueError("所有召回路都被关闭，无法检索")

        t0 = time.perf_counter()
        if cfg.fusion == "weighted" and len(routes) == len(cfg.weights):
            fused = weighted(routes, cfg.weights)
        else:
            fused = rrf(routes, k=cfg.rrf_k)
        timings["fusion"] = round((time.perf_counter() - t0) * 1000, 2)

        pool_size = len(fused)
        t0 = time.perf_counter()
        if cfg.use_rerank and len(fused) > top_k:
            final = self.reranker.rerank(query, fused[:pool_k], top_k)
        else:
            final = [h for h in fused[:top_k]]
        timings["rerank"] = round((time.perf_counter() - t0) * 1000, 2)

        graph_lines: list[str] = []
        if cfg.use_graph and self.graph is not None and entities:
            t0 = time.perf_counter()
            graph_lines = self.graph.triples_for(entities)
            timings["graph"] = round((time.perf_counter() - t0) * 1000, 2)

        confidence, basis = confidence_of(final, 0.60, 0.30)
        result = RetrievalResult(
            query=query, hits=final, timings_ms=timings,
            confidence=confidence, confidence_basis=basis, entities=entities,
            graph_lines=graph_lines, person_filter=person, pool_size=pool_size, config=cfg,
        )
        metrics.observe("retrieve.total_ms", result.total_ms)
        for stage, ms in timings.items():
            metrics.observe(f"retrieval.{stage}.ms", ms)
        metrics.inc("retrieval.queries")
        return result


def _relevance_scores(hits: list[Hit]) -> tuple[list[float], str]:
    """置信度只能用可比量纲的分数：精排 > 稠密余弦 > 无（判 low）。

    BM25 分数无上界，拿来比 0.6 阈值会恒判 high，所以显式排除。
    """
    reranked = [h.rerank_score for h in hits if h.rerank_score is not None]
    if len(reranked) >= max(1, len(hits) // 2):
        return reranked, "rerank"
    dense = [h.dense_score for h in hits if h.dense_score is not None]
    if len(dense) >= max(1, len(hits) // 2):
        return dense, "dense"
    return [], "lexical_only"


def confidence_of(hits: list[Hit], high: float, medium: float) -> tuple[str, str]:
    scores, basis = _relevance_scores(hits)
    if not scores:
        return "low", basis
    avg = sum(scores) / len(scores)
    if avg >= high:
        return "high", basis
    if avg >= medium:
        return "medium", basis
    return "low", basis
