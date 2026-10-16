"""精排 Provider：Cross-Encoder 与直通模式。

论文实验里精排是延迟瓶颈（同机实测：不精排约 2s / 精排约 36s，候选 20 条），
所以这里把它做成可配置开关，并把延迟打点交给上层 metrics，而不是默认开启。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.config import RerankSettings
from ..core.logging import get_logger
from .vector_store import Hit

log = get_logger(__name__)


class RerankProvider(ABC):
    name = "base"

    @abstractmethod
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class PassthroughRerank(RerankProvider):
    name = "passthrough"

    def rerank(self, query, hits, top_k):
        for hit in hits:
            hit.rerank_score = None
        return hits[:top_k]


class CrossEncoderRerank(RerankProvider):
    name = "cross-encoder"

    def __init__(self, cfg: RerankSettings, cache_dir: str):
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import CrossEncoder

        from .embedding import _resolve_snapshot

        path = _resolve_snapshot(cfg.model_name, cfg.cache_dir or cache_dir)
        log.info("加载 reranker 模型: %s", path)
        self.model = CrossEncoder(path, cache_dir=cache_dir, max_length=512,
                                  local_files_only=True)
        self.cfg = cfg

    def rerank(self, query, hits, top_k):
        if not hits:
            return []
        pairs = [(query, h.text[: self.cfg.doc_chars]) for h in hits]
        scores = self.model.predict(pairs, show_progress_bar=False,
                                    batch_size=self.cfg.batch_size)
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
        ordered = sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)
        return ordered[:top_k]


def build_rerank(cfg: RerankSettings, embedding_cache_dir: str) -> RerankProvider:
    if not cfg.enabled:
        return PassthroughRerank()
    try:
        return CrossEncoderRerank(cfg, embedding_cache_dir)
    except Exception as exc:
        raise RuntimeError("精排模型已启用但加载失败，请检查本地 rerank 模型及 ml 依赖") from exc
