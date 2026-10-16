"""Redis caches retrieval results only; conversation content is never cached."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from .logging import get_logger

log = get_logger(__name__)


class RetrievalCache:
    def __init__(self, settings):
        self.enabled = settings.enabled
        self.ttl = settings.ttl_seconds
        self.client = None
        if self.enabled:
            from redis import Redis
            self.client = Redis.from_url(settings.redis_url, decode_responses=True,
                                         socket_connect_timeout=1, socket_timeout=1)

    @staticmethod
    def key(ctx, query):
        payload = {"query": query, "config": asdict(ctx.retriever.cfg),
                   "index": ctx.index_stats, "embedding": ctx.embedder.name,
                   "model": ctx.settings.embedding.model_name,
                   "reranker": ctx.reranker.name, "schema": 1}
        # Build duration changes after restart and is deliberately excluded.
        payload["index"] = {k: v for k, v in ctx.index_stats.items() if k != "build_ms"}
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                            sort_keys=True).encode()).hexdigest()
        return f"mathkb:retrieval:v1:{digest}"

    def get(self, key):
        if not self.client:
            return None
        try:
            raw = self.client.get(key)
            if not raw:
                return None
            from ..rag.retriever import RetrievalConfig, RetrievalResult
            from ..rag.vector_store import Hit
            data = json.loads(raw)
            data["hits"] = [Hit(**hit) for hit in data["hits"]]
            data["config"] = RetrievalConfig(**data["config"]) if data.get("config") else None
            return RetrievalResult(**data)
        except Exception as exc:
            log.warning("检索缓存读取失败，执行正常召回: %s", type(exc).__name__)
            return None

    def put(self, key, result):
        if not self.client:
            return
        try:
            self.client.setex(key, self.ttl, json.dumps(asdict(result), ensure_ascii=False))
        except Exception as exc:
            log.warning("检索缓存写入失败: %s", type(exc).__name__)
