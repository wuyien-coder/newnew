"""稀疏检索：BM25 + 中文分词，带磁盘缓存。

生产替换点在 `search()`：把 rank_bm25 换成 Elasticsearch / OpenSearch 客户端即可，
上层只依赖 `list[Hit]` 与 `score` 语义。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from ..core.logging import get_logger
from .vector_store import Hit, VectorRecord

log = get_logger(__name__)

_STOPWORDS = set("的了和是在有与为及等把被者其 This a an the of to and in is are".split())


def records_signature(records: list[VectorRecord]) -> str:
    """对文本、主键及筛选元数据签名；文档数相同也能识别内容变更。"""
    payload = [{"doc_id": r.doc_id, "text": r.text, "metadata": r.metadata}
               for r in sorted(records, key=lambda item: item.doc_id)]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenize(text: str) -> list[str]:
    """jieba 优先，缺失时退化为字符 bigram（保证服务无 jieba 也能起）。"""
    try:
        import jieba

        parts = [t.strip() for t in jieba.lcut(text or "") if t.strip()]
    except Exception:
        squashed = re.sub(r"\s+", "", text or "")
        parts = [squashed[i:i + 2] for i in range(len(squashed) - 1)]
    return [t for t in parts if t not in _STOPWORDS and (len(t) > 1 or "\u4e00" <= t <= "\u9fff")]


class BM25Store:
    def __init__(self, cache_dir: Optional[str] = None, cache_name: str = "bm25.json"):
        self._index = None
        self._records: list[VectorRecord] = []
        # 从不读取遗留 pickle；即便调用方仍传 .pkl，实际也使用 JSON 文件。
        self._cache_path = (Path(cache_dir) / cache_name).with_suffix(".json") if cache_dir else None
        self.signature = ""

    def build(self, records: list[VectorRecord], *, persist: bool = True) -> None:
        from rank_bm25 import BM25Okapi

        self._records = records
        self.signature = records_signature(records)
        corpus = [tokenize(r.text) for r in records]
        self._index = BM25Okapi(corpus) if any(corpus) else None
        log.info("BM25 索引构建完成: %d 篇文档", len(records))
        if persist and self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "signature": self.signature,
                       "records": [{"doc_id": r.doc_id, "text": r.text, "metadata": r.metadata}
                                   for r in records]}
            # API/worker 可共享该目录；唯一临时文件 + 原子替换避免读到半份 JSON。
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                                 dir=self._cache_path.parent, delete=False) as fh:
                    temporary = fh.name
                    json.dump(payload, fh, ensure_ascii=False, allow_nan=False)
                os.replace(temporary, self._cache_path)
            finally:
                if temporary and Path(temporary).exists():
                    Path(temporary).unlink()

    def load(self, expected_signature: Optional[str] = None) -> bool:
        """只接受与当前源语料签名相符的 JSON 缓存；未提供签名则重新构建。"""
        if not expected_signature or not self._cache_path or not self._cache_path.exists():
            return False
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or payload.get("signature") != expected_signature:
                return False
            records = [VectorRecord(doc_id=r["doc_id"], text=r["text"], metadata=r.get("metadata", {}))
                       for r in payload["records"]]
            if records_signature(records) != expected_signature:
                return False
            self.build(records, persist=False)
            return True
        except Exception as exc:
            log.warning("BM25 缓存加载失败: %s", exc)
            return False

    def search(self, query: str, top_k: int = 20,
               person: Optional[str] = None) -> list[Hit]:
        if self._index is None or top_k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits: list[Hit] = []
        for i in order:
            rec = self._records[i]
            if person and rec.metadata.get("person_name") != person:
                continue
            hits.append(Hit(doc_id=rec.doc_id, text=rec.text, score=float(scores[i]),
                            source="keyword", metadata=rec.metadata))
            if len(hits) >= top_k:
                break
        return hits

    def count(self) -> int:
        return len(self._records)
