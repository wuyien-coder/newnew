"""多路召回融合：RRF（默认）与加权分数归一融合。"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .vector_store import Hit


def rrf(hits_list: Iterable[Iterable[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion：只依赖名次，天然免疫各路分数量纲差异。"""
    scores: dict[str, float] = defaultdict(float)
    pool: dict[str, Hit] = {}
    for hits in hits_list:
        for rank, hit in enumerate(hits):
            scores[hit.doc_id] += 1.0 / (k + rank + 1)
            if hit.doc_id not in pool:
                pool[hit.doc_id] = hit
            else:
                keep = pool[hit.doc_id]
                keep.sources = sorted(set(keep.sources) | {hit.source})
                # 融合只留一个 Hit 副本，另一路的原始分数必须并进来，不然量纲信息丢失
                keep.dense_score = keep.dense_score if keep.dense_score is not None else hit.dense_score
                keep.lexical_score = (keep.lexical_score if keep.lexical_score is not None
                                      else hit.lexical_score)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    fused: list[Hit] = []
    for doc_id, score in ordered:
        hit = pool[doc_id]
        hit.fused_score = score
        fused.append(hit)
    return fused


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def weighted(hits_list: list[list[Hit]], weights: list[float]) -> list[Hit]:
    """加权和融合：需要人工校准量纲时用它做对照实验。"""
    if len(weights) != len(list(hits_list)):
        raise ValueError("weights 与召回路数不一致")
    combined: dict[str, float] = defaultdict(float)
    pool: dict[str, Hit] = {}
    for hits, w in zip(hits_list, weights):
        if not hits:
            continue
        normed = _minmax([h.score for h in hits])
        for hit, s in zip(hits, normed):
            combined[hit.doc_id] += w * s
            other = pool.get(hit.doc_id)
            if other is None:
                pool[hit.doc_id] = hit
            else:
                other.sources = sorted(set(other.sources) | {hit.source})
                other.dense_score = (other.dense_score if other.dense_score is not None
                                     else hit.dense_score)
                other.lexical_score = (other.lexical_score if other.lexical_score is not None
                                       else hit.lexical_score)
    ordered = sorted(combined.items(), key=lambda kv: -kv[1])
    out = []
    for doc_id, score in ordered:
        hit = pool[doc_id]
        hit.fused_score = score
        out.append(hit)
    return out
