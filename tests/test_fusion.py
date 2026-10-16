from __future__ import annotations

import pytest

from mathkb.rag.fusion import rrf, weighted
from mathkb.rag.vector_store import Hit


def make(prefix: str, n: int, source: str = "vector", score_start: float = 1.0):
    return [Hit(doc_id=f"{prefix}{i}", text=f"t{i}", score=score_start - i * 0.1, source=source)
            for i in range(n)]


def test_rrf_rewards_docs_ranked_high_by_both_channels():
    dense = make("d", 5)
    sparse = [Hit(doc_id="d2", text="t", score=9.0, source="keyword")] + make("s", 4, source="keyword")
    fused = rrf([dense, sparse], k=60)
    assert fused[0].doc_id == "d2", "两路都靠前的文档应融合后登顶"
    assert len(fused) == 9, "去重后应等于并集大小"
    assert set(fused[0].sources) == {"vector", "keyword"}


def test_rrf_is_scale_invariant():
    """RRF 只看名次：把其中一路分数放大 1000 倍，融合顺序不应改变。"""
    a, b = make("a", 4), make("b", 4)
    before = [h.doc_id for h in rrf([a, b])]
    b_scaled = [Hit(doc_id=h.doc_id, text=h.text, score=h.score * 1000, source=h.source) for h in b]
    assert [h.doc_id for h in rrf([a, b_scaled])] == before


def test_weighted_fusion_respects_weights():
    a, b = make("a", 3), make("b", 3)
    fused = weighted([a, b], [1.0, 0.0])
    assert {h.doc_id for h in fused[:3]} == {"a0", "a1", "a2"}


def test_weighted_validates_weight_count():
    with pytest.raises(ValueError):
        weighted([make("a", 2), make("b", 2)], [1.0])
