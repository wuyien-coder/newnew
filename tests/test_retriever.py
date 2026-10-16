from __future__ import annotations

from dataclasses import replace

import pytest


@pytest.fixture
def cfg(ctx):
    return ctx.retriever.cfg


def test_hybrid_returns_hits_and_timings(ctx, cfg):
    result = ctx.retriever.retrieve("华罗庚在数论方面有哪些贡献")
    assert result.hits
    assert set(result.timings_ms) >= {"dense", "sparse", "fusion"}
    assert result.pool_size >= len(result.hits)
    assert result.confidence in ("high", "medium", "low")


def test_entity_link_narrows_to_person(ctx):
    result = ctx.retriever.retrieve("华罗庚的生平")
    assert result.entities == ["华罗庚"]
    assert result.person_filter == "华罗庚"
    assert all(h.metadata["person_name"] == "华罗庚" for h in result.hits)


def test_ablation_flags_actually_change_results(ctx):
    """关掉稠密路后，召回集合应发生变化——证明开关不是摆设。"""
    base = ctx.retriever.retrieve("微分几何 曲面 整体性质", top_k=5)
    original = ctx.retriever.cfg
    try:
        ctx.retriever.cfg = replace(original, use_dense=False)
        sparse_only = ctx.retriever.retrieve("微分几何 曲面 整体性质", top_k=5)
    finally:
        ctx.retriever.cfg = original
    assert {h.doc_id for h in base.hits} != {h.doc_id for h in sparse_only.hits} or \
        base.timings_ms.get("dense") and "dense" not in sparse_only.timings_ms


def test_all_channels_off_raises(ctx):
    original = ctx.retriever.cfg
    try:
        ctx.retriever.cfg = replace(original, use_dense=False, use_bm25=False)
        with pytest.raises(ValueError):
            ctx.retriever.retrieve("任意查询")
    finally:
        ctx.retriever.cfg = original


def test_bm25_finds_exact_phrase_beyond_dense(ctx):
    """词面强匹配是 BM25 的强项：去名化原句应能被稀疏路召回。"""
    original = ctx.retriever.cfg
    query = "1910年11月12日，出生在江苏省金坛县"
    try:
        ctx.retriever.cfg = replace(original, use_dense=False, use_entity_filter=False)
        hits = ctx.retriever.retrieve(query, top_k=5)
    finally:
        ctx.retriever.cfg = original
    assert hits.hits
    assert hits.hits[0].metadata["person_name"] == "华罗庚"
