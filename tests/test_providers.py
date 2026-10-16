from __future__ import annotations


from mathkb.core.exceptions import IndexNotReadyError
from mathkb.core.config import Settings
from mathkb.rag.embedding import HashEmbedding, build_embedding
from mathkb.rag.vector_store import MemoryVectorStore, VectorRecord
from mathkb.services.container import AppContext


def test_hash_embedding_is_deterministic_and_normalized():
    embedder = HashEmbedding(dim=64)
    a = embedder.encode(["华罗庚在数论方面的工作"])
    b = embedder.encode(["华罗庚在数论方面的工作"])
    assert a == b
    assert abs(sum(x * x for x in a[0]) - 1.0) < 1e-6


def test_similar_text_scores_higher_than_unrelated():
    embedder = HashEmbedding(dim=256)
    query = embedder.encode_one("陈省身在微分几何上的贡献")
    close = embedder.encode_one("陈省身开创了几何学研究")
    far = embedder.encode_one("今天天气适合打篮球")
    sim = lambda u, v: sum(x * y for x, y in zip(u, v))  # noqa: E731
    assert sim(query, close) > sim(query, far)


def test_memory_vector_store_cosine_ordering_and_filter():
    store = MemoryVectorStore()
    store.upsert([
        VectorRecord(doc_id="1", text="aa", metadata={"person_name": "甲"}, embedding=[1, 0]),
        VectorRecord(doc_id="2", text="bb", metadata={"person_name": "乙"}, embedding=[0.7, 0.7]),
        VectorRecord(doc_id="3", text="cc", metadata={"person_name": "乙"}, embedding=[0, 1]),
    ])
    hits = store.query([1, 0], top_k=3)
    assert [h.doc_id for h in hits] == ["1", "2", "3"]
    assert hits[0].score > hits[1].score > hits[2].score
    assert [h.doc_id for h in store.query([0, 1], top_k=5, person="乙")] == ["3", "2"]
    assert store.query([1, 0], top_k=5, person="不存在") == []


def test_missing_embedding_is_rejected():
    try:
        MemoryVectorStore().upsert([VectorRecord(doc_id="x", text="t")])
    except ValueError:
        return
    raise AssertionError("缺 embedding 的记录应被拒绝")


def test_empty_index_raises_index_not_ready(tmp_root):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_root / 'empty.db'}",
                        data_dir=str(tmp_root / "nowhere"))
    settings.vector.persist_dir = str(tmp_root / "empty_chroma")
    settings.keyword.persist_dir = str(tmp_root / "empty_bm25")
    settings.embedding.provider = "hash"
    settings.ensure_dirs()
    ctx = AppContext.create(settings, seed=False)
    try:
        ctx.build_index()
    except IndexNotReadyError:
        return
    raise AssertionError("空库应抛 IndexNotReadyError")


def test_rebuild_index_refills_empty_keyword_store(ctx):
    """持久化向量库重启后数量已一致，稀疏索引仍必须被补上（真实踩过的坑）。"""
    from mathkb.rag.keyword_store import BM25Store

    total = ctx.vectors.count()
    assert ctx.keywords.count() == total
    ctx.keywords = BM25Store(cache_dir=ctx.settings.keyword.persist_dir)
    assert ctx.keywords.count() == 0
    ctx.build_index()  # force=False：向量数一致走跳过分支
    assert ctx.keywords.count() == total, "重启后 BM25 不能是空的"


def test_explicit_bge_provider_fails_without_model(tmp_root):
    import pytest
    settings = Settings(_env_file=None)
    settings.embedding.provider = "bge"
    settings.embedding.cache_dir = str(tmp_root / "no_models")
    with pytest.raises(RuntimeError):
        build_embedding(settings.embedding)
