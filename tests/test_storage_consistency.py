"""真实 SQLite 集成回归：同数量更新、旧 JSON 缓存及跨进程版本刷新。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import delete, select

from mathkb.core.config import Settings
from mathkb.db.models import Chunk, Document, Person
from mathkb.rag.keyword_store import BM25Store, records_signature
from mathkb.rag.vector_store import VectorRecord
from mathkb.services.container import AppContext


def small_context(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path / 'storage.db'}",
                        index_dir=str(tmp_path / "index"),
                        embedding={"provider": "hash", "dim": 64},
                        llm={"provider": "stub"}, vector={"provider": "memory"},
                        keyword={"persist_dir": str(tmp_path / "bm25")})
    ctx = AppContext.create(settings, seed=False)
    with ctx.db.session() as session:
        doc = Document(person_name="测试人物", file_path="memory://person")
        session.add(doc)
        session.flush()
        session.add_all([
            Chunk(id="first", document_id=doc.id, person_name="测试人物", chunk_index=0,
                  text="解析数论与数学研究", metadata_={"source": "论文"}),
            Chunk(id="second", document_id=doc.id, person_name="其他人物", chunk_index=1,
                  text="微分几何与拓扑学研究", metadata_={}),
            Person(name="测试人物", hometown="江苏", institutions=[]),
        ])
    ctx.bump_revision()
    ctx.build_index()
    return ctx


def test_same_count_text_update_rebuilds_vectors_and_bm25(tmp_path):
    ctx = small_context(tmp_path)
    old_retriever = ctx.retriever
    old_embedding = ctx.vectors.get("first").embedding
    old_signature = ctx.index_stats["signature"]
    with ctx.db.session() as session:
        chunk = session.get(Chunk, "first")
        chunk.text = "计算机体系结构与分布式系统"
        chunk.person_name = "新人物"
    # 即使直接修改 SQL 未 bump revision，显式 build_index 也必须比较内容。
    stats = ctx.build_index()
    assert ctx.vectors.count() == 2
    assert ctx.vectors.get("first").text == "计算机体系结构与分布式系统"
    assert ctx.vectors.get("first").embedding != old_embedding
    assert ctx.keywords._records[0].metadata["person_name"] == "新人物"
    assert stats["signature"] != old_signature
    assert old_retriever.vectors.get("first").text == "解析数论与数学研究"
    assert old_retriever.keywords._records[0].text == "解析数论与数学研究"


def test_same_count_delete_insert_removes_obsolete_vector(tmp_path):
    ctx = small_context(tmp_path)
    with ctx.db.session() as session:
        document_id = session.scalar(select(Chunk.document_id).where(Chunk.id == "first"))
        session.execute(delete(Chunk).where(Chunk.id == "first"))
        session.add(Chunk(id="replacement", document_id=document_id, chunk_index=0,
                          person_name="新人物", text="替换后的文档", metadata_={}))
    ctx.build_index()
    assert ctx.vectors.count() == 2
    assert ctx.vectors.get("first") is None
    assert ctx.vectors.get("replacement").text == "替换后的文档"


def test_api_refreshes_after_other_context_bumps_revision(tmp_path):
    api = small_context(tmp_path)
    worker = AppContext.create(api.settings, seed=False)
    worker.build_index()
    previous = api.index_stats["revision"]
    with worker.db.session() as session:
        session.get(Chunk, "first").text = "Celery 摄取的新内容"
        session.scalar(select(Person).where(Person.name == "测试人物")).hometown = "北京"
    revision = worker.bump_revision()
    assert revision > previous
    refreshed = api.ensure_fresh()
    assert refreshed["revision"] == revision
    assert api.vectors.get("first").text == "Celery 摄取的新内容"
    assert api.graph.persons["测试人物"].hometown == "北京"
    retriever = api.retriever
    api.ensure_fresh()
    assert api.retriever is retriever


def test_bm25_json_rejects_same_count_stale_and_tampered_cache(tmp_path):
    import json

    old = [VectorRecord(doc_id="a", text="原始数学文本"), VectorRecord(doc_id="b", text="几何文本")]
    new = [VectorRecord(doc_id="a", text="更新后的文本"), VectorRecord(doc_id="b", text="几何文本")]
    store = BM25Store(str(tmp_path))
    store.build(old)
    loaded = BM25Store(str(tmp_path))
    assert loaded.load(records_signature(old)) is True
    assert loaded.load(records_signature(new)) is False
    payload = json.loads((tmp_path / "bm25.json").read_text(encoding="utf-8"))
    payload["records"][0]["text"] = "已被篡改"
    (tmp_path / "bm25.json").write_text(json.dumps(payload), encoding="utf-8")
    assert BM25Store(str(tmp_path)).load(records_signature(old)) is False
    # 遗留 pickle 不会被反序列化。
    (tmp_path / "bm25.pkl").write_bytes(b"not a pickle")
    assert not BM25Store(str(tmp_path), "bm25.pkl").load()


def test_bm25_restart_reloads_current_content(tmp_path):
    original = small_context(tmp_path)
    with original.db.session() as session:
        session.get(Chunk, "first").text = "相同数量下变更的启动语料"
    restarted = AppContext.create(original.settings, seed=False)
    restarted.build_index()
    records = {r.doc_id: r.text for r in restarted.keywords._records}
    assert records["first"] == "相同数量下变更的启动语料"


def test_revision_increment_is_atomic_under_concurrent_writers(tmp_path):
    ctx = small_context(tmp_path)
    start = ctx.db.index_revision()
    with ThreadPoolExecutor(max_workers=4) as pool:
        revisions = list(pool.map(lambda _: ctx.bump_revision(), range(12)))
    assert len(set(revisions)) == 12
    assert ctx.db.index_revision() == start + 12


def test_context_close_releases_distinct_pools_and_client_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    ctx = small_context(tmp_path)
    closed = []
    monkeypatch.setattr(ctx.db.engine, "dispose", lambda: closed.append("database"))
    ctx.vectors.engine = SimpleNamespace(dispose=lambda: closed.append("vectors"))
    ctx.llm.client = SimpleNamespace(close=lambda: closed.append("llm"))
    ctx.close()
    ctx.close()
    assert closed == ["vectors", "database", "llm"]


def test_request_snapshot_keeps_its_paired_signature_and_retriever(tmp_path):
    ctx = small_context(tmp_path)
    request = ctx.snapshot()
    old_signature = request.index_stats["signature"]
    with ctx.db.session() as session:
        session.get(Chunk, "first").text = "后台摄取后的最新内容"
    ctx.bump_revision()
    refreshed = ctx.snapshot()
    assert refreshed.index_stats["signature"] != old_signature
    assert refreshed.retriever.vectors.get("first").text == "后台摄取后的最新内容"
    assert request.index_stats["signature"] == old_signature
    assert request.retriever.vectors.get("first").text == "解析数论与数学研究"
