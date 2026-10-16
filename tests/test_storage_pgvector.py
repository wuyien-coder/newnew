"""验证真实 PostgreSQL SQL 语义，另提供可选数据库集成用例。"""
from __future__ import annotations

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mathkb.rag.pgvector_store import PgVectorStore, schema_statements
from mathkb.rag.vector_store import VectorRecord, build_vector_store


class Result:
    def __init__(self, rows=()):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class Connection:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, parameters=None):
        self.engine.calls.append((str(statement), parameters))
        return Result(self.engine.rows)

    def scalar(self, statement, parameters=None):
        sql = str(statement)
        self.engine.calls.append((sql, parameters))
        if "format_type" in sql:
            return "vector(3)"
        if "to_regclass" in sql:
            return "ix_vector_chunks_hnsw"
        return 0


class CapturingEngine:
    def __init__(self):
        self.calls = []
        self.rows = []

    def begin(self):
        return Connection(self)

    def connect(self):
        return Connection(self)


def test_schema_has_dimension_and_hnsw_cosine_index():
    sql = "\n".join(schema_statements(512))
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "embedding vector(512)" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "PRIMARY KEY (collection, doc_id)" in sql
    with pytest.raises(ValueError):
        schema_statements("512); DROP TABLE documents; --")


def test_pgvector_upsert_delete_old_and_query_are_parameter_bound():
    engine = CapturingEngine()
    collection = "collection'); DROP TABLE documents; --"
    store = PgVectorStore("postgresql://user:password@localhost/db", collection, 3,
                          create_schema=False, engine=engine)
    record = VectorRecord(doc_id="id'", text="文本'); DROP TABLE chunks; --",
                          metadata={"person_name": "张三", "research_field": "几何"}, embedding=[1, 0, 0])
    store.replace_all([record], signature="a" * 64)
    upsert = [(sql, params) for sql, params in engine.calls if "INSERT INTO vector_chunks" in sql][0]
    assert "ON CONFLICT (collection, doc_id) DO UPDATE" in upsert[0]
    assert collection not in upsert[0] and record.text not in upsert[0]
    assert upsert[1][0]["embedding"] == "[1.0,0.0,0.0]"
    deletion = [(sql, params) for sql, params in engine.calls if "doc_id NOT IN" in sql][0]
    assert deletion[1]["ids"] == ["id'"]
    person = "张三' OR TRUE; --"
    engine.rows = [{"doc_id": "id'", "text": "命中文本", "metadata": {"person_name": person}, "score": .9}]
    hits = store.query([1, 0, 0], top_k=7, person=person, field="几何")
    query = [(sql, params) for sql, params in engine.calls if "ORDER BY embedding" in sql][0]
    assert "embedding <=> CAST(:vector AS vector)" in query[0]
    assert "person_name=:person" in query[0]
    assert "LIMIT :top_k" in query[0]
    assert person not in query[0] and collection not in query[0]
    assert query[1]["person"] == person and query[1]["top_k"] == 7
    assert hits[0].score == .9 and hits[0].source == "vector"


def test_pgvector_rejects_sqlite_dimensions_and_nonfinite_values():
    with pytest.raises(ValueError, match="PostgreSQL"):
        PgVectorStore("sqlite:///:memory:")
    store = PgVectorStore("postgresql://localhost/db", dimension=3,
                          create_schema=False, engine=CapturingEngine())
    with pytest.raises(ValueError, match="向量长度"):
        store.query([1, 0])
    with pytest.raises(ValueError, match="NaN"):
        store.query([float("nan"), 0, 1])
    with pytest.raises(ValueError, match="缺少 embedding"):
        store.upsert([VectorRecord(doc_id="x", text="missing")])


def test_pgvector_factory_does_not_silently_fall_back(monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionError("postgres unavailable")

    monkeypatch.setattr("mathkb.rag.pgvector_store.PgVectorStore", fail)
    with pytest.raises(ConnectionError, match="postgres unavailable"):
        build_vector_store(SimpleNamespace(provider="pgvector", collection="test"),
                           database_url="postgresql://localhost/db")
    with pytest.raises(ValueError, match="provider"):
        build_vector_store(SimpleNamespace(provider="chroma"))


@pytest.mark.skipif(not os.environ.get("MATHKB_TEST_POSTGRES_URL"),
                    reason="设置 MATHKB_TEST_POSTGRES_URL 后运行真实 pgvector 集成测试")
def test_pgvector_live_upsert_filter_replace_and_delete():
    # 独立 collection；专用测试数据库应预留 512 维 vector_chunks 表。
    store = PgVectorStore(os.environ["MATHKB_TEST_POSTGRES_URL"],
                          collection="pytest_" + uuid4().hex, dimension=512)
    vector = [1.0] + [0.0] * 511
    other = [0.0, 1.0] + [0.0] * 510
    try:
        store.replace_all([
            VectorRecord("old", "旧文本", {"person_name": "甲"}, vector),
            VectorRecord("retained", "保留文本", {"person_name": "乙"}, other),
        ], signature="1" * 64)
        assert store.count() == 2
        assert [h.doc_id for h in store.query(vector, person="甲")] == ["old"]
        store.replace_all([
            VectorRecord("new", "新文本", {"person_name": "甲"}, vector),
            VectorRecord("retained", "已更新", {"person_name": "乙"}, other),
        ], signature="2" * 64)
        assert {r.doc_id for r in store.all_records()} == {"new", "retained"}
        assert store.signature() == "2" * 64
        assert store.query(vector, person="不存在") == []
    finally:
        store.clear()
        store.engine.dispose()
