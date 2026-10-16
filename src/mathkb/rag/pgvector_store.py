"""基于 SQLAlchemy + psycopg 3 的真正 pgvector 实现。

向量值、集合名、文本和筛选条件均通过绑定参数传递。唯一生成进 DDL 的
维度经过整数范围校验；建表、扩展和 HNSW 索引可由 Alembic 统一管理。
"""
from __future__ import annotations

import json
import math
from typing import Optional

from sqlalchemy import Engine, bindparam, create_engine, text

from ..db.base import normalize_database_url
from .vector_store import Hit, VectorRecord, VectorStore


def validate_dimension(dimension: int) -> int:
    if not isinstance(dimension, int) or isinstance(dimension, bool) or not 1 <= dimension <= 2000:
        raise ValueError("pgvector HNSW vector 维度必须是 1 到 2000 的整数")
    return dimension


def schema_statements(dimension: int) -> tuple[str, ...]:
    """供开发自动建表和初始迁移共用；生产权限可仅授予运行期 DML。"""
    dimension = validate_dimension(dimension)
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        """CREATE TABLE IF NOT EXISTS vector_collections (
            collection VARCHAR(128) PRIMARY KEY,
            signature VARCHAR(64) NOT NULL DEFAULT ''
        )""",
        f"""CREATE TABLE IF NOT EXISTS vector_chunks (
            collection VARCHAR(128) NOT NULL,
            doc_id VARCHAR(160) NOT NULL,
            text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            person_name VARCHAR(128) NOT NULL DEFAULT '',
            embedding vector({dimension}) NOT NULL,
            PRIMARY KEY (collection, doc_id)
        )""",
        "CREATE INDEX IF NOT EXISTS ix_vector_chunks_person ON vector_chunks (collection, person_name)",
        """CREATE INDEX IF NOT EXISTS ix_vector_chunks_hnsw
           ON vector_chunks USING hnsw (embedding vector_cosine_ops)
           WITH (m = 16, ef_construction = 64)""",
    )


def create_pgvector_schema(engine: Engine, dimension: int) -> None:
    with engine.begin() as conn:
        for statement in schema_statements(dimension):
            conn.execute(text(statement))


_UPSERT = text("""
    INSERT INTO vector_chunks (collection, doc_id, text, metadata, person_name, embedding)
    VALUES (:collection, :doc_id, :text, CAST(:metadata AS jsonb), :person_name,
            CAST(:embedding AS vector))
    ON CONFLICT (collection, doc_id) DO UPDATE SET
        text = EXCLUDED.text,
        metadata = EXCLUDED.metadata,
        person_name = EXCLUDED.person_name,
        embedding = EXCLUDED.embedding
    WHERE (vector_chunks.text, vector_chunks.metadata, vector_chunks.person_name, vector_chunks.embedding)
          IS DISTINCT FROM
          (EXCLUDED.text, EXCLUDED.metadata, EXCLUDED.person_name, EXCLUDED.embedding)
""")


class PgVectorStore(VectorStore):
    def __init__(self, database_url: str, collection: str = "mathkb", dimension: int = 512,
                 create_schema: bool = True, *, engine: Optional[Engine] = None):
        self.dimension = validate_dimension(dimension)
        if not collection or len(collection) > 128:
            raise ValueError("向量集合名长度必须为 1 到 128")
        url = normalize_database_url(database_url)
        if not url.startswith("postgresql"):
            raise ValueError("pgvector 仅支持 PostgreSQL database_url，离线请选 vector.provider=memory")
        self.collection = collection
        self.engine = engine or create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
        if create_schema:
            create_pgvector_schema(self.engine, self.dimension)
        # 不捕获数据库/扩展/权限异常，避免生产部署悄悄换为内存数据。
        with self.engine.connect() as conn:
            actual = conn.scalar(text("""
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                WHERE a.attrelid = 'vector_chunks'::regclass
                  AND a.attname = 'embedding' AND NOT a.attisdropped
            """))
            if actual != f"vector({self.dimension})":
                raise ValueError(f"向量表维度 {actual} 与当前模型 {self.dimension} 不符；请先迁移/重建索引")
            # create_schema=false 时也检查状态表及索引，尽早报告漏跑迁移。
            conn.execute(text("SELECT signature FROM vector_collections LIMIT 0"))
            indexed = conn.scalar(text("SELECT to_regclass('ix_vector_chunks_hnsw')"))
            if not indexed:
                raise ValueError("pgvector HNSW 索引缺失，请执行 alembic upgrade head")

    def _vector_literal(self, vector: list[float]) -> str:
        if len(vector) != self.dimension:
            raise ValueError(f"向量长度应为 {self.dimension}，实际 {len(vector)}")
        values = [float(v) for v in vector]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("向量不能包含 NaN 或 Infinity")
        return "[" + ",".join(str(v) for v in values) + "]"

    def _payloads(self, records: list[VectorRecord]) -> list[dict]:
        payloads = []
        for record in records:
            if record.embedding is None:
                raise ValueError(f"记录 {record.doc_id} 缺少 embedding")
            payloads.append({
                "collection": self.collection, "doc_id": record.doc_id, "text": record.text,
                "metadata": json.dumps(record.metadata, ensure_ascii=False, allow_nan=False),
                "person_name": str(record.metadata.get("person_name", "")),
                "embedding": self._vector_literal(record.embedding),
            })
        return payloads

    def upsert(self, records: list[VectorRecord]) -> int:
        payloads = self._payloads(records)
        if payloads:
            with self.engine.begin() as conn:
                conn.execute(_UPSERT, payloads)
                # 外部逐批写入使已有完整快照签名失效。
                conn.execute(text("DELETE FROM vector_collections WHERE collection=:collection"),
                             {"collection": self.collection})
        return len(records)

    def replace_all(self, records: list[VectorRecord], signature: str = "") -> int:
        """同一事务内 upsert + 清理旧块 + 更新签名，读请求看见完整提交。"""
        payloads = self._payloads(records)
        with self.engine.begin() as conn:
            # API 与 Celery 在同一集合上的替换不能交叉执行。
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:collection))"),
                         {"collection": "mathkb-vectors:" + self.collection})
            if payloads:
                conn.execute(_UPSERT, payloads)
                delete_old = text("""DELETE FROM vector_chunks
                    WHERE collection=:collection AND doc_id NOT IN :ids""").bindparams(
                        bindparam("ids", expanding=True)
                    )
                conn.execute(delete_old, {"collection": self.collection,
                                          "ids": [r.doc_id for r in records]})
            else:
                conn.execute(text("DELETE FROM vector_chunks WHERE collection=:collection"),
                             {"collection": self.collection})
            conn.execute(text("""INSERT INTO vector_collections (collection, signature)
                VALUES (:collection, :signature)
                ON CONFLICT (collection) DO UPDATE SET signature=EXCLUDED.signature"""),
                {"collection": self.collection, "signature": signature})
        return len(records)

    def query(self, vector: list[float], top_k: int = 10,
              person: Optional[str] = None, field: Optional[str] = None) -> list[Hit]:
        if top_k <= 0:
            return []
        params: dict = {"collection": self.collection, "vector": self._vector_literal(vector),
                        "top_k": int(top_k)}
        if not any(float(value) != 0 for value in vector):
            return []  # pgvector cosine distance is undefined for a zero vector
        filters = ["collection=:collection"]
        if person:
            filters.append("person_name=:person")
            params["person"] = person
        if field:
            filters.append("strpos(COALESCE(metadata->>'research_field', ''), :field) > 0")
            params["field"] = field
        where = " AND ".join(filters)
        if person or field or self.count() < 2000:
            # Materialize selective filters first. HNSW post-filtering can otherwise
            # return too few matches among thousands of other people/collections.
            prefix = "WITH candidates AS MATERIALIZED (SELECT * FROM vector_chunks WHERE " + where + ") "
            from_clause = "FROM candidates"
        else:
            prefix = ""
            from_clause = "FROM vector_chunks WHERE " + where
        statement = text(prefix + """
            SELECT doc_id, text, metadata,
                   1 - (embedding <=> CAST(:vector AS vector)) AS score
            """ + from_clause + """
            ORDER BY embedding <=> CAST(:vector AS vector) ASC LIMIT :top_k
        """)
        with self.engine.begin() as conn:
            conn.execute(text("SET LOCAL hnsw.ef_search = 100"))
            rows = conn.execute(statement, params).mappings().all()
        return [Hit(doc_id=row["doc_id"], text=row["text"], score=float(row["score"]),
                    source="vector", metadata=dict(row["metadata"] or {})) for row in rows]

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.scalar(text("SELECT count(*) FROM vector_chunks WHERE collection=:collection"),
                                   {"collection": self.collection}) or 0)

    def signature(self) -> str:
        with self.engine.connect() as conn:
            return str(conn.scalar(text("SELECT signature FROM vector_collections WHERE collection=:collection"),
                                   {"collection": self.collection}) or "")

    def all_records(self) -> list[VectorRecord]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("""SELECT doc_id, text, metadata FROM vector_chunks
                WHERE collection=:collection ORDER BY doc_id"""),
                {"collection": self.collection}).mappings().all()
        return [VectorRecord(doc_id=row["doc_id"], text=row["text"], metadata=dict(row["metadata"] or {}))
                for row in rows]

    def clear(self) -> None:
        self.replace_all([])
