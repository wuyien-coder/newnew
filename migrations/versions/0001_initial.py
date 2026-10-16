"""明确冻结的初始关系表、跨进程索引版本表和 pgvector HNSW schema。

Revision ID: 0001_initial
Revises: None

PG 首次迁移需要 CREATE EXTENSION 权限；之后运行期可只授予 DML 权限。
向量列维度由首次迁移的 MATHKB_EMBEDDING__DIM 决定；更换维度须新增迁移。
SQLite 离线模式创建关系表，跳过 PostgreSQL 专用 vector 表与扩展。
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("person_name", sa.String(128), nullable=False),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("page", sa.String(64), nullable=False),
        sa.Column("file_path", sa.String(512), nullable=False, unique=True),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_documents_person_name", "documents", ["person_name"])
    op.create_table("chunks",
        sa.Column("id", sa.String(160), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("person_name", sa.String(128), nullable=False),
        sa.Column("section_title", sa.String(255), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunk_pos"))
    op.create_index("ix_chunks_person_name", "chunks", ["person_name"])
    op.create_index("ix_chunk_person_section", "chunks", ["person_name", "section_title"])
    op.create_table("persons",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("hometown", sa.String(128), nullable=False),
        sa.Column("birth_year", sa.String(32), nullable=False),
        sa.Column("death_year", sa.String(32), nullable=False),
        sa.Column("achievements", sa.Text(), nullable=False),
        sa.Column("institutions", sa.JSON(), nullable=False))
    op.create_index("ix_persons_name", "persons", ["name"], unique=True)
    op.create_index("ix_persons_hometown", "persons", ["hometown"])
    op.create_table("person_fields",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field", sa.String(128), nullable=False),
        sa.UniqueConstraint("person_id", "field", name="uq_person_field"))
    op.create_index("ix_person_fields_person_id", "person_fields", ["person_id"])
    op.create_index("ix_person_fields_field", "person_fields", ["field"])
    op.create_table("relations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("from_name", sa.String(128), nullable=False),
        sa.Column("to_name", sa.String(128), nullable=False),
        sa.Column("rel_type", sa.String(32), nullable=False),
        sa.UniqueConstraint("from_name", "to_name", "rel_type", name="uq_relation"))
    op.create_index("ix_relation_from", "relations", ["from_name", "rel_type"])
    op.create_index("ix_relations_rel_type", "relations", ["rel_type"])
    op.create_table("conversations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("turn_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False))
    op.create_table("messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.String(64), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_yuan", sa.Float(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_table("feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("message_seq", sa.Integer(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_feedback_conversation_id", "feedback", ["conversation_id"])
    op.create_table("ingest_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("path", sa.String(512), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("chunks", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True))
    op.create_table("eval_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_eval_runs_name", "eval_runs", ["name"])
    op.create_table("index_state",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("revision", sa.BigInteger(), nullable=False, server_default=sa.text("0")))
    op.execute("INSERT INTO index_state (name, revision) VALUES ('knowledge', 0)")

    if op.get_context().dialect.name == "postgresql":
        dimension = op.get_context().config.attributes.get("vector_dimension", 512)
        if not isinstance(dimension, int) or not 1 <= dimension <= 2000:
            raise ValueError("pgvector HNSW 维度必须为 1 到 2000")
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.execute("""CREATE TABLE vector_collections (
            collection VARCHAR(128) PRIMARY KEY,
            signature VARCHAR(64) NOT NULL DEFAULT ''
        )""")
        op.execute(f"""CREATE TABLE vector_chunks (
            collection VARCHAR(128) NOT NULL,
            doc_id VARCHAR(160) NOT NULL,
            text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            person_name VARCHAR(128) NOT NULL DEFAULT '',
            embedding vector({dimension}) NOT NULL,
            PRIMARY KEY (collection, doc_id)
        )""")
        op.execute("CREATE INDEX ix_vector_chunks_person ON vector_chunks (collection, person_name)")
        op.execute("""CREATE INDEX ix_vector_chunks_hnsw ON vector_chunks
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)""")


def downgrade() -> None:
    if op.get_context().dialect.name == "postgresql":
        op.drop_table("vector_chunks")
        op.drop_table("vector_collections")
        # vector 扩展可能被同一数据库的其他应用使用，因此不删除扩展。
    for table in ("index_state", "eval_runs", "ingest_jobs", "feedback", "messages",
                  "conversations", "relations", "person_fields", "persons", "chunks", "documents"):
        op.drop_table(table)
