"""验证显式初始迁移与运行期 ORM schema 一致。"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from mathkb.core.config import Settings
from mathkb.db.base import Base
from mathkb.services.container import AppContext


def test_alembic_sqlite_upgrade_supports_create_schema_false(tmp_path, monkeypatch):
    alembic = pytest.importorskip("alembic")
    from alembic import command
    from alembic.config import Config

    assert alembic
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    monkeypatch.setenv("MATHKB_DATABASE_URL", url)
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "head")
    engine = create_engine(url)
    inspector = inspect(engine)
    assert set(Base.metadata.tables) <= set(inspector.get_table_names())
    for name, table in Base.metadata.tables.items():
        assert set(table.columns.keys()) == {column["name"] for column in inspector.get_columns(name)}
    settings = Settings(_env_file=None, database_url=url, create_schema=False,
                        keyword={"persist_dir": str(tmp_path / "bm25")},
                        vector={"provider": "memory"}, llm={"provider": "stub"},
                        embedding={"provider": "hash"})
    ctx = AppContext.create(settings, seed=False)
    assert ctx.db.index_revision() == 0
    assert ctx.bump_revision() == 1
    engine.dispose()


def test_alembic_postgres_offline_contains_reviewable_vector_schema(monkeypatch):
    pytest.importorskip("alembic")
    from alembic import command
    from alembic.config import Config

    monkeypatch.setenv("MATHKB_DATABASE_URL", "postgresql+psycopg://localhost/mathkb")
    monkeypatch.setenv("MATHKB_EMBEDDING__DIM", "768")
    output = StringIO()
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"), output_buffer=output)
    command.upgrade(config, "head", sql=True)
    sql = output.getvalue()
    assert "CREATE TABLE documents" in sql
    assert "CREATE TABLE index_state" in sql
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "embedding vector(768)" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql


def test_old_sqlite_demo_can_read_messages_after_metadata_upgrade(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text
    from mathkb.db.base import Database
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    monkeypatch.setenv("MATHKB_DATABASE_URL", url)
    command.upgrade(Config(str(Path(__file__).resolve().parents[1] / "alembic.ini")), "0001_initial")
    database = Database(url)
    with database.engine.begin() as connection:
        connection.execute(text("INSERT INTO messages (conversation_id,seq,role,content,prompt_tokens,completion_tokens,cost_yuan,citations,created_at) VALUES ('old',0,'assistant','旧回答',0,0,0,'[]',CURRENT_TIMESTAMP)"))
    database.create_all()
    database.create_all()
    with database.engine.connect() as connection:
        row = connection.execute(text("SELECT content,agent_metadata FROM messages")).one()
        assert row[0] == "旧回答" and row[1] == '{}'
    database.engine.dispose()
