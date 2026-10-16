"""Alembic 入口：生产启动前执行 alembic upgrade head。"""
from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from mathkb.core.config import Settings
from mathkb.db import models  # noqa: F401 -- 注册领域表供后续 autogenerate 审查
from mathkb.db.base import Base, normalize_database_url

config = context.config
settings = Settings()
url = normalize_database_url(settings.database_url)
config.attributes["vector_dimension"] = settings.embedding.dim
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    # vector 类型使用显式 SQL 迁移管理，autogenerate 不应把它们误判成废弃表。
    return not (type_ == "table" and reflected and name in {"vector_chunks", "vector_collections"})


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # SQLite 演示模式也可跑同一套迁移，目录由 Settings.ensure_dirs 创建。
    settings.ensure_dirs()
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          compare_type=True, include_object=include_object)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
