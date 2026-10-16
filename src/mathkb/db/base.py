"""SQLite 离线模式与 PostgreSQL 生产模式共用的会话及索引版本记录。"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import BigInteger, String, create_engine, inspect, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from ..core.logging import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class IndexState(Base):
    """摄取提交后递增；不同 API/Celery 进程以此使本地 BM25/图谱快照失效。"""

    __tablename__ = "index_state"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


def normalize_database_url(url: str) -> str:
    """PostgreSQL URL 默认选用 psycopg 3，与生产依赖保持一致。"""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class Database:
    def __init__(self, url: str, echo: bool = False):
        url = normalize_database_url(url)
        kwargs: dict = {"echo": echo, "future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            db_path = url.split("///", 1)[-1] if "///" in url else ":memory:"
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            kwargs["connect_args"] = {"check_same_thread": False}
            if db_path == ":memory:":
                kwargs["poolclass"] = StaticPool
        else:
            kwargs.update(pool_size=10, max_overflow=20, pool_recycle=1800)
        self.engine = create_engine(url, **kwargs)
        self._factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        # Older local demos used create_all without Alembic. Keep those SQLite
        # histories readable; PostgreSQL upgrades remain exclusively migrations.
        if self.engine.dialect.name == "sqlite":
            columns = {c["name"] for c in inspect(self.engine).get_columns("messages")}
            if "agent_metadata" not in columns:
                with self.engine.begin() as connection:
                    connection.execute(text("ALTER TABLE messages ADD COLUMN agent_metadata JSON NOT NULL DEFAULT '{}'"))
        log.info("数据库表结构就绪: %s", self.engine.url.render_as_string(hide_password=True))

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def session_factory(self) -> sessionmaker:
        return self._factory

    def _ensure_index_state(self, session: Session) -> None:
        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            stmt = pg_insert(IndexState).values(name="knowledge", revision=0)
        elif dialect == "sqlite":
            stmt = sqlite_insert(IndexState).values(name="knowledge", revision=0)
        else:
            raise ValueError("仅支持 SQLite 和 PostgreSQL 数据库")
        session.execute(stmt.on_conflict_do_nothing(index_elements=["name"]))

    def index_revision(self) -> int:
        with self.session() as session:
            revision = session.scalar(select(IndexState.revision).where(IndexState.name == "knowledge"))
            if revision is None:
                self._ensure_index_state(session)
                revision = session.scalar(select(IndexState.revision).where(IndexState.name == "knowledge"))
            return int(revision or 0)

    def bump_index_revision(self) -> int:
        """SQL 原子递增；并发摄取不会丢失 revision 更新。"""
        with self.session() as session:
            self._ensure_index_state(session)
            revision = session.scalar(
                update(IndexState).where(IndexState.name == "knowledge")
                .values(revision=IndexState.revision + 1).returning(IndexState.revision)
            )
            return int(revision)

    @contextmanager
    def index_build_lock(self, collection: str) -> Iterator[None]:
        """串行化 PostgreSQL 多进程的整次读取/编码/替换，防止旧构建覆盖新构建。"""
        if self.engine.dialect.name != "postgresql":
            yield
            return
        params = {"key": "mathkb-build:" + collection}
        with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("SELECT pg_advisory_lock(hashtext(:key))"), params)
            try:
                yield
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(hashtext(:key))"), params)
