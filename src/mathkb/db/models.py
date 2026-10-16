"""领域模型：知识库、人物图谱、会话、反馈、评测与摄取任务。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def now() -> datetime:
    return datetime.now()


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_name: Mapped[str] = mapped_column(String(128), index=True)
    source: Mapped[str] = mapped_column(String(255), default="")
    page: Mapped[str] = mapped_column(String(64), default="")
    file_path: Mapped[str] = mapped_column(String(512), unique=True)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="ready")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_pos"),
        Index("ix_chunk_person_section", "person_name", "section_title"),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    person_name: Mapped[str] = mapped_column(String(128), index=True)
    section_title: Mapped[str] = mapped_column(String(255), default="正文")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    hometown: Mapped[str] = mapped_column(String(128), default="", index=True)
    birth_year: Mapped[str] = mapped_column(String(32), default="")
    death_year: Mapped[str] = mapped_column(String(32), default="")
    achievements: Mapped[str] = mapped_column(Text, default="")
    institutions: Mapped[list] = mapped_column(JSON, default=list)


class PersonField(Base):
    __tablename__ = "person_fields"
    __table_args__ = (UniqueConstraint("person_id", "field", name="uq_person_field"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), index=True)
    field: Mapped[str] = mapped_column(String(128), index=True)


class Relation(Base):
    __tablename__ = "relations"
    __table_args__ = (
        UniqueConstraint("from_name", "to_name", "rel_type", name="uq_relation"),
        Index("ix_relation_from", "from_name", "rel_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_name: Mapped[str] = mapped_column(String(128))
    to_name: Mapped[str] = mapped_column(String(128))
    rel_type: Mapped[str] = mapped_column(String(32), index=True)  # teacher|student|advisor


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.seq",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_yuan: Mapped[float] = mapped_column(Float, default=0.0)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    agent_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    message_seq: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[int] = mapped_column(Integer)  # 1 有帮助 / -1 无帮助
    comment: Mapped[str] = mapped_column(Text, default="")
    query: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|running|done|failed
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
