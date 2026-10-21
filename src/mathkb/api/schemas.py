"""请求/响应模型（OpenAPI schema 来源）。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..llm.connection import ModelConnection


class RetrievalOverrides(BaseModel):
    """消融开关：一次只改一个变量，便于对照实验与线上灰度。"""

    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    use_dense: Optional[bool] = None
    use_bm25: Optional[bool] = None
    use_rerank: Optional[bool] = None
    use_graph: Optional[bool] = None
    fusion: Optional[Literal["rrf", "weighted"]] = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: Optional[str] = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    mode: Literal["rag", "agent"] = "agent"
    engine: Literal["auto", "react", "sequential", "langgraph"] = "auto"
    overrides: RetrievalOverrides = Field(default_factory=RetrievalOverrides)
    model_connection: Optional[ModelConnection] = Field(default=None, repr=False)


class Citation(BaseModel):
    index: int
    chunk_id: str
    person: str
    section: str
    source: str
    page: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    confidence: str
    entities: list[str]
    mode: str
    engine: str
    citations: list[Citation]
    retrieval: dict
    timings_ms: dict
    usage: dict
    cost_yuan: float
    notes: list[str]
    prompt_version: str
    agent_trace: list[dict] = Field(default_factory=list)
    agent_stop: str = ""


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    person: Optional[str] = None
    overrides: RetrievalOverrides = Field(default_factory=RetrievalOverrides)


class SearchHit(BaseModel):
    doc_id: str
    person: str
    section: str
    page: str
    source: str
    score: float
    rerank_score: Optional[float]
    sources: list[str]
    text: str


class SearchResponse(BaseModel):
    query: str
    confidence: str
    entities: list[str]
    person_filter: Optional[str]
    pool_size: int
    timings_ms: dict
    hits: list[SearchHit]


class IngestRequest(BaseModel):
    path: str = Field(description="data/ 目录下的相对路径或绝对路径（仅限 .txt）")
    rebuild: bool = True


class FeedbackRequest(BaseModel):
    session_id: str = Field(max_length=64)
    message_seq: int = 0
    rating: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=2000)


class StatsResponse(BaseModel):
    index: dict
    llm_provider: str
    usage: dict
    config: dict
