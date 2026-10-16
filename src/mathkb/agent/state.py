"""Agent 状态定义。"""
from __future__ import annotations

from typing import Optional, TypedDict

from ..rag.context import AssembledContext
from ..rag.retriever import RetrievalResult


class AgentState(TypedDict, total=False):
    question: str                 # 原始问题
    rewritten: str                # 改写后的检索式问题
    route: str                    # text | graph | both
    history: list[dict]           # [{role, content}]
    entities: list[str]
    retrieval: Optional[RetrievalResult]
    context: Optional[AssembledContext]
    answer: str
    notes: list[str]              # 每一步的诊断信息，随响应返回便于诊断展示
    timings_ms: dict[str, float]
    usage: dict
    agent_trace: list[dict]
    agent_stop: str
