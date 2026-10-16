"""问答服务：编排 Agent、落库会话与引用、汇总用量成本。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from copy import copy
from typing import Iterator, Optional

from sqlalchemy import select, delete, func

from ..agent.graph import build_agent
from ..agent.state import AgentState
from ..core.exceptions import IndexNotReadyError, InvalidInputError
from ..core.logging import get_logger
from ..core.metrics import metrics
from ..db.models import Conversation, Message
from ..rag.prompts import PROMPT_VERSION, SUGGEST
from .container import AppContext

log = get_logger(__name__)

MAX_HISTORY_MESSAGES = 8


@dataclass
class ChatAnswer:
    session_id: str
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: str = "low"
    entities: list[str] = field(default_factory=list)
    mode: str = "rag"
    engine: str = "sequential"
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    cost_yuan: float = 0.0
    notes: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    agent_trace: list[dict] = field(default_factory=list)
    agent_stop: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "question": self.question, "answer": self.answer,
            "citations": self.citations, "confidence": self.confidence,
            "entities": self.entities, "mode": self.mode, "engine": self.engine,
            "timings_ms": self.timings_ms, "retrieval": self.retrieval, "usage": self.usage,
            "cost_yuan": self.cost_yuan, "notes": self.notes,
            "prompt_version": self.prompt_version,
            "agent_trace": self.agent_trace, "agent_stop": self.agent_stop,
        }


class ChatService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._agents: dict[str, object] = {}

    # ---- 会话 ----
    def list_sessions(self, limit: int = 30) -> list[dict]:
        with self.ctx.db.session() as session:
            conversations = session.scalars(select(Conversation).where(
                Conversation.turn_count > 0).order_by(Conversation.updated_at.desc()).limit(limit)).all()
            ids = [c.id for c in conversations]
            rows = session.execute(select(Message.conversation_id, Message.content).where(
                Message.conversation_id.in_(ids), Message.seq == 0)).all() if ids else []
            titles = dict(rows)
        return [{"id": c.id, "title": titles.get(c.id, "未命名会话")[:60],
                 "turn_count": c.turn_count, "updated_at": c.updated_at.isoformat()} for c in conversations]

    def delete_session(self, session_id: str) -> bool:
        from ..db.models import Feedback
        with self.ctx.db.session() as session:
            conversation = session.get(Conversation, session_id)
            if conversation is None:
                return False
            session.execute(delete(Feedback).where(Feedback.conversation_id == session_id))
            session.delete(conversation)
        return True

    def history(self, session_id: Optional[str]) -> list[dict]:
        if not session_id:
            return []
        with self.ctx.db.session() as session:
            rows = session.scalars(
                select(Message).where(Message.conversation_id == session_id).order_by(Message.seq)
            ).all()
        return [{"role": r.role, "content": r.content} for r in rows][-MAX_HISTORY_MESSAGES:]

    def _ensure_conversation(self, session_id: str) -> None:
        with self.ctx.db.session() as session:
            if not session.get(Conversation, session_id):
                session.add(Conversation(id=session_id))

    def _persist(self, session_id: str, question: str, answer: ChatAnswer) -> None:
        usage = answer.usage or {}
        with self.ctx.db.session() as session:
            conv = session.scalar(select(Conversation).where(Conversation.id == session_id).with_for_update())
            if conv is None:
                raise InvalidInputError("会话已删除，请新建对话")
            seq = session.scalar(select(func.count(Message.id)).where(
                Message.conversation_id == session_id)) or 0
            session.add(Message(conversation_id=session_id, seq=seq, role="user",
                                content=question))
            session.add(Message(conversation_id=session_id, seq=seq + 1, role="assistant",
                                content=answer.answer,
                                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                                completion_tokens=int(usage.get("completion_tokens", 0)),
                                cost_yuan=answer.cost_yuan, citations=answer.citations,
                                agent_metadata={"engine": answer.engine, "agent_trace": answer.agent_trace,
                                                "agent_stop": answer.agent_stop}))
            conv.turn_count = (conv.turn_count or 0) + 1

    def _request_context(self, top_k=None, overrides=None):
        base = self.ctx.snapshot()
        if base.retriever is None:
            raise IndexNotReadyError("索引未构建，请先构建索引")
        from ..llm.usage import UsageTracker
        retriever = copy(base.retriever)
        retriever.cfg = _apply_overrides(retriever.cfg, top_k, overrides)
        if not retriever.cfg.use_dense and not retriever.cfg.use_bm25:
            raise InvalidInputError("至少启用一路文本检索")
        tracker = UsageTracker(self.ctx.settings.llm.price_input, self.ctx.settings.llm.price_output)
        return replace(base, retriever=retriever, tracker=tracker)

    def _finish(self, ctx, state, session_id, question, mode, engine, started):
        context, retrieval = state.get("context"), state.get("retrieval")
        usage = ctx.tracker.snapshot()
        answer = ChatAnswer(
            session_id=session_id, question=question, answer=state.get("answer", ""),
            citations=[c.__dict__ for c in context.citations] if context else [],
            confidence=context.confidence if context else "low",
            entities=state.get("entities") or [], mode=mode, engine=engine,
            timings_ms=state.get("timings_ms", {}),
            retrieval={"pool_size": retrieval.pool_size, "hits": len(retrieval.hits),
                       "person_filter": retrieval.person_filter,
                       "config": _cfg_dict(ctx.retriever.cfg)} if retrieval else {},
            usage=usage, cost_yuan=usage.get("cost_yuan", 0.0), notes=state.get("notes", []),
            agent_trace=state.get("agent_trace", []), agent_stop=state.get("agent_stop", ""),
        )
        answer.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
        self._persist(session_id, question, answer)
        self.ctx.tracker.add_snapshot(usage)
        metrics.inc("chat.requests")
        metrics.observe("chat.latency_ms", answer.timings_ms["total"])
        return answer

    def ask(self, question, session_id=None, mode="rag", engine="auto", top_k=None, overrides=None):
        if not question or not question.strip():
            raise InvalidInputError("问题不能为空")
        question = question.strip()
        started = time.perf_counter()
        ctx = self._request_context(top_k, overrides)
        session_id = session_id or f"sess-{uuid.uuid4().hex}"
        self._ensure_conversation(session_id)
        agent = build_agent(ctx, mode=mode, engine=engine)
        state = agent.run({"question": question, "history": self.history(session_id),
                           "timings_ms": {}, "notes": []})
        return self._finish(ctx, state, session_id, question, mode, agent.name, started)

    def stream(self, question, session_id=None, mode="rag", engine="auto", overrides=None):
        """Retrieve before generation, then relay actual upstream model deltas."""
        from ..agent import nodes
        from ..llm.base import LLMResult
        if not question or not question.strip():
            raise InvalidInputError("问题不能为空")
        started = time.perf_counter()
        question = question.strip()
        ctx = self._request_context(overrides=overrides)
        session_id = session_id or f"sess-{uuid.uuid4().hex}"
        self._ensure_conversation(session_id)
        agent = build_agent(ctx, mode=mode, engine=engine)
        state = agent.prepare({"question": question, "history": self.history(session_id),
                               "timings_ms": {}, "notes": []})
        context = state["context"]
        yield {"event": "retrieval", "data": {
            "entities": state.get("entities", []), "confidence": context.confidence,
            "citations": [c.__dict__ for c in context.citations],
            "timings_ms": state.get("timings_ms", {}), "engine": agent.name,
            "agent_trace": state.get("agent_trace", []), "agent_stop": state.get("agent_stop", ""),
        }}
        pieces = []
        t0 = time.perf_counter()
        for event in ctx.llm.stream_events(nodes.generation_messages(ctx, state)):
            if isinstance(event, LLMResult):
                ctx.tracker.record(event)
            elif event:
                if not pieces:
                    state["timings_ms"]["first_token"] = round((time.perf_counter()-started)*1000, 2)
                pieces.append(event)
                yield {"event": "token", "data": {"text": event}}
        state["answer"] = "".join(pieces)
        state["timings_ms"]["generate"] = round((time.perf_counter()-t0)*1000, 2)
        state = nodes.verify(ctx, state)
        answer = self._finish(ctx, state, session_id, question, mode, agent.name, started)
        yield {"event": "done", "data": answer.to_dict()}

    def session_detail(self, session_id: str) -> list[dict]:
        with self.ctx.db.session() as session:
            rows = session.scalars(
                select(Message).where(Message.conversation_id == session_id).order_by(Message.seq)
            ).all()
        return [
            {"seq": m.seq, "role": m.role, "content": m.content,
             "citations": m.citations or [], "cost_yuan": m.cost_yuan,
             "agent_metadata": m.agent_metadata or {},
             "created_at": m.created_at.isoformat()}
            for m in rows
        ]

    def add_feedback(self, session_id: str, message_seq: int, rating: int,
                     comment: str = "") -> None:
        if rating not in (-1, 1):
            raise InvalidInputError("rating 只接受 1 或 -1")
        from ..db.models import Feedback

        with self.ctx.db.session() as session:
            last_user = session.scalars(
                select(Message).where(Message.conversation_id == session_id,
                                      Message.role == "user").order_by(Message.seq.desc())
            ).first()
            session.add(Feedback(conversation_id=session_id, message_seq=message_seq,
                                 rating=rating, comment=comment,
                                 query=last_user.content if last_user else ""))
        metrics.inc("chat.feedback")

    def suggest(self, session_id: str) -> list[str]:
        history = self.history(session_id)
        if len(history) < 2:
            return []
        prompt = SUGGEST.format(question=history[-2]["content"], answer=history[-1]["content"][:300])
        result = self.ctx.llm.chat([{"role": "user", "content": prompt}])
        self.ctx.tracker.record(result)
        return [line.strip("•- ") for line in result.text.splitlines() if line.strip()][:3]


def _apply_overrides(cfg, top_k: Optional[int], overrides: Optional[dict]):
    from dataclasses import replace

    if not overrides and not top_k:
        return cfg
    changes = dict(overrides or {})
    if top_k:
        changes["top_k"] = top_k
    return replace(cfg, **changes)


def _cfg_dict(cfg) -> dict:
    from dataclasses import asdict

    return asdict(cfg)
