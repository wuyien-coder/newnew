from __future__ import annotations

from copy import copy
from dataclasses import replace
import json

from mathkb.agent.graph import build_agent
from mathkb.llm.base import LLMResult, Usage
from mathkb.llm.usage import UsageTracker
from mathkb.rag.chunking import Document, chunk_document, parse_document
from mathkb.rag.kg_store import GraphStore
from mathkb.rag.vector_store import MemoryVectorStore
from mathkb.services.chat_service import ChatService


class MeasuredLLM:
    name = "test"

    def __init__(self):
        self.started = False

    def chat(self, messages):
        return LLMResult("答案 [1]", "test", "test", Usage(prompt_tokens=10, completion_tokens=3))

    def stream_events(self, messages):
        self.started = True
        yield "第一段\n"
        yield "第二段 [1]"
        yield LLMResult("第一段\n第二段 [1]", "test", "test", Usage(10, 3))


def test_usage_is_per_request_not_running_total(ctx):
    llm = MeasuredLLM()
    local = replace(ctx, llm=llm, tracker=UsageTracker())
    svc = ChatService(local)
    first = svc.ask("华罗庚的贡献", mode="rag")
    second = svc.ask("陈省身的贡献", mode="rag")
    assert first.usage["prompt_tokens"] == second.usage["prompt_tokens"] == 10
    assert first.usage["completion_tokens"] == second.usage["completion_tokens"] == 3
    assert local.tracker.snapshot()["prompt_tokens"] == 20


def test_stream_retrieval_precedes_generation_and_preserves_deltas(ctx):
    llm = MeasuredLLM()
    svc = ChatService(replace(ctx, llm=llm, tracker=UsageTracker()))
    stream = svc.stream("华罗庚的贡献", mode="rag")
    assert next(stream)["event"] == "retrieval"
    assert llm.started is False
    events = list(stream)
    text = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert text == "第一段\n第二段 [1]"
    assert events[-1]["data"]["answer"] == text
    assert events[-1]["data"]["usage"]["completion_tokens"] == 3
    assert svc.session_detail(events[-1]["data"]["session_id"])[-1]["content"] == text


def test_overrides_are_isolated_and_graph_switch_is_honored(ctx):
    svc = ChatService(ctx)
    base = ctx.retriever.cfg
    local = svc._request_context(overrides={"use_graph": False})
    assert local.retriever is not ctx.retriever
    assert ctx.retriever.cfg is base and base.use_graph is True
    answer = svc.ask("苏步青师从谁？", mode="agent", engine="sequential", overrides={"use_graph": False})
    assert answer.retrieval["config"]["use_graph"] is False
    assert not any(c["section"] == "图谱三元组" for c in answer.citations)


def test_empty_graph_and_text_workflow_terminates(ctx):
    class EmptyKeywords:
        def search(self, *args, **kwargs):
            return []
    retriever = copy(ctx.retriever)
    retriever.vectors, retriever.keywords = MemoryVectorStore(), EmptyKeywords()
    graph = GraphStore([], [])
    local = replace(ctx, retriever=retriever, graph=graph)
    for engine in ("sequential", "langgraph"):
        try:
            agent = build_agent(local, mode="agent", engine=engine)
        except ImportError:
            continue
        state = agent.prepare({"question": "不知名人物的导师有谁", "history": [], "notes": [], "timings_ms": {}})
        assert state["context"].citations == []


def test_chunks_have_content_identity_and_enforce_hard_limit(tmp_path):
    doc = Document("甲", sections=[("正文", "无" * 3000)])
    pieces = chunk_document(doc, max_chunk=1200)
    assert all(len(c.text) <= 1200 for c in pieces)
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("人物：甲\n====\n同一段正文。", encoding="utf-8")
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    assert chunk_document(parse_document(first))[0].chunk_id != chunk_document(parse_document(second))[0].chunk_id
    old = chunk_document(parse_document(first))[0].chunk_id
    first.write_text("人物：甲\n====\n正文已修改。", encoding="utf-8")
    assert chunk_document(parse_document(first))[0].chunk_id != old


def test_upload_ingests_and_exposes_durable_job(tmp_path):
    from fastapi.testclient import TestClient
    from mathkb.main import create_app
    from tests.conftest import make_settings
    with TestClient(create_app(make_settings(tmp_path))) as client:
        content = "人物：演示学者\n来源：测试资料\n====\n演示学者研究组合数学和图论。".encode()
        response = client.post("/api/v1/documents/upload", files={"file": ("example.txt", content, "text/plain")})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        jobs = client.get("/api/v1/documents/jobs").json()
        assert next(j for j in jobs if j["id"] == job_id)["status"] == "done"
        documents = client.get("/api/v1/documents/list").json()
        assert any(d["person_name"] == "演示学者" and d["chunks"] > 0 for d in documents)
        body = client.post("/api/v1/search", json={"query": "演示学者组合数学"}).json()
        assert any("组合数学" in hit["text"] for hit in body["hits"])
        assert client.post("/api/v1/documents/upload", files={"file": ("bad.txt", b"\xff", "text/plain")}).status_code == 400
        assert client.post("/api/v1/documents/upload", files={"file": ("bad.exe", b"data", "text/plain")}).status_code == 400


def test_sse_endpoint_returns_complete_done_payload(client):
    response = client.post("/api/v1/chat/stream", json={"question": "陈省身的贡献", "overrides": {"use_graph": False}})
    assert response.status_code == 200
    records = response.text.split("\n\n")
    done = next(r for r in records if r.startswith("event: done"))
    body = json.loads(done.split("data: ", 1)[1])
    assert body["answer"] and body["session_id"] and body["citations"]
    assert body["retrieval"]["config"]["use_graph"] is False


def test_named_graph_route_preserves_biography_evidence(ctx, monkeypatch):
    from types import SimpleNamespace
    from mathkb.agent import nodes
    local = replace(ctx, llm=SimpleNamespace(name="openai"))
    monkeypatch.setattr(nodes, "_call", lambda *args: "graph")
    state = nodes.route(local, {"question": "苏步青师从谁？", "notes": []})
    assert "苏步青" in state["entities"]
    assert state["route"] == "both"
