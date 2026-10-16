from dataclasses import replace
from copy import copy
import json

from mathkb.agent.graph import build_agent
from mathkb.llm.base import LLMResult, Usage
from mathkb.llm.usage import UsageTracker
from mathkb.services.chat_service import ChatService


class ScriptedModel:
    name = "scripted"

    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        text = next(self.actions)
        return LLMResult(text, "scripted", self.name, Usage(10, 3))

    def stream_events(self, messages):
        self.messages.append(messages)
        yield "答案 [1]"
        yield LLMResult("答案 [1]", "scripted", self.name, Usage(10, 3))


def action(tool, **arguments):
    return json.dumps({"action": tool, "arguments": arguments}, ensure_ascii=False)


def local_context(ctx, actions, max_steps=4):
    settings = ctx.settings.model_copy(deep=True)
    settings.agent.max_steps = max_steps
    return replace(ctx, settings=settings, llm=ScriptedModel(actions), tracker=UsageTracker())


def test_react_uses_observation_to_choose_a_second_tool(ctx):
    local = local_context(ctx, [action("graph_neighbors", name="苏步青", relation="teacher", direction="out"),
        action("text_search", query="苏步青的求学经历与老师", person="苏步青"), action("finish"), "根据资料 [1]"])
    result = ChatService(local).ask("苏步青师从谁？", mode="agent")
    assert result.engine == "react"
    assert [s["tool"] for s in result.agent_trace] == ["graph_neighbors", "text_search"]
    assert result.agent_stop == "finished"
    assert result.citations and result.usage["calls"] == 4
    saved = ChatService(local).session_detail(result.session_id)[-1]["agent_metadata"]
    assert saved["agent_trace"] == result.agent_trace and saved["engine"] == "react"
    assert any("relations" in m["content"] for m in local.llm.messages[1] if m["role"] == "user")


def test_react_repeated_action_is_rejected_and_budget_terminates(ctx):
    request = action("person_profile", name="华罗庚")
    local = local_context(ctx, [request]*3, max_steps=3)
    state = build_agent(local).prepare({"question": "介绍华罗庚", "history": [], "notes": []})
    assert state["agent_stop"] == "step_limit"
    assert len(state["agent_trace"]) == 3
    assert state["agent_trace"][1]["status"] == "error"
    assert len(state["retrieval"].hits) == 1


def test_react_invalid_json_and_unknown_tools_can_be_repaired(ctx):
    local = local_context(ctx, ['not json', action("delete_database"),
        action("text_search", query="华罗庚数论"), action("finish")])
    state = build_agent(local).prepare({"question": "华罗庚贡献", "history": []})
    assert [s["status"] for s in state["agent_trace"]] == ["error", "error", "ok"]
    assert state["agent_stop"] == "finished"
    assert state["context"].citations


def test_react_graph_disabled_and_argument_bounds_are_enforced(ctx):
    local = local_context(ctx, [])
    runner = build_agent(local)
    from pydantic import ValidationError
    import pytest
    with pytest.raises(ValidationError):
        runner._tool("find_people", {"keyword": "数论", "limit": 30000})
    local.retriever = copy(local.retriever)
    local.retriever.cfg = replace(local.retriever.cfg, use_graph=False)
    with pytest.raises(ValueError, match="图谱工具已关闭"):
        runner._tool("person_profile", {"name": "华罗庚"})
    local.llm = ScriptedModel([action("text_search", query="华罗庚数论"), action("finish")])
    result = runner.prepare({"question": "华罗庚数论", "history": []})
    assert "graph_neighbors" not in local.llm.messages[0][0]["content"]
    assert [step["tool"] for step in result["agent_trace"]] == ["text_search"]


def test_react_paths_respect_hop_limit_and_people_are_paginated(ctx):
    runner = build_agent(local_context(ctx, []))
    data, hits, names = runner._tool("find_people", {"keyword": "几何", "limit": 1})
    assert len(data["names"]) <= 1 and data["total_matches"] >= len(data["names"])
    edges = list(ctx.graph.graph.edges())
    source, target = edges[0]
    data, _, _ = runner._tool("graph_path", {"source": source, "target": target, "max_hops": 1})
    assert data["within_hops"] and len(data["path"]) == 2
    assert "无向" in data["semantics"]
    from mathkb.rag.kg_store import GraphPerson, GraphStore
    runner.ctx.graph = GraphStore([GraphPerson(n) for n in ("甲", "乙", "丙")],
                                  [("甲", "乙", "teacher"), ("乙", "丙", "teacher")])
    data, _, _ = runner._tool("graph_path", {"source": "甲", "target": "丙", "max_hops": 1})
    assert not data["within_hops"] and data["path"] == []


def test_react_stream_has_trace_citations_and_combined_usage(ctx):
    local = local_context(ctx, [action("text_search", query="华罗庚数论"), action("finish")])
    events = list(ChatService(local).stream("华罗庚数论贡献", mode="agent"))
    assert events[0]["event"] == "retrieval"
    assert events[0]["data"]["agent_trace"][0]["tool"] == "text_search"
    done = events[-1]["data"]
    assert done["engine"] == "react" and done["answer"] == "答案 [1]"
    assert done["usage"]["calls"] == 3
    assert done["citations"]


def test_relation_direction_and_empty_results_are_not_positive_evidence(ctx):
    from mathkb.rag.kg_store import GraphPerson, GraphStore
    local = local_context(ctx, [])
    local.graph = GraphStore([GraphPerson(n) for n in ("学生甲", "老师乙")],
                             [("学生甲", "老师乙", "teacher")])
    runner = build_agent(local)
    empty, hits, _ = runner._tool("graph_neighbors", {"name": "老师乙", "relation": "teacher", "direction": "out"})
    assert empty["found"] and empty["count"] == 0 and hits == []
    found, hits, _ = runner._tool("graph_neighbors", {"name": "学生甲", "relation": "teacher", "direction": "out"})
    assert found["count"] == 1 and len(hits) == 1
    assert found["relations"][0]["statement"] == "学生甲师从老师乙"
    incoming, _, _ = runner._tool("graph_neighbors", {"name": "老师乙", "relation": "teacher", "direction": "in"})
    assert incoming["relations"][0]["source"] == "学生甲"
    assert incoming["relations"][0]["statement"] == "学生甲师从老师乙"


def test_empty_relation_trace_preserves_scope_without_citing_empty_result(ctx):
    from mathkb.rag.kg_store import GraphPerson, GraphStore
    from mathkb.agent.nodes import generation_messages
    local = local_context(ctx, [action("graph_neighbors", name="老师乙", relation="teacher", direction="out"), action("finish")])
    local.graph = GraphStore([GraphPerson("老师乙")], [])
    state = build_agent(local).prepare({"question": "老师乙师从谁？", "history": []})
    trace = state["agent_trace"][0]
    assert trace["status"] == "ok" and trace["result_count"] == 0
    assert "0 条" in trace["result_summary"]
    assert state["context"].citations == []
    assert '"direction": "out"' in generation_messages(local, state)[-1]["content"]
