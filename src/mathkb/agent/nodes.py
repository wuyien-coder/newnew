"""Agent 节点实现：改写、路由、检索、图谱聚合、生成、自检。

节点是纯函数（读取 AppContext + AgentState，返回状态增量），
所以同一份节点逻辑既能跑在自研顺序执行器上，也能挂到 LangGraph StateGraph 上。
"""
from __future__ import annotations

import time
import json
from typing import TYPE_CHECKING

from ..core.exceptions import InvalidInputError
from ..core.logging import get_logger
from ..rag import prompts
from ..rag.context import assemble
from ..llm.base import Message

if TYPE_CHECKING:  # 避免与 services.container 形成循环导入
    from ..services.container import AppContext
    from .state import AgentState

log = get_logger(__name__)

GRAPH_HINTS = ("师从", "导师", "学生", "同门", "籍贯", "哪里人", "哪些人", "多少人",
               "几个", "之间", "传承", "属于", "关系网", "有谁")


def _call(ctx: "AppContext", messages: list[Message], label: str, state: "AgentState") -> str:
    t0 = time.perf_counter()
    result = ctx.llm.chat(messages)
    ctx.tracker.record(result)
    state.setdefault("timings_ms", {})[label] = round((time.perf_counter() - t0) * 1000, 2)
    return result.text.strip()


def rewrite(ctx: "AppContext", state: "AgentState") -> "AgentState":
    """多轮指代消解 + 检索式改写：只在有历史时调用，省一次 LLM 往返。"""
    question = state["question"]
    history = state.get("history") or []
    if not history:
        state["rewritten"] = question
        state.setdefault("notes", []).append("rewrite: skipped(无历史)")
        return state
    if ctx.llm.name == "stub":
        # Offline mode resolves simple references without pretending to run an LLM.
        if ctx.linker and any(word in question for word in ("他", "这位", "该数学家")):
            for item in reversed(history):
                names = ctx.linker.link(item["content"]) if item["role"] == "user" else []
                if len(names) == 1:
                    import re
                    question = re.sub(r"这位数学家|这位|该数学家|他", names[0], question)
                    break
        state["rewritten"] = question
        state.setdefault("notes", []).append("rewrite: offline rule")
        return state
    history_text = "\n".join(f"{h['role']}: {h['content'][:200]}" for h in history[-6:])
    prompt = prompts.REWRITE.format(history=history_text, question=question)
    out = _call(ctx, [{"role": "user", "content": prompt}], "rewrite", state)
    state["rewritten"] = out or question
    state.setdefault("notes", []).append(f"rewrite: {question} → {state['rewritten']}")
    return state


def route(ctx: "AppContext", state: "AgentState") -> "AgentState":
    """意图路由：文本检索 / 图谱查询 / 两者。默认关键词规则，零成本零延迟。

    规则优先、LLM 兜底：命中图谱线索词 → graph；出现两个人名（对比/关系类）→ both；
    其余 → text。用真实 provider 时再让 LLM 复核一次，规则与模型不一致以模型为准。
    """
    query = state.get("rewritten") or state["question"]
    entities = state.get("entities") or (ctx.linker.link(query) if ctx.linker else [])
    state["entities"] = entities
    rule_hit = "text"
    if any(hint in query for hint in GRAPH_HINTS):
        rule_hit = "both" if entities else "graph"
    elif len(entities) >= 2:
        rule_hit = "both"

    if ctx.llm.name == "stub":
        state["route"] = rule_hit
        state.setdefault("notes", []).append(f"route(rule): {rule_hit}")
        return state
    prompt = prompts.ROUTE.format(question=query)
    out = _call(ctx, [{"role": "user", "content": prompt}], "route", state).lower().strip()
    llm_hit = out.split("\n")[0].strip()
    state["route"] = llm_hit if llm_hit in ("text", "graph", "both") else rule_hit
    # A graph may contain only students, not a person's own teacher. Preserve
    # biography evidence for named graph questions even when the router says graph.
    if state["route"] == "graph" and entities:
        state["route"] = "both"
    state.setdefault("notes", []).append(f"route(rule={rule_hit}, llm={llm_hit or 'n/a'})")
    return state


def retrieve(ctx: "AppContext", state: "AgentState") -> "AgentState":
    query = state.get("rewritten") or state["question"]
    from ..core.cache import RetrievalCache
    cache = RetrievalCache(ctx.settings.cache)
    cache_key = cache.key(ctx, query)
    t0 = time.perf_counter()
    result = cache.get(cache_key)
    if result is None:
        result = ctx.retriever.retrieve(query)
        cache.put(cache_key, result)
        state.setdefault("notes", []).append("cache: miss" if cache.enabled else "cache: disabled")
    else:
        result.timings_ms = {"cache": round((time.perf_counter() - t0) * 1000, 2)}
        state.setdefault("notes", []).append("cache: hit")
    state["retrieval"] = result
    state["entities"] = result.entities
    state.setdefault("timings_ms", {}).update(result.timings_ms)
    state.setdefault("notes", []).append(
        f"retrieve: pool={result.pool_size} kept={len(result.hits)} "
        f"conf={result.confidence} person={result.person_filter}"
    )
    return state


def graph_aggregate(ctx: "AppContext", state: "AgentState") -> "AgentState":
    """图谱分支：实体一跳关系 + 领域/籍贯聚合 + 两点路径，结果并入检索结果。"""
    from ..rag.retriever import RetrievalResult
    from ..rag.vector_store import Hit

    query = state.get("rewritten") or state["question"]
    graph = ctx.graph
    retrieval = state.get("retrieval")
    if not ctx.retriever.cfg.use_graph:
        if retrieval is None:
            return retrieve(ctx, state)
        state.setdefault("notes", []).append("graph: disabled")
        return state
    if graph is None:
        state.setdefault("notes", []).append("graph: 图谱未加载")
        state["retrieval"] = retrieval or RetrievalResult(query=query, hits=[])
        return state

    lines: list[str] = list(retrieval.graph_lines if retrieval else []) or []
    entities = list(state.get("entities") or (retrieval.entities if retrieval else []))
    if not entities and ctx.linker is not None:
        entities = ctx.linker.link(query)
    if not entities:
        entities = graph.by_field(query, limit=10) + graph.by_hometown(query, limit=10)
        if entities:
            lines.append(f"[图谱-候选] {query} 相关人物: {', '.join(entities[:10])}")

    for name in entities[:3]:
        lines.extend(graph.triples_for([name]))
        other = _second_name(query, entities)
        if other:
            path = graph.path(name, other)
            if len(path) > 2:
                lines.append(f"[图谱-路径] {' → '.join(path)}")

    synthetic = [
        Hit(doc_id=f"graph:{i}", text=line, score=1.0, source="graph",
            metadata={"person_name": entities[min(i, len(entities) - 1)] if entities else "",
                      "section_title": "图谱三元组"})
        for i, line in enumerate(dict.fromkeys(lines))
    ]
    if retrieval is None:
        retrieval = RetrievalResult(query=query, hits=synthetic, entities=entities,
                                    graph_lines=lines, confidence="medium" if synthetic else "low")
    else:
        retrieval.graph_lines = lines
        retrieval.entities = entities
        retrieval.hits = synthetic[:3] + retrieval.hits
    state["retrieval"] = retrieval
    state["entities"] = entities
    state.setdefault("notes", []).append(f"graph: {len(synthetic)} 条三元组")
    return state


def build_context(ctx: "AppContext", state: "AgentState") -> "AgentState":
    retrieval = state.get("retrieval")
    if retrieval is None:
        raise InvalidInputError("缺少检索结果，无法构造上下文")
    state["context"] = assemble(retrieval)
    return state


def generation_messages(ctx: "AppContext", state: "AgentState") -> list[Message]:
    context = state["context"]
    question = state.get("rewritten") or state["question"]
    from ..rag.context import CONFIDENCE_NOTE

    note = CONFIDENCE_NOTE.get(context.confidence, "")
    messages: list[Message] = [{"role": "system", "content": prompts.SYSTEM}]
    summary = state.get("summary")
    if summary:
        messages.append({"role": "system", "content": f"【历史摘要】{summary}"})
    messages.extend(state.get("history") or [])
    messages.append({
        "role": "user",
        "content": f"【参考资料】\n{context.prompt_block}{note}"
                   + ("\n\n【工具查询摘要，仅表示本次查询范围】\n" + json.dumps([
                       {k: step[k] for k in ("tool", "arguments", "result_count", "result_summary") if k in step}
                       for step in state.get("agent_trace", [])], ensure_ascii=False)
                      if state.get("agent_trace") else "")
                   + f"\n\n【用户问题】\n{question}",
    })
    return messages


def generate(ctx: "AppContext", state: "AgentState") -> "AgentState":
    context = state["context"]
    state["answer"] = _call(ctx, generation_messages(ctx, state), "generate", state)
    state.setdefault("notes", []).append(
        f"generate: ctx={context.context_chars}字 conf={context.confidence}"
    )
    return state


def verify(ctx: "AppContext", state: "AgentState") -> "AgentState":
    """引用自检：答案声称了引用但上下文里没有对应编号时补标注释（不重写答案）。"""
    import re

    answer = state.get("answer", "")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    available = {c.index for c in state["context"].citations} if state.get("context") else set()
    dangling = sorted(cited - available)
    if dangling:
        state.setdefault("notes", []).append(f"verify: 引用编号越界 {dangling}")
    else:
        state.setdefault("notes", []).append("verify: 引用编号全部可溯源")
    return state


def _second_name(query: str, entities: list[str]) -> str:
    return entities[1] if len(entities) > 1 and any(n in query for n in entities) else ""
