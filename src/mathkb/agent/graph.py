"""LangGraph workflow with a preparation graph for actual token streaming."""
from __future__ import annotations
from abc import ABC, abstractmethod
from . import nodes
from .state import AgentState
from ..core.logging import get_logger
log = get_logger(__name__)

class Agent(ABC):
    name = "base"
    @abstractmethod
    def prepare(self, state: AgentState) -> AgentState: ...
    def run(self, state: AgentState) -> AgentState:
        state = self.prepare(state)
        return nodes.verify(self.ctx, nodes.generate(self.ctx, state))

class SequentialRunner(Agent):
    name = "sequential"
    def __init__(self, ctx, use_rewrite=True, use_route=True):
        self.ctx, self.use_rewrite, self.use_route = ctx, use_rewrite, use_route
    def prepare(self, state):
        if self.use_rewrite:
            state = nodes.rewrite(self.ctx, state)
        else:
            state["rewritten"] = state["question"]
        if self.use_route:
            state = nodes.route(self.ctx, state)
        else:
            state["route"] = "both"
            state["entities"] = self.ctx.linker.link(state["rewritten"]) if self.ctx.linker else []
        if state["route"] in ("text", "both"):
            state = nodes.retrieve(self.ctx, state)
        if state["route"] in ("graph", "both"):
            state = nodes.graph_aggregate(self.ctx, state)
        if state.get("retrieval") is None or not state["retrieval"].hits:
            state = nodes.retrieve(self.ctx, state)
        return nodes.build_context(self.ctx, state)

class LangGraphRunner(Agent):
    name = "langgraph"
    def __init__(self, ctx):
        self.ctx = ctx
        self.compiled = self._build(ctx, generation=True)
        self.preparation = self._build(ctx, generation=False)

    @staticmethod
    def _build(ctx, generation):
        from langgraph.graph import END, StateGraph
        g = StateGraph(AgentState)
        labels = ["rewrite", "route", "retrieve", "graph_aggregate", "build_context"]
        if generation:
            labels += ["generate", "verify"]
        for label in labels:
            fn = getattr(nodes, label)
            g.add_node(label, lambda s, fn=fn: fn(ctx, s))
        g.add_node("fallback_retrieve", lambda s: nodes.retrieve(ctx, s))
        g.set_entry_point("rewrite")
        g.add_edge("rewrite", "route")
        g.add_conditional_edges("route", lambda s: "graph" if s.get("route") == "graph" else "text",
                                {"graph": "graph_aggregate", "text": "retrieve"})
        g.add_conditional_edges("retrieve", lambda s: "both" if s.get("route") == "both" else "text",
                                {"both": "graph_aggregate", "text": "build_context"})
        g.add_conditional_edges("graph_aggregate", lambda s: "empty" if not s.get("retrieval") or not s["retrieval"].hits else "ready",
                                {"empty": "fallback_retrieve", "ready": "build_context"})
        g.add_edge("fallback_retrieve", "build_context")
        if generation:
            g.add_edge("build_context", "generate")
            g.add_edge("generate", "verify")
            g.add_edge("verify", END)
        else:
            g.add_edge("build_context", END)
        return g.compile()

    def prepare(self, state):
        return self.preparation.invoke(dict(state))

    def run(self, state):
        return self.compiled.invoke(dict(state))

def build_agent(ctx, mode="agent", engine="auto"):
    if mode == "rag":
        return SequentialRunner(ctx, use_rewrite=False, use_route=False)
    if engine == "react" or (engine == "auto" and ctx.llm.name != "stub"):
        from .react import ReactRunner
        return ReactRunner(ctx)
    if engine in ("auto", "langgraph"):
        try:
            return LangGraphRunner(ctx)
        except ImportError:
            if engine == "langgraph":
                raise
            log.info("LangGraph未安装，使用sequential执行器")
    return SequentialRunner(ctx)
