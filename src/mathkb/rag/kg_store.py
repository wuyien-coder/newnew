"""知识图谱检索：内存 networkx 图 + 结构化三元组输出。

图数据来自人物结构化表（SQLAlchemy），这里只接收普通记录，保持 RAG 层与 ORM 解耦。
生产可换成 Neo4j：实现同样的 `GraphProvider` 协议即可，`graph_search` 的返回契约不变。
"""
from __future__ import annotations

import re
from dataclasses import field as dc_field
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from ..core.logging import get_logger

log = get_logger(__name__)

REL_LABELS = {
    "teacher": "师从",
    "student": "培养学生",
    "advisor": "导师",
}

# 聚合类问句里的通用词，切词时优先剥掉，避免「研究」「哪些人」误命中领域名
_GENERIC_RE = re.compile(
    r"(哪些|多少人|几个人|从事|研究|领域|方向|籍贯|哪里|什么|请问|介绍一下|有|的|是|在|和|与|了|吗|呢|[，。？！、\s])"
)


def _query_tokens(query: str) -> list[str]:
    stripped = _GENERIC_RE.sub(" ", query or "")
    parts = [p for p in stripped.split() if len(p) >= 2]
    # 「江苏籍」→「江苏」，「数论方向」→「数论」：地名/领域名在索引里不带这些后缀
    expanded = [p for part in parts for p in (part, re.sub(r"[籍省市]$", "", part)) if len(p) >= 2]
    return list(dict.fromkeys(expanded))


@dataclass
class GraphPerson:
    name: str
    hometown: str = ""
    birth_year: str = ""
    death_year: str = ""
    fields: list[str] = dc_field(default_factory=list)
    institutions: list[str] = dc_field(default_factory=list)
    achievements: str = ""


class GraphStore:
    def __init__(self, persons: list[GraphPerson], relations: list[tuple[str, str, str]]):
        self.persons: dict[str, GraphPerson] = {p.name: p for p in persons}
        self.graph = nx.DiGraph()
        for p in persons:
            self.graph.add_node(p.name)
        for src, dst, rel in relations:
            if src in self.persons and dst in self.persons:
                self.graph.add_edge(src, dst, rel=rel)
        self._field_index: dict[str, list[str]] = {}
        self._hometown_index: dict[str, list[str]] = {}
        for p in persons:
            for f in p.fields:
                self._field_index.setdefault(f, []).append(p.name)
            if p.hometown:
                self._hometown_index.setdefault(p.hometown, []).append(p.name)
        log.info("图谱加载: %d 节点 / %d 边 / %d 领域 / %d 籍贯",
                 self.graph.number_of_nodes(), self.graph.number_of_edges(),
                 len(self._field_index), len(self._hometown_index))

    @property
    def names(self) -> list[str]:
        return list(self.persons)

    def profile_text(self, name: str) -> str:
        p = self.persons.get(name)
        if not p:
            return ""
        bits = [f"{name}"]
        if p.birth_year:
            bits.append(f"生年 {p.birth_year}")
        if p.hometown:
            bits.append(f"籍贯 {p.hometown}")
        if p.fields:
            bits.append("研究领域 " + "、".join(p.fields))
        if p.institutions:
            bits.append("机构 " + "、".join(p.institutions[:3]))
        if p.achievements:
            bits.append(p.achievements[:120])
        return "；".join(bits)

    def neighbors(self, name: str, rel: Optional[str] = None) -> list[tuple[str, str, str]]:
        """返回 (方向, 关系, 对端) 三元组，方向 ∈ {出, 入}。"""
        if name not in self.graph:
            return []
        out = []
        for _, dst, data in self.graph.out_edges(name, data=True):
            if rel and data["rel"] != rel:
                continue
            out.append(("出", REL_LABELS.get(data["rel"], data["rel"]), dst))
        for src, _, data in self.graph.in_edges(name, data=True):
            if rel and data["rel"] != rel:
                continue
            out.append(("入", REL_LABELS.get(data["rel"], data["rel"]), src))
        return out

    def by_field(self, keyword: str, limit: int = 20) -> list[str]:
        """双向匹配：领域名出现在问句中（「数论」⊂「研究数论的有哪些人」），
        以及问句切出的候选词出现在领域名中（「数论」⊂「解析数论」）。"""
        hits: list[str] = []
        tokens = _query_tokens(keyword)
        for field_name, names in self._field_index.items():
            if not field_name:
                continue
            if field_name in keyword or any(t in field_name for t in tokens):
                hits.extend(names)
        return list(dict.fromkeys(hits))[:limit]

    def by_hometown(self, keyword: str, limit: int = 20) -> list[str]:
        hits: list[str] = []
        for place, names in self._hometown_index.items():
            if place and (place in keyword or any(t in place for t in _query_tokens(keyword))):
                hits.extend(names)
        return list(dict.fromkeys(hits))[:limit]

    def path(self, a: str, b: str, max_hops: int = 4) -> list[str]:
        undirected = self.graph.to_undirected()
        if a not in undirected or b not in undirected:
            return []
        try:
            return list(nx.shortest_path(undirected, a, b))
        except nx.NetworkXNoPath:
            return []

    def triples_for(self, names: list[str]) -> list[str]:
        """把实体的一跳邻居渲染成可注入 prompt 的三元组文本。"""
        lines: list[str] = []
        for name in names:
            profile = self.profile_text(name)
            if profile:
                lines.append(f"[图谱-实体] {profile}")
            for direction, rel, other in self.neighbors(name):
                lines.append(f"[图谱-关系] {name} --{rel}({'→' if direction == '出' else '←'}) {other}")
        return lines
