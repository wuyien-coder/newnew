"""上下文装配：引用编号、置信度提示、图谱三元组注入。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import RetrievalResult

CONFIDENCE_NOTE = {
    "high": "",
    "medium": "\n\n提示：参考资料相关度中等，对不确定之处请显式说明。",
    "low": "\n\n警告：参考资料与问题相关度较低，若信息不足请如实回答「根据现有资料无法确定」。",
}


@dataclass
class Citation:
    index: int
    chunk_id: str
    person: str
    section: str
    source: str
    page: str
    score: float
    snippet: str


@dataclass
class AssembledContext:
    prompt_block: str
    citations: list[Citation] = field(default_factory=list)
    confidence: str = "low"
    confidence_basis: str = "none"

    @property
    def context_chars(self) -> int:
        return len(self.prompt_block)


def _display_score(hit) -> tuple[str, float]:
    """标注分数来自哪一路，避免把 BM25 的 26 分当成 0~1 的相关度读。"""
    if hit.rerank_score is not None:
        return "精排", hit.rerank_score
    if hit.dense_score is not None:
        return "余弦", hit.dense_score
    if hit.lexical_score is not None:
        return "BM25", hit.lexical_score
    return "融合", hit.fused_score


def assemble(result: RetrievalResult) -> AssembledContext:
    parts: list[str] = []
    citations: list[Citation] = []

    for i, hit in enumerate(result.hits, 1):
        meta = hit.metadata or {}
        label, value = _display_score(hit)
        person = meta.get("person_name", "未知")
        section = meta.get("section_title", "")
        page = meta.get("page", "")
        source = meta.get("source", "")
        header = f"【引用 [{i}]】({label}: {value:.3f} | 召回: {'+'.join(hit.sources)})"
        trail = " | ".join(x for x in [f"人物: {person}" if person else "",
                                       f"章节: {section}" if section and section != "正文" else "",
                                       f"页码: {page}" if page else "",
                                       f"来源: {source}" if source else ""] if x)
        parts.append(f"{header}\n{trail}\n内容: {hit.text}")
        citations.append(Citation(index=i, chunk_id=hit.doc_id, person=person, section=section,
                                  source=source, page=str(page), score=round(value, 4),
                                  snippet=hit.text[:160]))

    for line in result.graph_lines:
        parts.append(line)

    block = "\n\n".join(parts) if parts else "（未检索到资料）"
    return AssembledContext(prompt_block=block, citations=citations,
                            confidence=result.confidence,
                            confidence_basis=result.confidence_basis)
