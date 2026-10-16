from __future__ import annotations

from mathkb.rag.context import CONFIDENCE_NOTE, assemble


def test_assemble_numbers_citations_and_keeps_traceability(ctx):
    result = ctx.retriever.retrieve("华罗庚在数论方面有哪些贡献", top_k=3)
    assembled = assemble(result)
    assert assembled.prompt_block.startswith("【引用 [1]】")
    assert [c.index for c in assembled.citations] == list(range(1, len(assembled.citations) + 1))
    for citation in assembled.citations:
        assert citation.chunk_id
        assert citation.snippet
        assert citation.score >= 0


def test_low_confidence_note_is_injected(ctx):
    result = ctx.retriever.retrieve("完全无关的查询词 zzqqxx", top_k=3)
    assembled = assemble(result)
    assert assembled.confidence in CONFIDENCE_NOTE


def test_graph_lines_are_included(ctx):
    result = ctx.retriever.retrieve("华罗庚师从谁")
    assembled = assemble(result)
    assert "图谱" in assembled.prompt_block or result.graph_lines
