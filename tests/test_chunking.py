from __future__ import annotations

from pathlib import Path

from mathkb.rag.chunking import chunk_document, parse_document


def test_parse_header_and_sections(sample_bio: Path):
    doc = parse_document(sample_bio)
    assert doc.person_name
    assert doc.meta["birth_year"].startswith("19")
    assert len(doc.sections) >= 2, "章节感知切分应识别出多个小节"


def test_chunks_respect_size_and_carry_prefix(sample_bio: Path):
    doc = parse_document(sample_bio)
    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=80, max_chunk=900)
    assert chunks
    assert all(len(c.text) <= 1200 for c in chunks), "分块必须受硬上限约束"
    assert all(c.text.startswith(f"【{doc.person_name} |") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk_id 必须唯一"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks))), "全局序号连续"


def test_overlap_keeps_sentences_whole(sample_bio: Path):
    doc = parse_document(sample_bio)
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=120, max_chunk=800)
    long_enough = [c for c in chunks if len(c.text) > 200]
    assert long_enough
    # 相邻块的重叠区应以完整句子开头，而不是半句
    for a, b in zip(long_enough, long_enough[1:]):
        body_b = b.text.split("】", 1)[-1].strip()
        assert body_b and body_b[0] not in "，、；:,", f"重叠区出现半句: {body_b[:20]}"
