"""传记文档解析与章节感知分块。

与朴素定长切块的差异（论文里做过消融，这里保留同样的开关）：
1. 章节边界优先切分，保持叙事完整；
2. 块头注入「人物 | 章节」上下文前缀，提升 dense/BM25 命中；
3. 重叠区按句子边界截断，避免半句。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_META_KEYS = {
    "人物": "person_name",
    "生年": "birth_year",
    "逝世": "death_year",
    "研究方向": "research_field",
    "来源": "source",
    "页码": "page",
    "奖项": "award",
}

_SECTION_RE = re.compile(
    r"^\s*(?:"
    r"[\(（][一二三四五六七八九十百\d]+[\)）]"
    r"|[一二三四五六七八九十百]+\s*[、.．,，]\s*"
    r"|第\s*[一二三四五六七八九十\d]+\s*[章节部分]"
    r"|\d{4}\s*[-~～—]\s*\d{4}\s*[,，、\s]"
    r")[^\n]{0,30}\s*$"
)
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    person_name: str = ""
    section_title: str = "正文"
    chunk_index: int = 0
    birth_year: str = ""
    death_year: str = ""
    research_field: str = ""
    source: str = ""
    page: str = ""
    file_path: str = ""

    def metadata(self) -> dict:
        return {
            "person_name": self.person_name,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "birth_year": self.birth_year,
            "death_year": self.death_year,
            "research_field": self.research_field,
            "source": self.source,
            "page": self.page,
            "file_path": self.file_path,
        }


@dataclass
class Document:
    person_name: str
    meta: dict = field(default_factory=dict)
    sections: list[tuple[str, str]] = field(default_factory=list)  # (title, text)


def _parse_header(path: Path) -> tuple[dict, int]:
    """读取文件头元数据块，返回 (meta, 正文起始行号偏移)。"""
    meta = {v: "" for v in _META_KEYS.values()}
    with path.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    body_start = 0
    for i, raw in enumerate(lines[:12]):
        line = raw.strip()
        if line.startswith("="):
            body_start = i + 1
            break
        if line.startswith("【") and line.endswith("】"):
            meta["person_name"] = line.strip("【】")
            continue
        for prefix, key in _META_KEYS.items():
            if line.startswith(f"{prefix}：") or line.startswith(f"{prefix}:"):
                meta[key] = re.split(r"[：:]", line, maxsplit=1)[1].strip()
                body_start = i + 1
                break
    return meta, body_start


def parse_document(path: Path) -> Document:
    meta, body_start = _parse_header(path)
    with path.open(encoding="utf-8") as fh:
        body = fh.readlines()[body_start:]

    sections: list[tuple[str, list[str]]] = []
    title = "正文"
    for raw in body:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("====") and line.endswith("===="):
            title = line.strip("=").strip() or title
            sections.append((title, []))
            continue
        if len(line) <= 40 and _SECTION_RE.match(line):
            title = line
            sections.append((title, []))
            continue
        if not sections:
            sections.append((title, []))
        sections[-1][1].append(raw)

    doc_sections = [
        (t, re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip())
        for t, parts in sections
        if "".join(parts).strip()
    ]
    name = meta.get("person_name") or path.stem
    meta["file_path"] = str(path.resolve())
    return Document(person_name=name, meta=meta, sections=doc_sections)


def _split_by_sentence(text: str, size: int, overlap: int, max_size: int) -> list[str]:
    """句子级滑动窗口：重叠区落在句子边界上，max_size 为硬上限。"""
    sentences = [part[i:i+max_size] for part in _SENT_SPLIT_RE.split(text) if part.strip()
                 for i in range(0, len(part), max_size)]
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        if cur and cur_len + len(sent) > size:
            out.append("".join(cur).strip())
            cur, cur_len = _overlap_tail(cur, overlap)
        if cur and cur_len + len(sent) > max_size:
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += len(sent)
        if cur_len >= max_size:
            out.append("".join(cur).strip())
            cur, cur_len = [], 0
    if cur:
        out.append("".join(cur).strip())
    return [c for c in out if c]


def _overlap_tail(window: list[str], overlap: int) -> tuple[list[str], int]:
    kept: list[str] = []
    length = 0
    for sent in reversed(window):
        if length + len(sent) > overlap:
            break
        kept.insert(0, sent)
        length += len(sent)
    return kept, length


def chunk_document(doc: Document, chunk_size: int = 500, chunk_overlap: int = 100,
                   max_chunk: int = 1200, use_prefix: bool = True) -> list[Chunk]:
    if chunk_size <= 0 or max_chunk < 32 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("分块参数无效")
    chunks: list[Chunk] = []
    index = 0
    for section_title, text in doc.sections:
        if not text:
            continue
        prefix = f"【{doc.person_name} | {section_title}】\n" if use_prefix else ""
        prefix = prefix[:max_chunk//2]
        body_limit = max_chunk - len(prefix)
        pieces = _split_by_sentence(text, min(chunk_size, body_limit),
                                    min(chunk_overlap, max(0, body_limit//4)), body_limit)
        for piece in pieces:
            body = prefix + piece
            identity = f"{doc.meta.get('file_path', '')}#{doc.person_name}#{section_title}#{index}#{body}"
            digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
            chunks.append(
                Chunk(
                    chunk_id=f"chunk:{digest}",
                    text=body,
                    person_name=doc.person_name,
                    section_title=section_title,
                    chunk_index=index,
                    birth_year=str(doc.meta.get("birth_year", "")),
                    death_year=str(doc.meta.get("death_year", "")),
                    research_field=str(doc.meta.get("research_field", "")),
                    source=str(doc.meta.get("source", "")),
                    page=str(doc.meta.get("page", "")),
                    file_path=str(doc.meta.get("file_path", "")),
                )
            )
            index += 1
    return chunks


def load_directory(data_dir: Path, **kwargs) -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(p for p in Path(data_dir).rglob("*.txt")):
        doc = parse_document(path)
        for chunk in chunk_document(doc, **kwargs):
            chunk.file_path = str(path)
            out.append(chunk)
    return out
