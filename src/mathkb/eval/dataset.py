"""评测数据集加载。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalCase:
    id: str
    query: str
    gold_person: str
    gold_source: str


@dataclass
class MCQCase:
    id: str
    person: str
    question: str
    options: list[str]
    answer: int
    gold_field: str


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_retrieval(path: Path) -> list[RetrievalCase]:
    return [RetrievalCase(r["id"], r["query"], r["gold_person"], r.get("gold_source", ""))
            for r in _read_jsonl(path)]


def load_mcq(path: Path) -> list[MCQCase]:
    return [MCQCase(r["id"], r["person"], r["question"], r["options"], r["answer"],
                    r.get("gold_field", "")) for r in _read_jsonl(path)]
