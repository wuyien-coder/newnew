"""实体链接：从问句中确定性地识别学人名（零 LLM 调用、零延迟）。

策略来自论文工程实践：完整显示名（含括注）优先，其次核心名；
短名（2 字）需要排除「作为更长名字子串」的误命中。
"""
from __future__ import annotations

import re
from typing import Iterable


def normalize_name(name: str) -> str:
    s = (name or "").replace(" ", "").replace("\u3000", "")
    return s.replace("（", "(").replace("）", ")").strip()


def core_name(display_name: str) -> str:
    return re.sub(r"\([^)]*\)", "", normalize_name(display_name)).strip()


class EntityLinker:
    def __init__(self, candidates: Iterable[str]):
        self._names = [n for n in candidates if n]
        self._by_len = sorted(self._names, key=len, reverse=True)
        self._core_index = {core_name(n): n for n in self._names if core_name(n)}

    def link(self, question: str, top_n: int = 3) -> list[str]:
        q_norm = normalize_name(question)
        matches: list[str] = []
        seen: set[str] = set()

        for name in self._by_len:
            if normalize_name(name) in q_norm:
                key = core_name(name)
                if not key or key in seen:
                    continue
                # 已被更长的名字覆盖时不再单独命中（「王元」不应抢「王元深」）
                if any(key != other and key in other for other in seen):
                    continue
                seen.add(key)
                matches.append(name)

        for core, name in self._core_index.items():
            if core in seen or len(core) < 2:
                continue
            if core in q_norm:
                if len(core) == 2 and any(core in s and len(s) > 2 for s in seen):
                    continue
                seen.add(core)
                matches.append(name)

        return matches[:top_n]

    def add(self, name: str) -> None:
        if name and name not in self._names:
            self._names.append(name)
            self._by_len = sorted(self._names, key=len, reverse=True)
            self._core_index.setdefault(core_name(name), name)

    def __len__(self) -> int:
        return len(self._names)
