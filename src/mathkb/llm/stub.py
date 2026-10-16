"""离线 Stub Provider：不联网、不花钱，但产出与真实链路同结构的回答。

用途有三：
1. CI/单测里跑通「检索 → 引用 → 生成 → 落库」全链路；
2. 没有 API Key 时也能演示；
3. 作为对照基线，量化真实 LLM 相对抽取式回答的增益。
它不是语言模型：只做「按引用摘录 + 标注编号」，因此指标会明显低于真实生成。
"""
from __future__ import annotations

import re
import time
from typing import Iterator

from .base import LLMClient, LLMResult, Message, Usage

_REF_BLOCK = re.compile(r"【引用 \[(\d+)]】.*?内容: (.*?)(?=\n\n【引用|\Z)", re.S)
_QUESTION = re.compile(r"【用户问题】\s*(.+)$", re.S)


def _extract_refs(prompt: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in _REF_BLOCK.finditer(prompt)]


class StubLLMClient(LLMClient):
    name = "stub"

    def __init__(self, model: str = "stub-extractive", max_sentences: int = 2):
        self.model = model
        self.max_sentences = max_sentences

    def chat(self, messages: list[Message], *, temperature=None, max_tokens=None) -> LLMResult:
        t0 = time.perf_counter()
        prompt = "\n\n".join(m.get("content", "") for m in messages)
        question_match = _QUESTION.search(prompt)
        question = question_match.group(1).strip() if question_match else ""

        if "检索式问题" in prompt:          # rewrite 任务
            return self._as_result(question or prompt, t0)
        if "检索通道" in prompt:            # route 任务
            label = "graph" if any(
                kw in question for kw in ["师从", "导师", "学生", "同门", "籍贯", "人数", "哪些人"]
            ) else "text"
            return self._as_result(label, t0)
        if "继续追问" in prompt:            # suggest 任务
            lines = [f"• {question}的相关人物有哪些？", f"• {question}的主要贡献是什么？"]
            return self._as_result("\n".join(lines), t0)

        refs = _extract_refs(prompt)
        if not refs:
            text = "根据现有资料，无法确定。"
        else:
            picked = []
            for idx, content in refs[:3]:
                sentences = [s for s in re.split(r"(?<=[。！？])", content) if s.strip()]
                picked.append(f"{ ''.join(sentences[: self.max_sentences])[:180] } [{idx}]")
            text = (f"围绕「{question[:40]}」，资料摘录如下：\n"
                    + "\n".join(f"- {p}" for p in picked)
                    + "\n（注：以上由离线抽取式 Provider 生成，仅用于链路验证。）")
        return self._as_result(text, t0)

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        for token in re.findall(r"\S+|", result.text):
            if token:
                yield token

    def _as_result(self, text: str, t0: float, prompt_chars: int = 0) -> LLMResult:
        # 离线摘录不调用模型，因此不记录模型token和费用。
        usage = Usage()  # 离线摘录没有真实模型用量，避免伪造token或计费
        return LLMResult(
            text=text, model=self.model, provider="stub", usage=usage,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
