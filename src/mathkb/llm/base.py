"""LLM Provider 协议与用量结构。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

Message = dict  # {"role": str, "content": str}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


class LLMClient(ABC):
    name = "base"

    @abstractmethod
    def chat(self, messages: list[Message], *, temperature: float | None = None,
             max_tokens: int | None = None) -> LLMResult: ...

    def stream(self, messages: list[Message], *, temperature: float | None = None,
               max_tokens: int | None = None) -> Iterator[str]:
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens).text

    def stream_events(self, messages: list[Message]) -> Iterator[str | LLMResult]:
        """Yield content deltas followed by one result carrying measured usage."""
        result = self.chat(messages)
        for i in range(0, len(result.text), 24):
            yield result.text[i:i + 24]
        yield result
