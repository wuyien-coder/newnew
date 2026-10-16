"""Token 用量与成本台账：线程安全累计，供 API 与评测报告读取。"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

from .base import LLMResult, Usage


@dataclass
class Ledger:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_yuan: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class UsageTracker:
    def __init__(self, price_input_per_k: float = 0.0, price_output_per_k: float = 0.0):
        self._lock = threading.Lock()
        self.totals = Ledger()
        self.price_input_per_k = price_input_per_k
        self.price_output_per_k = price_output_per_k

    def record(self, result: LLMResult) -> Usage:
        usage = result.usage
        cost = (usage.prompt_tokens / 1000.0 * self.price_input_per_k
                + usage.completion_tokens / 1000.0 * self.price_output_per_k)
        with self._lock:
            t = self.totals
            t.calls += 1
            t.prompt_tokens += usage.prompt_tokens
            t.completion_tokens += usage.completion_tokens
            t.reasoning_tokens += usage.reasoning_tokens
            t.cached_tokens += usage.cached_tokens
            t.cost_yuan = round(t.cost_yuan + cost, 6)
        return usage

    def add_snapshot(self, snapshot: dict) -> None:
        with self._lock:
            for key, value in snapshot.items():
                if hasattr(self.totals, key):
                    setattr(self.totals, key, getattr(self.totals, key) + value)

    def snapshot(self) -> dict:
        with self._lock:
            return self.totals.as_dict()

    def last_cost(self, usage: Usage) -> float:
        return round(usage.prompt_tokens / 1000.0 * self.price_input_per_k
                     + usage.completion_tokens / 1000.0 * self.price_output_per_k, 6)
