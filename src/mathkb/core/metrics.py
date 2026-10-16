"""进程内指标：计数 + 分位数摘要，输出 Prometheus 文本格式（无第三方依赖）。"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _Series:
    values: list[float] = field(default_factory=list)

    def observe(self, v: float) -> None:
        self.values.append(v)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        return float(sum(self.values))

    def quantile(self, q: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]


class Metrics:
    """线程安全的轻量指标注册表（多进程部署时应换成 Redis / PushGateway）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, _Series] = defaultdict(_Series)
        self._gauges: dict[str, float] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name].observe(value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._counters[name].observe(value)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": {
                    k: {"count": s.count, "sum": round(s.total, 4),
                        "p50": round(s.quantile(0.5), 4), "p95": round(s.quantile(0.95), 4)}
                    for k, s in self._counters.items()
                },
                "gauges": dict(self._gauges),
            }

    def exposition(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name, series in self._counters.items():
                metric = name.replace(".", "_")
                lines.append(f"# TYPE {metric} summary")
                lines.append(f"{metric}_count {series.count}")
                lines.append(f"{metric}_sum {series.total}")
                for q in (0.5, 0.95):
                    lines.append(f'{metric}{{quantile="{q}"}} {series.quantile(q)}')
            for name, value in self._gauges.items():
                metric = name.replace(".", "_")
                lines.append(f"# TYPE {metric} gauge")
                lines.append(f"{metric} {value}")
            return "\n".join(lines) + "\n"


metrics = Metrics()
