"""结构化日志 + 请求级 trace_id 上下文。"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get(),
            "msg": record.getMessage(),
        }
        for key in ("event", "cost_ms", "extra"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True


def setup_logging(level: str = "INFO", as_json: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(TraceFilter())
    handler.setFormatter(
        JsonFormatter() if as_json
        else logging.Formatter("%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s",
                              "%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    _CONFIGURED = True


def new_trace_id(prefix: str = "req") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
