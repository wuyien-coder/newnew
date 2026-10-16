"""中间件：trace_id 贯穿日志与响应头，请求耗时打点。"""
from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware

from ..core.logging import get_logger, new_trace_id, trace_id_var
from ..core.metrics import metrics

log = get_logger("mathkb.access")


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or new_trace_id()
        token = trace_id_var.set(trace_id)
        t0 = time.perf_counter()
        metrics.inc("http.requests")
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            metrics.observe("http.latency_ms", elapsed)
            log.info("%s %s -> %s in %.2fms", request.method, request.url.path, status, elapsed)
            trace_id_var.reset(token)
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Elapsed-Ms"] = str(elapsed)
        return response
