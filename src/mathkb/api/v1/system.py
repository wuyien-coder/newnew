"""健康检查与指标。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from ...core.metrics import metrics
from ...services.container import AppContext
from ..deps import get_ctx

router = APIRouter(tags=["system"])


@router.get("/health")
def health(ctx: AppContext = Depends(get_ctx)) -> dict:
    return {
        "status": "ok" if ctx.ready else "warming",
        "chunks": ctx.index_stats.get("chunks", 0),
        "persons": ctx.index_stats.get("persons", 0),
        "embedder": ctx.index_stats.get("embedder"),
        "llm_provider": ctx.llm.name,
    }


@router.get("/metrics")
def prometheus() -> Response:
    return Response(metrics.exposition(), media_type="text/plain; version=0.0.4")
