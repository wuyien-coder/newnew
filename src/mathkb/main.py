"""FastAPI 应用工厂：lifespan 内完成建库、种子与索引构建。"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from .api.v1.router import build_root_router
from .core.config import Settings, get_settings
from .core.exceptions import AppError
from .core.logging import get_logger, setup_logging, trace_id_var
from .services.chat_service import ChatService
from .services.container import AppContext
from .services.ingest_service import IngestService

log = get_logger("mathkb.app")


def create_app(settings: Settings | None = None, build: bool = True,
               force_seed: bool = False) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        ctx = AppContext.create(settings, seed=True)
        if build:
            app.state.index = ctx.build_index(force=force_seed)
        app.state.ctx = ctx
        app.state.chat = ChatService(ctx)
        app.state.ingest = IngestService(ctx)
        if settings.queue.provider == "local":
            app.state.ingest.recover_interrupted()
        log.info("应用就绪，启动耗时 %.1fms", (time.perf_counter() - t0) * 1000)
        try:
            yield
        finally:
            ctx.close()

    app = FastAPI(
        title="数学人物知识分析",
        description="GraphRAG + 混合检索 + Agent 编排的领域知识问答平台（数学人物知识分析系统）",
        version="2.2.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id", "X-Elapsed-Ms"],
    )
    from .api.middleware import TraceMiddleware

    app.add_middleware(TraceMiddleware)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        log.warning("业务异常 %s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": exc.to_dict(), "trace_id": trace_id_var.get()},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("未处理异常")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "服务暂不可用，请稍后重试"},
                     "trace_id": trace_id_var.get()},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        # Credentials can appear anywhere in an invalid request; never echo input.
        errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    app.include_router(build_root_router())

    # 本地演示客户端（生产由 nginx 直接托管 web/ 静态目录）
    demo_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if demo_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/web", StaticFiles(directory=str(demo_dir), html=True), name="web")

        @app.get("/", include_in_schema=False)
        async def home() -> RedirectResponse:
            return RedirectResponse("/web/")

    return app


app = create_app()
