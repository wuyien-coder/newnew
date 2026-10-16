"""FastAPI 依赖：从 app.state 取容器与服务，附带可选鉴权。"""
from __future__ import annotations

from fastapi import Depends, Header, Request

from ..core.exceptions import UnauthorizedError
from ..services.chat_service import ChatService
from ..services.container import AppContext
from ..services.ingest_service import IngestService


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_chat(request: Request) -> ChatService:
    return request.app.state.chat


def get_ingest(request: Request) -> IngestService:
    return request.app.state.ingest


def require_token(request: Request, authorization: str = Header(default="")) -> None:
    """配置 MATHKB_SERVER__API_TOKEN 后启用 Bearer 鉴权；为空表示本地开发免鉴权。"""
    token = request.app.state.ctx.settings.server.api_token
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise UnauthorizedError("缺少或错误的 Authorization 头")


ChatDep = Depends(get_chat)
AuthDep = Depends(require_token)
