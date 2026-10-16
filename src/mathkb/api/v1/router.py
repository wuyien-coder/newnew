"""v1 路由聚合。"""
from __future__ import annotations

from fastapi import APIRouter

from . import chat, kb, sessions, system

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(kb.router)
api_router.include_router(sessions.router)


def build_root_router() -> APIRouter:
    router = APIRouter()
    router.include_router(system.router)
    router.include_router(api_router, prefix="/api/v1")
    return router
