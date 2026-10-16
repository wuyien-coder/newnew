"""会话与反馈接口：多轮历史、追问建议、答案评价回流。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, HTTPException

from ...services.chat_service import ChatService
from ..deps import ChatDep
from ..deps import require_token
from ..schemas import FeedbackRequest

router = APIRouter(prefix="/sessions", tags=["session"], dependencies=[Depends(require_token)])


@router.get("")
def list_sessions(limit: int = Query(default=30, ge=1, le=200),
                  chat_svc: ChatService = ChatDep) -> dict:
    return {"sessions": chat_svc.list_sessions(limit)}


@router.delete("/{session_id}")
def delete_session(session_id: str, chat_svc: ChatService = ChatDep) -> dict:
    if not chat_svc.delete_session(session_id):
        raise HTTPException(404, "会话不存在")
    return {"status": "deleted"}


@router.get("/{session_id}")
def get_session(session_id: str, chat_svc: ChatService = ChatDep) -> dict:
    return {"session_id": session_id, "messages": chat_svc.session_detail(session_id)}


@router.get("/{session_id}/suggestions")
def suggestions(session_id: str, chat_svc: ChatService = ChatDep) -> dict:
    return {"suggestions": chat_svc.suggest(session_id)}


@router.post("/{session_id}/feedback", status_code=201)
def feedback(session_id: str, req: FeedbackRequest, chat_svc: ChatService = ChatDep) -> dict:
    if session_id != req.session_id:
        raise HTTPException(400, "会话编号不一致")
    chat_svc.add_feedback(session_id=req.session_id, message_seq=req.message_seq,
                          rating=req.rating, comment=req.comment)
    return {"status": "recorded"}
