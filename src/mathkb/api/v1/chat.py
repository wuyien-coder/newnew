"""问答接口：非流式 + SSE 流式。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...services.chat_service import ChatService
from ..deps import ChatDep, require_token
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_token)])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, chat_svc: ChatService = ChatDep) -> ChatResponse:
    answer = chat_svc.ask(
        question=req.question,
        session_id=req.session_id,
        mode=req.mode,
        engine=req.engine,
        overrides={k: v for k, v in req.overrides.model_dump().items() if v is not None},
    )
    return ChatResponse(**answer.to_dict())


@router.post("/stream")
def chat_stream(req: ChatRequest, chat_svc: ChatService = ChatDep) -> StreamingResponse:
    """SSE：event=retrieval|token|done，前端可先渲染引用面板再逐字输出。"""

    def event_source():
        try:
            for event in chat_svc.stream(req.question, session_id=req.session_id,
                                         mode=req.mode, engine=req.engine,
                                         overrides={k: v for k, v in req.overrides.model_dump().items() if v is not None}):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        except Exception:
            from ...core.logging import get_logger
            get_logger(__name__).exception("流式问答中断")
            yield 'event: error\ndata: {"message":"回答中断，请检查服务配置或稍后重试"}\n\n'

    return StreamingResponse(event_source(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
