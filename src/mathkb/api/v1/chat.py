"""问答接口：非流式 + SSE 流式。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...core.exceptions import AppError, ProviderError
from ...llm.connection import ModelConnection, request_model
from ...services.chat_service import ChatService
from ..deps import ChatDep, require_token
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_token)])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, chat_svc: ChatService = ChatDep) -> ChatResponse:
    with request_model(req.model_connection) as llm:
        answer = chat_svc.ask(
            question=req.question,
            session_id=req.session_id,
            mode=req.mode,
            engine=req.engine,
            overrides={k: v for k, v in req.overrides.model_dump().items() if v is not None},
            llm=llm,
        )
    return ChatResponse(**answer.to_dict())


@router.post("/stream")
def chat_stream(req: ChatRequest, chat_svc: ChatService = ChatDep) -> StreamingResponse:
    """SSE：event=retrieval|token|done，前端可先渲染引用面板再逐字输出。"""

    def event_source():
        try:
            with request_model(req.model_connection) as llm:
                for event in chat_svc.stream(req.question, session_id=req.session_id,
                                             mode=req.mode, engine=req.engine, llm=llm,
                                             overrides={k: v for k, v in req.overrides.model_dump().items() if v is not None}):
                    yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        except AppError as exc:
            payload = {"message": exc.message, "code": exc.code, "detail": exc.detail}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception:
            from ...core.logging import get_logger
            get_logger(__name__).exception("流式问答中断")
            yield 'event: error\ndata: {"message":"回答中断，请检查服务配置或稍后重试"}\n\n'

    return StreamingResponse(event_source(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/model/check")
def check_model(connection: ModelConnection) -> dict:
    with request_model(connection, timeout=25) as llm:
        result = llm.chat([{"role": "user", "content": "只回复 OK"}], max_tokens=32)
        if not result.text.strip():
            raise ProviderError("模型返回了空正文，请检查模型名称或换用非推理模型。")
        return {"ok": True, "model": llm.model, "latency_ms": result.latency_ms}
