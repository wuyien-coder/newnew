"""检索与知识库接口：只查不答、语料统计、增量摄取。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Query, UploadFile, File, HTTPException

from ...core.exceptions import InvalidInputError
from ...rag.prompts import PROMPT_VERSION
from ...services.container import AppContext
from ...services.ingest_service import IngestService
from ..deps import get_ctx, get_ingest, require_token
from ..schemas import IngestRequest, SearchRequest, SearchResponse

router = APIRouter(tags=["kb"], dependencies=[Depends(require_token)])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, ctx: AppContext = Depends(get_ctx)) -> SearchResponse:
    ctx = ctx.snapshot()
    if ctx.retriever is None:
        raise InvalidInputError("索引未构建")
    from dataclasses import replace

    from copy import copy
    retriever = copy(ctx.retriever)
    base_cfg = retriever.cfg
    changes = {k: v for k, v in req.overrides.model_dump().items() if v is not None}
    if req.top_k:
        changes["top_k"] = req.top_k
    retriever.cfg = replace(base_cfg, **changes) if changes else base_cfg
    if not retriever.cfg.use_dense and not retriever.cfg.use_bm25:
        raise InvalidInputError("至少启用一路文本检索")
    result = retriever.retrieve(req.query, person=req.person)

    return SearchResponse(
        query=req.query, confidence=result.confidence, entities=result.entities,
        person_filter=result.person_filter, pool_size=result.pool_size,
        timings_ms=result.timings_ms,
        hits=[
            {
                "doc_id": h.doc_id, "person": h.metadata.get("person_name", ""),
                "section": h.metadata.get("section_title", ""),
                "page": str(h.metadata.get("page", "")),
                "source": h.metadata.get("source", ""), "score": round(h.score, 4),
                "rerank_score": h.rerank_score, "sources": h.sources,
                "text": h.text[:600],
            }
            for h in result.hits
        ],
    )


@router.get("/stats")
def stats(ctx: AppContext = Depends(get_ctx)) -> dict:
    ctx = ctx.snapshot()
    from sqlalchemy import select, func
    from ...db.models import Document
    with ctx.db.session() as session:
        document_count = session.scalar(select(func.count(Document.id))) or 0
    return {
        "index": {**ctx.index_stats, "documents": document_count},
        "llm_provider": ctx.llm.name,
        "llm_model": (getattr(ctx.llm, "model", ctx.settings.llm.model)
                      if ctx.llm.name != "stub" else "offline-extractive"),
        "dataset": "full" if Path(ctx.settings.data_dir).name == "full" else "sample",
        "usage": ctx.tracker.snapshot(),
        "config": {
            "agent": "react" if ctx.llm.name != "stub" else "sequential-offline",
            "agent_max_steps": ctx.settings.agent.max_steps,
            "embedding": ctx.settings.embedding.provider,
            "vector_store": ctx.settings.vector.provider,
            "rerank_enabled": ctx.settings.rerank.enabled,
            "kg_provider": ctx.settings.kg.provider,
            "prompt_version": PROMPT_VERSION,
            "queue": ctx.settings.queue.provider,
            "cache_enabled": ctx.settings.cache.enabled,
            "database": ctx.db.engine.dialect.name,
            "graph_engine": "NetworkX + relational tables",
        },
    }


@router.post("/documents", status_code=202)
def ingest(req: IngestRequest, background: BackgroundTasks,
           ingest_svc: IngestService = Depends(get_ingest),
           ctx: AppContext = Depends(get_ctx)) -> dict:
    """只接受项目 data 目录内的 .txt，避免接口变成任意文件读取原语。"""
    allowed_root = Path(ctx.settings.data_dir).resolve()
    candidate = (allowed_root / req.path).resolve() if not Path(req.path).is_absolute() \
        else Path(req.path).resolve()
    if not candidate.is_relative_to(allowed_root):
        raise InvalidInputError("只允许摄取 data 目录内的文件")
    if candidate.suffix.lower() not in (".txt", ".md", ".pdf") or not candidate.is_file():
        raise InvalidInputError("文件不存在或格式不受支持")
    return _submit(candidate, background, ingest_svc, ctx)


def _submit(path, background, ingest_svc, ctx):
    job_id = ingest_svc.create_job(path)
    if ctx.settings.queue.provider == "celery":
        try:
            from ...worker import celery_app
            celery_app.send_task("mathkb.ingest", args=[job_id], task_id=f"ingest-{job_id}",
                                 retry=False)
        except Exception:
            ingest_svc.fail_job(job_id, "任务未入队，请检查Redis与工作进程后重新上传")
            raise HTTPException(503, "摄取队列暂不可用") from None
    else:
        background.add_task(ingest_svc.run_background, job_id)
    return {"job_id": job_id, "status": "pending",
            "path": str(path.relative_to(Path(ctx.settings.data_dir).resolve()))}


@router.post("/documents/upload", status_code=202)
async def upload_document(background: BackgroundTasks, file: UploadFile = File(...),
                          ingest_svc: IngestService = Depends(get_ingest),
                          ctx: AppContext = Depends(get_ctx)) -> dict:
    import hashlib
    import re
    name = (file.filename or "document.txt").replace("\\", "/").split("/")[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in (".txt", ".md", ".pdf"):
        raise InvalidInputError("请上传txt、md或含文字的PDF文件")
    raw = await file.read(20 * 1024 * 1024 + 1)
    await file.close()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "文件不能超过20MB")
    if suffix == ".pdf":
        if not raw.startswith(b"%PDF-"):
            raise InvalidInputError("文件不是有效的PDF")
        text = ""
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise InvalidInputError("文本文件需使用UTF-8编码") from None
        if not text.strip() or "\x00" in text:
            raise InvalidInputError("文件正文为空或包含二进制内容")
    digest = hashlib.sha256(raw + name.encode()).hexdigest()[:24]
    folder = Path(ctx.settings.data_dir).resolve() / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w\u4e00-\u9fff-]", "_", Path(name).stem)[:60] or "document"
    path = folder / f"{stem}_{digest}{'.pdf' if suffix == '.pdf' else '.txt'}"
    if suffix == ".pdf":
        path.write_bytes(raw)
    else:
        if not any(re.match(r"^人物[：:]", line) for line in text.splitlines()[:12]):
            text = f"人物：{stem}\n来源：用户上传 {name}\n====\n{text}"
        path.write_text(text, encoding="utf-8")
    return _submit(path, background, ingest_svc, ctx)


@router.get("/documents/list")
@router.get("/documents", include_in_schema=False)
def list_documents(ctx: AppContext = Depends(get_ctx)) -> list[dict]:
    from sqlalchemy import select, func
    from ...db.models import Document, Chunk
    with ctx.db.session() as session:
        rows = session.execute(select(Document, func.count(Chunk.id)).outerjoin(
            Chunk, Chunk.document_id == Document.id).group_by(Document.id).order_by(Document.id)).all()
    return [{"id": doc.id, "person_name": doc.person_name, "source": doc.source,
             "char_count": doc.char_count, "chunks": count, "status": doc.status}
            for doc, count in rows]


@router.delete("/documents/{document_id}")
def remove_document(document_id: int, ctx: AppContext = Depends(get_ctx)) -> dict:
    from ...db.models import Document
    from sqlalchemy import select, func
    with ctx.db.session() as session:
        record = session.get(Document, document_id)
        if record is None:
            raise HTTPException(404, "文档不存在")
        count = session.scalar(select(func.count(Document.id))) or 0
        if count <= 1:
            raise HTTPException(409, "请至少保留一份知识库文档")
        session.delete(record)
    ctx.bump_revision()
    ctx.build_index()
    return {"status": "deleted", "document_id": document_id}


@router.get("/documents/jobs")
def ingest_jobs(limit: int = Query(default=20, ge=1, le=200),
                ingest_svc: IngestService = Depends(get_ingest)) -> list[dict]:
    return ingest_svc.jobs(limit)


@router.post("/documents/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: int, background: BackgroundTasks,
              ingest_svc: IngestService = Depends(get_ingest),
              ctx: AppContext = Depends(get_ctx)) -> dict:
    from ...db.models import IngestJob
    with ctx.db.session() as session:
        job = session.get(IngestJob, job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        if job.status != "failed":
            raise HTTPException(409, "仅失败任务可以重新提交")
        path = Path(job.path).resolve()
    if not path.is_relative_to(Path(ctx.settings.data_dir).resolve()) or not path.is_file():
        raise InvalidInputError("原文件已不存在，请重新上传")
    return _submit(path, background, ingest_svc, ctx)
