"""Durable ingestion jobs shared by local tasks and a Celery worker."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from sqlalchemy import select
from ..db.models import Chunk, Document, IngestJob
from ..rag.chunking import chunk_document
from ..rag.document_loader import load_document
from ..core.logging import get_logger
log = get_logger(__name__)

class IngestService:
    def __init__(self, ctx):
        self.ctx = ctx

    def create_job(self, path: Path) -> int:
        with self.ctx.db.session() as session:
            job = IngestJob(path=str(path.resolve()), status="pending")
            session.add(job)
            session.flush()
            return job.id

    def fail_job(self, job_id, error="摄取失败，请检查工作进程日志"):
        with self.ctx.db.session() as session:
            job = session.get(IngestJob, job_id)
            job.status, job.error, job.finished_at = "failed", error, datetime.now()

    def execute(self, job_id, rebuild=True):
        try:
            with self.ctx.db.session() as session:
                job = session.scalar(select(IngestJob).where(IngestJob.id == job_id).with_for_update())
                if job is None:
                    raise ValueError("摄取任务不存在")
                if job.status == "done":
                    return {"job_id": job.id, "chunks": job.chunks, "status": "done"}
                job.status, job.error = "running", ""
                path = Path(job.path).resolve()
                if not path.is_relative_to(Path(self.ctx.settings.data_dir).resolve()):
                    raise ValueError("摄取路径超出知识库目录")
                doc = load_document(path)
                cfg = self.ctx.settings.retrieval
                chunks = chunk_document(doc, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
                if not chunks:
                    raise ValueError("文档没有可索引正文")
                record = session.scalar(select(Document).where(Document.file_path == str(path)))
                if record is None:
                    record = Document(file_path=str(path), person_name=doc.person_name)
                    session.add(record)
                    session.flush()
                else:
                    session.query(Chunk).filter(Chunk.document_id == record.id).delete(synchronize_session=False)
                    session.flush()
                record.person_name = doc.person_name
                record.source = str(doc.meta.get("source", ""))
                record.page = str(doc.meta.get("page", ""))
                record.char_count = sum(len(c.text) for c in chunks)
                for c in chunks:
                    if doc.meta.get("format") == "pdf":
                        c.page = c.section_title.removeprefix("PDF第").removesuffix("页")
                    session.add(Chunk(id=c.chunk_id, document_id=record.id, person_name=c.person_name,
                                      section_title=c.section_title, chunk_index=c.chunk_index,
                                      text=c.text, token_count=len(c.text)//2, metadata_=c.metadata()))
                job.chunks = len(chunks)
            # The revision makes new data visible to other API/worker processes.
            self.ctx.bump_revision()
            if rebuild:
                self.ctx.build_index()
            with self.ctx.db.session() as session:
                job = session.get(IngestJob, job_id)
                job.status, job.finished_at = "done", datetime.now()
            return {"job_id": job_id, "chunks": len(chunks), "person": doc.person_name, "status": "done"}
        except Exception as exc:
            log.exception("摄取任务失败 job_id=%s", job_id)
            self.fail_job(job_id, str(exc) if isinstance(exc, ValueError) else "摄取失败，请检查工作进程日志")
            raise

    def run_background(self, job_id, rebuild=True):
        try:
            return self.execute(job_id, rebuild)
        except Exception:
            # The task status is persisted; HTTP 202 has already been returned.
            return None

    def ingest_file(self, path, rebuild=True):
        return self.execute(self.create_job(Path(path)), rebuild)

    def ingest_directory(self, directory):
        results = [self.ingest_file(p, rebuild=False) for p in sorted(Path(directory).rglob("*.txt"))]
        self.ctx.build_index()
        return results

    def jobs(self, limit=20):
        with self.ctx.db.session() as session:
            rows = session.scalars(select(IngestJob).order_by(IngestJob.id.desc()).limit(limit)).all()
        root = Path(self.ctx.settings.data_dir).resolve()
        return [{"id": j.id, "path": str(Path(j.path).relative_to(root)) if Path(j.path).is_relative_to(root) else Path(j.path).name,
                 "status": j.status, "chunks": j.chunks, "error": j.error,
                 "started_at": j.started_at.isoformat()} for j in rows]
