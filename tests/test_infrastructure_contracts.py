from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pytest


def test_redis_cache_roundtrip_and_version_isolation(ctx):
    from mathkb.core.cache import RetrievalCache
    class MemoryRedis:
        def __init__(self):
            self.values = {}
        def setex(self, key, ttl, value):
            self.values[key] = value
            assert ttl > 0
        def get(self, key):
            return self.values.get(key)
    cache = RetrievalCache(ctx.settings.cache)
    cache.client = MemoryRedis()
    key = cache.key(ctx, "华罗庚的贡献")
    result = ctx.retriever.retrieve("华罗庚的贡献")
    cache.put(key, result)
    restored = cache.get(key)
    assert [h.doc_id for h in restored.hits] == [h.doc_id for h in result.hits]
    assert restored.config == result.config
    changed = replace(ctx, index_stats={**ctx.index_stats, "revision": 999})
    assert cache.key(changed, "华罗庚的贡献") != key


def test_ingestion_failure_is_persisted(tmp_path):
    from tests.conftest import make_settings
    from mathkb.services.container import AppContext
    from mathkb.services.ingest_service import IngestService
    context = AppContext.create(make_settings(tmp_path), seed=False)
    svc = IngestService(context)
    job_id = svc.create_job(Path(context.settings.data_dir) / "missing.txt")
    with pytest.raises(FileNotFoundError):
        svc.execute(job_id)
    assert svc.jobs()[0]["status"] == "failed"
    assert "Traceback" not in svc.jobs()[0]["error"]


def test_celery_registered_task_executes_ingestion_and_is_idempotent(tmp_path, monkeypatch):
    pytest.importorskip("celery")
    from tests.conftest import make_settings
    from mathkb.services.container import AppContext
    from mathkb.services.ingest_service import IngestService
    import mathkb.worker as worker
    settings = make_settings(tmp_path)
    monkeypatch.setattr(worker, "settings", settings)
    ctx = AppContext.create(settings)
    path = Path(settings.data_dir) / "queued.txt"
    path.write_text("人物：队列测试\n====\n通过Celery任务写入知识库。", encoding="utf-8")
    svc = IngestService(ctx)
    job_id = svc.create_job(path)
    task = worker.celery_app.tasks["mathkb.ingest"]
    # Eager execution validates task registration/body; it does not claim a Redis broker test.
    result = task.apply(args=[job_id], throw=True).get()
    assert result["status"] == "done"
    revision = ctx.db.index_revision()
    assert task.apply(args=[job_id], throw=True).get()["status"] == "done"
    assert ctx.db.index_revision() == revision
    assert svc.jobs()[0]["status"] == "done"
