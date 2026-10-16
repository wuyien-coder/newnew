"""Celery ingestion worker; start with --concurrency=1 for serialized document ingestion."""
import threading
from celery import Celery
from celery.signals import worker_shutdown
from sqlalchemy.exc import OperationalError
from .core.config import get_settings
settings = get_settings()
celery_app = Celery("mathkb", broker=settings.queue.redis_url, backend=settings.queue.redis_url)
celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"],
                       task_acks_late=True, task_reject_on_worker_lost=True,
                       worker_prefetch_multiplier=1, broker_connection_retry_on_startup=True,
                       task_track_started=True, result_expires=3600,
                       broker_transport_options={"visibility_timeout": 3600})

_context = None
_context_lock = threading.RLock()


def get_worker_context():
    global _context
    from .services.container import AppContext
    with _context_lock:
        if _context is None or _context.settings.database_url != settings.database_url:
            if _context is not None:
                _context.close()
            _context = AppContext.create(settings, seed=False)
        return _context


@worker_shutdown.connect
def close_worker_context(**kwargs):
    global _context
    if _context is not None:
        _context.close()
        _context = None


@celery_app.task(name="mathkb.ingest", bind=True,
                 autoretry_for=(OperationalError, ConnectionError, TimeoutError, OSError),
                 retry_kwargs={"max_retries": 3}, retry_backoff=True, retry_jitter=True)
def ingest(self, job_id):
    from .services.ingest_service import IngestService
    return IngestService(get_worker_context()).execute(job_id)
