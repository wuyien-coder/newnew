"""Read-only checks for configured PostgreSQL/pgvector, Redis and Celery workers.

No LLM calls or migrations are made. Unconfigured services are reported as skipped.
Use --require-all in the Compose API container to require all three services.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_checks(require_all: bool = False) -> tuple[list[dict], int]:
    from mathkb.core.config import Settings
    from mathkb.db.base import normalize_database_url

    settings = Settings()
    database_url = normalize_database_url(settings.database_url)
    checks: list[dict] = []

    def record(component: str, configured: bool, check) -> None:
        if not configured:
            checks.append({"component": component, "status": "skipped",
                           "reason": "not configured"})
            return
        try:
            detail = check()
            checks.append({"component": component, "status": "passed", "detail": detail})
        except Exception as exc:
            # Do not echo connection strings, passwords, tokens or provider error bodies.
            checks.append({"component": component, "status": "failed",
                           "error_type": type(exc).__name__})

    def check_postgres() -> dict:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url, connect_args={"connect_timeout": 5})
        try:
            with engine.connect() as connection:
                assert connection.scalar(text("SELECT 1")) == 1
                version = connection.scalar(text(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
                if not version:
                    raise RuntimeError("pgvector extension is not enabled")
            return {"connection": "ready", "pgvector": version}
        finally:
            engine.dispose()

    def check_redis() -> dict:
        from redis import Redis

        urls = {}
        if settings.cache.enabled:
            urls["cache"] = settings.cache.redis_url
        if settings.queue.provider == "celery":
            urls["queue"] = settings.queue.redis_url
        verified = []
        for role, url in urls.items():
            client = Redis.from_url(url, socket_connect_timeout=5, socket_timeout=5)
            try:
                if not client.ping():
                    raise RuntimeError("Redis PING failed")
                verified.append(role)
            finally:
                client.close()
        return {"roles": verified}

    def check_celery() -> dict:
        from mathkb.worker import celery_app

        # Fail promptly when the broker is unavailable instead of retrying silently.
        with celery_app.connection_for_read(connect_timeout=5) as connection:
            connection.ensure_connection(max_retries=0)
        responses = celery_app.control.inspect(timeout=5).ping()
        if not responses:
            raise RuntimeError("no live Celery worker responded")
        return {"workers": sorted(responses)}

    postgres_configured = database_url.startswith(("postgresql:", "postgresql+"))
    cache_enabled = bool(getattr(getattr(settings, "cache", None), "enabled", False))
    celery_enabled = getattr(getattr(settings, "queue", None), "provider", "local") == "celery"
    record("postgresql_pgvector", postgres_configured, check_postgres)
    record("redis", cache_enabled or celery_enabled, check_redis)
    record("celery_worker", celery_enabled, check_celery)
    failed = any(item["status"] == "failed" for item in checks)
    incomplete = require_all and any(item["status"] != "passed" for item in checks)
    return checks, 1 if failed or incomplete else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true",
                        help="exit nonzero unless every external service passes")
    args = parser.parse_args()
    try:
        checks, exit_code = run_checks(args.require_all)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"checks": checks, "all_required": args.require_all},
                     ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
