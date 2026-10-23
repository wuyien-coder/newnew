"""Persistent native app behavior needed by a hosted container."""
import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

from mathkb.main import create_app
from tests.conftest import make_settings


def test_session_uploaded_document_and_interrupted_job_survive_restart(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        answer = client.post("/api/v1/chat", json={"question": "苏步青的贡献"}).json()
        sid = answer["session_id"]
        upload = client.post("/api/v1/documents/upload", files={
            "file": ("sample.txt", "人物：在线验证人物\n====\n云杉蓝桥标记测试资料用于验证持久化。".encode(), "text/plain"),
        })
        assert upload.status_code == 202
        jobs = client.get("/api/v1/documents/jobs").json()
        assert next(j for j in jobs if j["id"] == upload.json()["job_id"])["status"] == "done"
        unfinished = client.app.state.ingest.create_job(Path(settings.data_dir) / "uploads" / "interrupted.txt")

    with TestClient(create_app(settings)) as client:
        history = client.get(f"/api/v1/sessions/{sid}").json()["messages"]
        assert len(history) == 2 and history[-1]["citations"]
        results = client.post("/api/v1/search", json={"query": "云杉蓝桥标记"}).json()["hits"]
        assert any("云杉蓝桥标记" in hit["text"] for hit in results)
        job = next(j for j in client.get("/api/v1/documents/jobs").json() if j["id"] == unfinished)
        assert job["status"] == "failed" and "重试" in job["error"]


def test_online_boot_requires_access_token_and_keeps_existing_uploads(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("online_boot", root / "deploy" / "online.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("MATHKB_PERSIST_DIR", str(tmp_path / "persistent"))
    monkeypatch.delenv("MATHKB_SERVER__API_TOKEN", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="API_TOKEN"):
        module.configure_environment()
    monkeypatch.setenv("MATHKB_SERVER__API_TOKEN", "test-site-token-with-32-characters")
    # All values are sandboxed to this test's environment, not shared settings.
    for key in ["MATHKB_DATA_DIR", "MATHKB_DATABASE_URL", "MATHKB_EMBEDDING__PROVIDER"]:
        monkeypatch.delenv(key, raising=False)
    with monkeypatch.context() as isolated:
        for key in list(module.os.environ):
            if key.startswith("MATHKB_") and key not in {"MATHKB_PERSIST_DIR", "MATHKB_SERVER__API_TOKEN"}:
                isolated.delenv(key, raising=False)
        # Restore defaults written with setdefault when the context exits.
        before = dict(module.os.environ)
        try:
            module.configure_environment()
            uploaded = tmp_path / "persistent" / "corpus" / "uploads" / "keep.txt"
            uploaded.parent.mkdir(exist_ok=True)
            uploaded.write_text("keep", encoding="utf-8")
            module.configure_environment()
            assert uploaded.read_text() == "keep"
            assert module.os.environ["MATHKB_EMBEDDING__PROVIDER"] == "bge"
            assert len(list((uploaded.parents[1] / "biographies").glob("*.txt"))) == 326
        finally:
            for key in set(module.os.environ) - set(before):
                module.os.environ.pop(key, None)
