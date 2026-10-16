from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from reportlab.pdfgen.canvas import Canvas

from mathkb.core.config import Settings
from mathkb.main import create_app
from tests.conftest import make_settings


def test_external_llm_credentials_are_loaded_only_for_real_model(tmp_path):
    config = tmp_path / "local.env"
    config.write_text("MATHKB_LLM__API_KEY=unit-test-credential\n", encoding="utf-8")
    settings = Settings(_env_file=None, llm={"provider": "openai"}, llm_config_file=str(config))
    assert settings.llm.api_key == "unit-test-credential"
    assert "unit-test-credential" not in repr(settings.llm)
    offline = Settings(_env_file=None, llm={"provider": "stub"}, llm_config_file=str(tmp_path / "missing.env"))
    assert offline.llm.api_key == ""


def test_session_list_switch_and_delete(client):
    answer = client.post("/api/v1/chat", json={"question": "陈省身的学术贡献"}).json()
    session_id = answer["session_id"]
    sessions = client.get("/api/v1/sessions?limit=100").json()["sessions"]
    saved = next(s for s in sessions if s["id"] == session_id)
    assert saved["title"] == "陈省身的学术贡献" and saved["turn_count"] == 1
    detail = client.get(f"/api/v1/sessions/{session_id}").json()
    assert detail["messages"][1]["citations"]
    assert client.delete(f"/api/v1/sessions/{session_id}").status_code == 200
    assert client.get(f"/api/v1/sessions/{session_id}").json()["messages"] == []
    assert client.delete(f"/api/v1/sessions/{session_id}").status_code == 404


def test_pdf_upload_page_metadata_and_retry(tmp_path):
    stream = BytesIO()
    canvas = Canvas(stream)
    canvas.drawString(60, 700, "BlueBridge evaluates retrieval using recall and MRR.")
    canvas.showPage()
    canvas.drawString(60, 700, "This is the second page of the research record.")
    canvas.save()
    with TestClient(create_app(make_settings(tmp_path))) as client:
        result = client.post("/api/v1/documents/upload", files={"file": ("research.pdf", stream.getvalue(), "application/pdf")})
        assert result.status_code == 202
        job_id = result.json()["job_id"]
        jobs = client.get("/api/v1/documents/jobs").json()
        assert next(j for j in jobs if j["id"] == job_id)["status"] == "done"
        result = client.post("/api/v1/search", json={"query": "BlueBridge recall MRR"}).json()
        assert any(h["page"] == "1" and "BlueBridge" in h["text"] for h in result["hits"])
        client.app.state.ingest.fail_job(job_id, "模拟索引失败")
        retry = client.post(f"/api/v1/documents/jobs/{job_id}/retry")
        assert retry.status_code == 202 and retry.json()["job_id"] != job_id
        retried = next(j for j in client.get("/api/v1/documents/jobs").json() if j["id"] == retry.json()["job_id"])
        assert retried["status"] == "done"
        assert client.post(f"/api/v1/documents/jobs/{retry.json()['job_id']}/retry").status_code == 409
        stats = client.get("/api/v1/stats").json()
        assert stats["index"]["documents"] == 15 and "llm_model" in stats


def test_scanned_pdf_rejection_explains_how_to_repair(tmp_path):
    from pypdf import PdfWriter
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=300)
    writer.write(output)
    with TestClient(create_app(make_settings(tmp_path))) as client:
        response = client.post("/api/v1/documents/upload", files={"file": ("scan.pdf", output.getvalue(), "application/pdf")})
        job_id = response.json()["job_id"]
        job = next(j for j in client.get("/api/v1/documents/jobs").json() if j["id"] == job_id)
        assert job["status"] == "failed" and "OCR" in job["error"]
