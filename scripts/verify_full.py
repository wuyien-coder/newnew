"""Accept a running full deployment, including real model streaming and upload lifecycle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]


def verify(client: httpx.Client) -> dict:
    manifest = json.loads((ROOT / "data/full/manifest.json").read_text(encoding="utf-8"))

    def get(path):
        response = client.get(path)
        response.raise_for_status()
        return response.json()

    stats = get("/api/v1/stats")
    assert stats["dataset"] == "full"
    assert stats["index"]["chunks"] >= manifest["chunks"]
    assert stats["index"]["persons"] >= manifest["graph_persons"]
    assert stats["index"]["embedder"] == "BgeEmbedding"
    assert stats["llm_provider"] == "openai"
    assert stats["config"]["database"] == "postgresql"
    assert stats["config"]["vector_store"] == "pgvector"
    assert stats["config"]["queue"] == "celery"
    assert stats["config"]["cache_enabled"]
    report = {"full_stack": "passed", "initial_chunks": stats["index"]["chunks"]}
    sessions = []
    document_id = None
    marker = "DELIVERY-" + uuid.uuid4().hex
    try:
        # 丁伟岳 is outside the 14 seed biographies.
        response = client.post("/api/v1/search", json={"query": "丁伟岳的研究贡献", "person": "丁伟岳"})
        response.raise_for_status()
        assert any(h["person"] == "丁伟岳" for h in response.json()["hits"])
        report["non_seed_retrieval"] = "passed"
        done = None
        tokens = 0
        event = ""
        with client.stream("POST", "/api/v1/chat/stream", json={
            "question": "丁伟岳有哪些研究贡献？请根据原文简要回答。", "mode": "rag"}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    assert event != "error", "stream error"
                    tokens += event == "token"
                    if event == "done":
                        done = data
        assert done and done["citations"] and tokens > 1
        assert done["usage"]["completion_tokens"] > 0
        sessions.append(done["session_id"])
        assert len(get("/api/v1/sessions/" + sessions[0])["messages"]) == 2
        report["real_stream_and_history"] = "passed"
        response = client.post("/api/v1/chat", json={
            "question": "苏步青师从谁？请使用关系查询工具并根据资料简要回答。", "mode": "agent"})
        response.raise_for_status()
        result = response.json()
        sessions.append(result["session_id"])
        assert result["engine"] == "react" and result["citations"] and result["agent_trace"]
        report["real_react"] = "passed"
        text = f"人物：交付验收探针\n来源：临时验收资料 {marker}\n====\n交付验收编号为 {marker}，约定标记为紫杉。"
        response = client.post("/api/v1/documents/upload", files={
            "file": (marker + ".txt", text.encode("utf-8"), "text/plain")})
        response.raise_for_status()
        job_id = response.json()["job_id"]
        for _ in range(120):
            job = next(j for j in get("/api/v1/documents/jobs") if j["id"] == job_id)
            if job["status"] in {"done", "completed", "success"}:
                break
            assert job["status"] != "failed", "upload ingestion failed"
            time.sleep(1)
        else:
            raise TimeoutError("upload ingestion did not finish")
        document_id = next(d["id"] for d in get("/api/v1/documents/list")
                           if d["person_name"] == "交付验收探针" and marker in d.get("source", ""))
        response = client.post("/api/v1/search", json={"query": marker})
        response.raise_for_status()
        assert any(marker in h["text"] for h in response.json()["hits"])
        report["async_upload_and_search"] = "passed"
    finally:
        if document_id is not None:
            client.delete(f"/api/v1/documents/{document_id}").raise_for_status()
        for session_id in sessions:
            client.delete("/api/v1/sessions/" + session_id).raise_for_status()
    assert get("/api/v1/stats")["index"]["chunks"] == report["initial_chunks"]
    report["cleanup"] = "passed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/full_acceptance.json")
    args = parser.parse_args()
    token = os.environ.get("MATHKB_SERVER__API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(base_url=args.base_url, headers=headers, timeout=240, trust_env=False) as client:
            report = {"status": "passed", **verify(client)}
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
