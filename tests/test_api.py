from __future__ import annotations


def test_health_and_stats(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["chunks"] > 0
    stats = client.get("/api/v1/stats").json()
    assert stats["llm_provider"] == "stub"
    assert stats["config"]["embedding"] == "hash"


def test_chat_returns_citations_and_persists_session(client):
    resp = client.post("/api/v1/chat", json={"question": "华罗庚在数论方面有哪些贡献？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["citations"], "回答必须带可溯源引用"
    assert body["session_id"]
    assert body["timings_ms"]["total"] > 0
    assert resp.headers["X-Trace-Id"]

    detail = client.get(f"/api/v1/sessions/{body['session_id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["citations"]


def test_agent_mode_routes_graph_questions(client):
    body = client.post("/api/v1/chat",
                       json={"question": "苏步青师从谁？", "mode": "agent"}).json()
    assert any(note.startswith("route(") for note in body["notes"])
    assert any("graph:" in note for note in body["notes"])
    assert body["entities"] == ["苏步青"]


def test_ablation_overrides_are_honored(client):
    base = client.post("/api/v1/search",
                       json={"query": "微分几何 曲面 整体性质", "top_k": 5}).json()
    variant = client.post(
        "/api/v1/search",
        json={"query": "微分几何 曲面 整体性质", "top_k": 5,
              "overrides": {"use_dense": False}}).json()
    assert base["timings_ms"].get("dense")
    assert "dense" not in variant["timings_ms"]


def test_invalid_input_is_rejected(client):
    assert client.post("/api/v1/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/v1/chat",
                       json={"question": "x", "mode": "bogus"}).status_code == 422


def test_ingest_endpoint_rejects_path_traversal(client):
    resp = client.post("/api/v1/documents", json={"path": "../../../../etc/passwd.txt"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_feedback_and_metrics(client):
    chat = client.post("/api/v1/chat", json={"question": "陈省身的籍贯"}).json()
    ok = client.post(f"/api/v1/sessions/{chat['session_id']}/feedback",
                     json={"session_id": chat["session_id"], "rating": 1})
    assert ok.status_code == 201
    body = client.get("/metrics").text
    assert "chat_requests_count" in body or "chat.requests" in body.replace("_", ".")
