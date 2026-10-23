"""The native API keeps BYOK credentials request-scoped, including SSE failures."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mathkb.core.exceptions import InvalidInputError, ProviderError
from mathkb.llm.connection import ModelConnection, request_model
from mathkb.llm.openai_compat import _provider_error
from mathkb.llm.stub import StubLLMClient


def connection(key="sk-test-credential-one", **updates):
    return {"provider": "deepseek", "api_key": key, "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat", **updates}


@pytest.mark.parametrize("key", ["问题写在这里，不是密钥", "Bearer sk-example-key", "sk-key with spaces", "short"])
def test_key_validation_before_transport(key):
    with pytest.raises(InvalidInputError):
        ModelConnection(**connection(key)).settings()


@pytest.mark.parametrize("url", ["https://api.deepseek.com.attacker.example/v1",
                                "http://127.0.0.1/v1", "https://user:pass@api.deepseek.com/v1",
                                "https://api.deepseek.com:443/v1", "https://api.deepseek.com/v1?key=bad"])
def test_visitor_endpoint_is_official_only(url):
    with pytest.raises(InvalidInputError):
        ModelConnection(**connection(base_url=url)).settings()


def test_endpoint_normalization_and_secret_repr():
    deepseek = ModelConnection(**connection(base_url="https://api.deepseek.com/v1/chat/completions"))
    assert deepseek.settings().base_url == "https://api.deepseek.com/v1"
    assert "sk-test-credential-one" not in repr(deepseek)
    qwen = ModelConnection(**connection(provider="qwen", base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/"))
    assert qwen.settings().base_url.endswith("/compatible-mode/v1")


def test_parallel_request_clients_do_not_change_shared_configuration(client, monkeypatch):
    closed = []

    class FakeClient(StubLLMClient):
        def __init__(self, settings):
            super().__init__(settings.model)
            self.client = SimpleNamespace(close=lambda: closed.append(settings.api_key))

    monkeypatch.setattr("mathkb.llm.openai_compat.OpenAICompatClient", FakeClient)
    global_llm = client.app.state.ctx.llm

    def run(key):
        response = client.post("/api/v1/chat", json={"question": "华罗庚的学术贡献", "model_connection": connection(key)})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] and body["citations"]
        detail = client.get(f"/api/v1/sessions/{body['session_id']}")
        assert key not in detail.text and key not in response.text
        return key

    keys = ["sk-test-visitor-one", "sk-test-visitor-two"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, keys)) == keys
    assert sorted(closed) == sorted(keys)
    assert client.app.state.ctx.llm is global_llm
    assert client.app.state.ctx.settings.llm.api_key == ""


def test_sse_closes_model_client_and_exposes_safe_error(client, monkeypatch):
    closed = []

    class FailingClient(StubLLMClient):
        def __init__(self, settings):
            super().__init__()
            self.client = SimpleNamespace(close=lambda: closed.append(True))

        def stream_events(self, messages):
            raise ProviderError("API Key 无效，请更换。", detail={"upstream_status": 401})
            yield  # generator contract

    monkeypatch.setattr("mathkb.llm.openai_compat.OpenAICompatClient", FailingClient)
    result = client.post("/api/v1/chat/stream", json={"question": "华罗庚的贡献", "model_connection": connection()})
    assert "event: error" in result.text and '"upstream_status": 401' in result.text
    assert "sk-test-credential-one" not in result.text
    assert closed == [True]


def test_validation_errors_never_echo_credentials(client):
    result = client.post("/api/v1/chat", json={"question": "华罗庚", "model_connection": connection(model="")})
    assert result.status_code == 422
    assert "sk-test-credential-one" not in result.text
    assert "input" not in result.text


def test_upstream_errors_never_echo_response_body_or_key():
    class UpstreamError(Exception):
        status_code = 401
    error = _provider_error(UpstreamError("sk-do-not-echo-this-key"))
    assert "sk-do-not-echo-this-key" not in str(error)
    assert error.detail == {"upstream_status": 401}


def test_model_check_requires_nonempty_generation_and_closes(client, monkeypatch):
    closed = []

    class CheckClient:
        model = "deepseek-chat"

        def __init__(self, settings):
            self.client = SimpleNamespace(close=lambda: closed.append(True))

        def chat(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 32
            return SimpleNamespace(text="OK", latency_ms=12)

    monkeypatch.setattr("mathkb.llm.openai_compat.OpenAICompatClient", CheckClient)
    result = client.post("/api/v1/chat/model/check", json=connection())
    assert result.json() == {"ok": True, "model": "deepseek-chat", "latency_ms": 12}
    assert closed == [True]
    with request_model(None) as model:
        assert model is None
