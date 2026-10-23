"""OpenAI 兼容 Provider（DeepSeek / Qwen / vLLM / OneAPI 同源）。

论文工程踩过的两个坑在这里固化成代码：
- Qwen 系默认开启思考模式，评测时响应极慢 → 显式 `enable_thinking=False`；
- DeepSeek 思考系模型流式会先吐 reasoning_content 耗尽 max_tokens 导致正文为空
  → 显式 `thinking={"type":"disabled"}`。
重试只做一次（并发评测下过度重试会加剧拥塞），失败统一抛 ProviderError。
"""
from __future__ import annotations

import time

from ..core.config import LLMSettings
from ..core.exceptions import ProviderError
from ..core.logging import get_logger
from .base import LLMClient, LLMResult, Message, Usage

log = get_logger(__name__)


class OpenAICompatClient(LLMClient):
    name = "openai"

    def __init__(self, cfg: LLMSettings):
        if not cfg.api_key:
            raise ProviderError("LLM api_key 为空，请在 .env 配置 MATHKB_LLM__API_KEY 或改用 stub provider")
        import httpx
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                             timeout=cfg.timeout, max_retries=cfg.max_retries,
                             http_client=httpx.Client(timeout=cfg.timeout, follow_redirects=False))
        self.model = cfg.model
        log.info("LLM provider=openai base=%s model=%s", cfg.base_url, cfg.model)

    def _extra_body(self) -> dict:
        model = (self.model or "").lower()
        if "qwen" in model:
            return {"enable_thinking": False}
        if "deepseek" in model and any(t in model for t in ("v4", "r1", "reasoner")):
            return {"thinking": {"type": "disabled"}}
        return {}

    def chat(self, messages: list[Message], *, temperature=None, max_tokens=None) -> LLMResult:
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.cfg.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.cfg.max_tokens,
                extra_body=self._extra_body() or None,
            )
        except Exception as exc:
            raise _provider_error(exc) from None
        usage = _parse_usage(resp)
        return LLMResult(text=resp.choices[0].message.content or "", model=self.model,
                         provider="openai", usage=usage,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    def stream_events(self, messages):
        t0 = time.perf_counter()
        pieces = []
        usage = Usage()
        response = None
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens, stream=True,
                stream_options={"include_usage": True},
                extra_body=self._extra_body() or None,
            )
            for chunk in response:
                if getattr(chunk, "usage", None):
                    usage = _parse_usage(chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    pieces.append(delta)
                    yield delta
            yield LLMResult(text="".join(pieces), model=self.model, provider=self.name,
                            usage=usage, latency_ms=round((time.perf_counter()-t0)*1000, 2))
        except Exception as exc:
            raise _provider_error(exc) from None
        finally:
            if response is not None:
                response.close()

    def stream(self, messages, *, temperature=None, max_tokens=None):
        for event in self.stream_events(messages):
            if isinstance(event, str):
                yield event


def _parse_usage(resp) -> Usage:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return Usage()
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
    )


def _provider_error(exc: Exception) -> ProviderError:
    """Use only status/type; upstream bodies may echo the submitted credential."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        message = "API Key 无效或已过期，请在连接设置中更换。"
    elif status == 403:
        message = "模型访问被拒绝，请检查账号权限、模型授权和千问地域。"
    elif status == 429:
        message = "额度不足或请求过于频繁，请检查余额和调用限额。"
    elif status in (400, 404, 422):
        message = "模型名称或请求配置不匹配，请检查模型名称及千问地域。"
    elif isinstance(status, int) and 300 <= status < 400:
        message = "模型接口返回了重定向，请使用官方 Base URL。"
    elif isinstance(status, int) and status >= 500:
        message = "模型服务暂不可用，请稍后重试。"
    elif "Timeout" in type(exc).__name__:
        message = "模型响应超时，请稍后重试或更换模型。"
    else:
        message = "无法连接模型服务，请检查网络及官方 Base URL。"
    return ProviderError(message, detail={"upstream_status": status})
