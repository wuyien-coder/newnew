"""Visitor-supplied model credentials, used only for one request."""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr

from ..core.config import LLMSettings
from ..core.exceptions import InvalidInputError


class ModelConnection(BaseModel):
    provider: Literal["deepseek", "qwen"]
    api_key: SecretStr = Field(repr=False)
    model: str = Field(min_length=1, max_length=160)
    base_url: str = Field(min_length=1, max_length=300)

    def settings(self) -> LLMSettings:
        key = self.api_key.get_secret_value().strip()
        if not re.fullmatch(r"[!-~]{8,512}", key) or key.lower().startswith("bearer"):
            raise InvalidInputError("API Key 格式错误，请填写服务商提供的密钥；不要填写问题或 Bearer 前缀。")
        model = self.model.strip()
        if not re.fullmatch(r"[A-Za-z0-9._:/-]{1,160}", model):
            raise InvalidInputError("模型名称格式错误。")
        try:
            url = urlsplit(self.base_url.strip().rstrip("/"))
            if url.scheme != "https" or url.username or url.password or url.port or url.query or url.fragment:
                raise ValueError
            host = (url.hostname or "").lower()
            path = url.path.removesuffix("/chat/completions").rstrip("/")
            if self.provider == "deepseek":
                valid = host == "api.deepseek.com" and path in ("", "/v1")
                path = "/v1"
            else:
                official = host in {
                    "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com",
                } or bool(re.fullmatch(
                    r"[a-z0-9-]+\.(?:cn-beijing|cn-hongkong|ap-southeast-1|us-east-1|ap-northeast-1)\.maas\.aliyuncs\.com", host,
                ))
                valid = official and path == "/compatible-mode/v1"
            if not valid:
                raise ValueError
        except ValueError:
            raise InvalidInputError("Base URL 需为所选服务商的官方 HTTPS 接口，请检查千问地域和路径。") from None
        return LLMSettings(provider="openai", api_key=key, model=model,
                           base_url=f"https://{host}{path}", timeout=120, max_retries=0)


@contextmanager
def request_model(connection: ModelConnection | None, *, timeout: float | None = None):
    """Never update shared settings, persist the key, or reuse a visitor's client."""
    if connection is None:
        yield None
        return
    from .openai_compat import OpenAICompatClient

    settings = connection.settings()
    if timeout is not None:
        settings = settings.model_copy(update={"timeout": timeout})
    client = OpenAICompatClient(settings)
    try:
        yield client
    finally:
        client.client.close()
