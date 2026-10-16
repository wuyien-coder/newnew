"""LLM Provider 工厂：按配置装配，密钥只从 .env / 配置注入。"""
from __future__ import annotations

from ..core.config import LLMSettings
from ..core.logging import get_logger
from .base import LLMClient
from .stub import StubLLMClient

log = get_logger(__name__)


def build_llm(cfg: LLMSettings) -> LLMClient:
    if cfg.provider == "stub":
        log.info("LLM provider=stub（离线抽取式，不产生任何 API 费用）")
        return StubLLMClient(model=cfg.model or "stub-extractive")
    if cfg.provider == "openai":
        from .openai_compat import OpenAICompatClient

        return OpenAICompatClient(cfg)
    raise ValueError(f"未知 LLM provider: {cfg.provider}")
