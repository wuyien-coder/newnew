"""应用配置：.env / config 文件注入，禁止在代码中写密钥。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class LLMSettings(BaseModel):
    provider: str = "stub"  # stub | openai
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = Field(default="", repr=False)
    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 1200
    timeout: float = 60.0
    max_retries: int = 1
    # 单价（元/千 token），仅用于成本核算，未配置则不计成本
    price_input: float = 0.0
    price_output: float = 0.0


class EmbeddingSettings(BaseModel):
    provider: str = "hash"  # hash（离线、可复现） | bge（本地模型）
    model_name: str = "BAAI/bge-small-zh-v1.5"
    cache_dir: str = str(ROOT / "models")
    dim: int = 512  # hash 降级时的向量维度
    batch_size: int = 64
    vector_cache_dir: str = str(ROOT / ".index" / "embedding-cache")


class RerankSettings(BaseModel):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-base"
    cache_dir: str = str(ROOT / "models")
    doc_chars: int = 512
    batch_size: int = 32


class RetrievalSettings(BaseModel):
    top_k: int = 5
    candidate_multiplier: int = 4
    max_candidates: int = 60
    rrf_k: int = 60
    chunk_size: int = 500
    chunk_overlap: int = 100
    conf_high: float = 0.60
    conf_medium: float = 0.30


class VectorSettings(BaseModel):
    provider: Literal["memory", "pgvector"] = "memory"
    # 兼容旧配置；memory 不持久化向量，pgvector 使用 database_url。
    persist_dir: str = str(ROOT / ".index" / "vectors")
    collection: str = "mathkb"


class KeywordSettings(BaseModel):
    persist_dir: str = str(ROOT / ".index" / "bm25")


class KGSettings(BaseModel):
    provider: str = "relational"  # 实际来源是 SQL 关系表，进程内构建 networkx 快照
    path: str = str(ROOT / ".index" / "kg.sqlite3")
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""


class QueueSettings(BaseModel):
    provider: Literal["local", "celery"] = "local"
    redis_url: str = "redis://localhost:6379/0"


class CacheSettings(BaseModel):
    enabled: bool = False
    redis_url: str = "redis://localhost:6379/1"
    ttl_seconds: int = Field(default=300, ge=1)


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_token: str = ""  # 为空表示本地开发不鉴权


class AgentSettings(BaseModel):
    max_steps: int = Field(default=4, ge=1, le=8)
    max_evidence: int = Field(default=12, ge=1, le=20)
    max_observation_chars: int = Field(default=6000, ge=1000, le=12000)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="MATHKB_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "mathkb"
    env: str = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    database_url: str = f"sqlite:///{ROOT / '.index' / 'mathkb.db'}"
    # 本地/测试自动建表；生产设为 false，先执行 alembic upgrade head。
    create_schema: bool = True
    data_dir: str = str(ROOT / "data" / "seed")
    index_dir: str = str(ROOT / ".index")
    # 全量语料留在论文目录，不复制进本仓库（避免把受版权保护的整本书带进 git）
    # 为空表示用 data_dir/biographies；生产/全量场景设 MATHKB_BIOGRAPHIES_DIR 指向原文目录
    biographies_dir: str = ""
    llm_config_file: str = ""  # 可只读引用本机已有配置的密钥，不复制到新仓库

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    keyword: KeywordSettings = Field(default_factory=KeywordSettings)
    kg: KGSettings = Field(default_factory=KGSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    @model_validator(mode="after")
    def load_existing_llm_key(self):
        if self.llm.provider == "openai" and not self.llm.api_key and self.llm_config_file:
            from dotenv import dotenv_values
            source = Path(self.llm_config_file).expanduser().resolve()
            if not source.is_file():
                raise ValueError("配置的模型凭据文件不存在")
            self.llm.api_key = dotenv_values(source).get("MATHKB_LLM__API_KEY") or ""
        return self
    server: ServerSettings = Field(default_factory=ServerSettings)

    def ensure_dirs(self) -> None:
        for p in (self.index_dir, self.vector.persist_dir, self.keyword.persist_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
