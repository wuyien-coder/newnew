"""pytest 共享夹具：临时库 + 种子语料 + 离线索引（不联网、不花钱）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathkb.core.config import Settings  # noqa: E402
from mathkb.services.chat_service import ChatService  # noqa: E402
from mathkb.services.container import AppContext  # noqa: E402

SEED_DIR = ROOT / "data" / "seed"


def make_settings(tmp_path: Path) -> Settings:
    import shutil
    corpus = tmp_path / "corpus"
    shutil.copytree(SEED_DIR, corpus, dirs_exist_ok=True)
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        data_dir=str(corpus),
        index_dir=str(tmp_path / "index"),
    )
    settings.vector.persist_dir = str(tmp_path / "index" / "chroma")
    settings.keyword.persist_dir = str(tmp_path / "index" / "bm25")
    settings.embedding.provider = "hash"
    settings.rerank.enabled = False
    settings.llm.provider = "stub"
    settings.ensure_dirs()
    return settings


@pytest.fixture(scope="session")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("mathkb")


@pytest.fixture(scope="session")
def settings(tmp_root):
    return make_settings(tmp_root)


@pytest.fixture(scope="session")
def ctx(settings):
    app_ctx = AppContext.create(settings)
    app_ctx.build_index()
    return app_ctx


@pytest.fixture(scope="session")
def chat(ctx):
    return ChatService(ctx)


@pytest.fixture(scope="session")
def client(settings):
    from fastapi.testclient import TestClient

    from mathkb.main import create_app

    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_bio() -> Path:
    return SEED_DIR / "biographies" / "华罗庚.txt"


@pytest.fixture(scope="session")
def bios() -> list[Path]:
    return sorted((SEED_DIR / "biographies").glob("*.txt"))
