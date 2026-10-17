"""建库 + 种子 + 构建索引（服务启动前的离线步骤，也可单独重跑）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathkb.core.config import get_settings  # noqa: E402
from mathkb.services.container import AppContext  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重算全部向量")
    parser.add_argument("--sync-corpus", action="store_true", help="补齐当前配置目录的语料与图谱，保留已有会话和上传资料")
    args = parser.parse_args()

    settings = get_settings()
    ctx = AppContext.create(settings, seed=not args.sync_corpus)
    try:
        if args.sync_corpus:
            from mathkb.db.seed import seed_knowledgebase
            with ctx.db.session() as session:
                changes = seed_knowledgebase(session, settings, force=True)
            if any(changes.get(key, 0) for key in ("documents", "chunks", "persons", "relations")):
                ctx.bump_revision()
        stats = ctx.build_index(force=args.force)
        print(json.dumps(stats, ensure_ascii=False, indent=1))
        print(f"[index] provider={settings.embedding.provider} "
              f"vector={settings.vector.provider} rerank={settings.rerank.enabled}")
    finally:
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
