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
    args = parser.parse_args()

    settings = get_settings()
    ctx = AppContext.create(settings)
    stats = ctx.build_index(force=args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"[index] provider={settings.embedding.provider} "
          f"vector={settings.vector.provider} rerank={settings.rerank.enabled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
