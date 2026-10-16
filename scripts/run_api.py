"""本地启动 API：PYTHONPATH=src python scripts/run_api.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from mathkb.core.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "mathkb.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=os.environ.get("MATHKB_RELOAD") == "1",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
