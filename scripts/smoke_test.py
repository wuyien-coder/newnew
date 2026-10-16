"""端到端自检：不起服务器，直接用 TestClient 走一遍检索→生成→落库→引用。

退出码非 0 表示链路有问题；打印每项耗时，便于快速看瓶颈在哪一段。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from mathkb.core.config import get_settings  # noqa: E402
from mathkb.main import create_app  # noqa: E402

CASES = [
    ("华罗庚在数论方面有哪些贡献？", "rag"),
    ("陈省身的籍贯在哪里？", "rag"),
    ("苏步青师从谁？", "agent"),
    ("研究微分几何的有哪些人？", "agent"),
]


def main() -> int:
    settings = get_settings()
    failures = 0
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        print(f"[health] status={health['status']} chunks={health['chunks']} "
              f"embedder={health['embedder']} llm={health['llm_provider']}")

        for question, mode in CASES:
            t0 = time.perf_counter()
            resp = client.post("/api/v1/chat", json={"question": question, "mode": mode})
            ms = round((time.perf_counter() - t0) * 1000, 1)
            if resp.status_code != 200:
                print(f"[FAIL] {question} → {resp.status_code} {resp.text[:120]}")
                failures += 1
                continue
            body = resp.json()
            ok = bool(body["answer"]) and bool(body["citations"])
            failures += 0 if ok else 1
            print(f"[{'OK ' if ok else 'FAIL'}] {mode:<7} {ms:>8}ms "
                  f"conf={body['confidence']:<6} cites={len(body['citations']):>2} "
                  f"ents={body['entities']} :: {question}")
            print(f"       引用样例: {body['citations'][0]['person']} "
                  f"{body['citations'][0]['section']} score={body['citations'][0]['score']}")

        search = client.post("/api/v1/search", json={"query": "堆垒素数论", "top_k": 3}).json()
        print(f"[search] hits={len(search['hits'])} pool={search['pool_size']} "
              f"lat={search['timings_ms']}")
        metrics_text = client.get("/metrics").text
        print(f"[metrics] 指标行数={len(metrics_text.strip().splitlines())}")

    print(f"[smoke] 失败项={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
