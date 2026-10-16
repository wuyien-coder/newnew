"""离线评测入口：消融矩阵 + 去名化难题集 + 选择题基线。

用法：
    PYTHONPATH=src python scripts/run_eval.py                     # 默认 hash embedding，秒级
    MATHKB_EMBEDDING__PROVIDER=bge MATHKB_EMBEDDING__CACHE_DIR=<模型缓存> \
        PYTHONPATH=src python scripts/run_eval.py --rerank         # 论文同配置，含精排对照

说明：
- named 集（查询含人名）用于验证产品行为；hard 集（去名化原句）才反映排序质量；
- 精排是 CPU 上的延迟瓶颈，默认只在前 N 条查询上做对照，避免整轮评测被拖成十几分钟。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathkb.core.config import get_settings  # noqa: E402
from mathkb.eval.dataset import load_mcq, load_retrieval  # noqa: E402
from mathkb.eval.runner import (  # noqa: E402
    DEFAULT_ABLATIONS, QUALITY_ABLATIONS, evaluate_ablations, evaluate_mcq, save_reports,
    write_json,
)
from mathkb.services.container import AppContext  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true", help="加载 Cross-Encoder 并做精排对照")
    parser.add_argument("--rerank-subset", type=int, default=12, help="精排对照的查询条数上限")
    parser.add_argument("--skip-mcq", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "reports" / "eval_latest.json"))
    args = parser.parse_args()

    settings = get_settings()
    if args.rerank:
        settings.rerank.enabled = True
    ctx = AppContext.create(settings)
    ctx.build_index()

    qa_dir = Path(settings.data_dir) / "qa"
    named = load_retrieval(qa_dir / "retrieval.jsonl")
    hard = load_retrieval(qa_dir / "retrieval_hard.jsonl")

    reports = []
    reports += _tag("named", evaluate_ablations(ctx, named, DEFAULT_ABLATIONS, top_k=args.top_k))
    reports += _tag("hard", evaluate_ablations(
        ctx, hard, {f"q/{k}": v for k, v in QUALITY_ABLATIONS.items()}, top_k=args.top_k))

    if args.rerank:
        subset = hard[: args.rerank_subset]
        reports += _tag("hard", evaluate_ablations(
            ctx, subset, {"rerank_off": {"use_entity_filter": False, "use_rerank": False},
                          "rerank_on": {"use_entity_filter": False, "use_rerank": True}},
            top_k=args.top_k))

    if not args.skip_mcq:
        mcq = evaluate_mcq(ctx, load_mcq(qa_dir / "mcq.jsonl"))
        mcq.name = f"mcq/{mcq.name}"
        reports.append(mcq)

    print(_table(reports))
    save_reports(ctx.db, settings, reports)
    out = write_json(reports, Path(args.out))
    print(f"[eval] 报告已写入 {out}")
    return 0


def _tag(group: str, reports) -> list:
    for report in reports:
        report.name = f"{group}/{report.name}"
        report.config["dataset"] = group
    return reports


def _table(reports) -> str:
    header = f"{'config':<40}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>8}{'nD@5':>8}{'p50':>8}{'p95':>8}"
    lines = [header, "-" * len(header)]
    for r in reports:
        m, l = r.metrics, r.latency_ms
        if "accuracy" in m:
            lines.append(f"{r.name:<40} ACC={m['accuracy']:.4f} ({m['correct']}/{r.n})  "
                         f"p50={l['p50']}ms")
            continue
        lines.append(
            f"{r.name:<40}{m['recall@1']:>7}{m['recall@3']:>7}{m['recall@5']:>7}"
            f"{m['mrr']:>8.3f}{m['ndcg@5']:>8.3f}{l['p50']:>8}{l['p95']:>8}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
