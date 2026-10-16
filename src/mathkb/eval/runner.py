"""离线评测：检索指标 + 消融矩阵 + 选择题准确率。

设计约束（与论文实验一致）：
1. 一次只改一个变量，其余配置从同一份 RetrievalConfig 复制；
2. 指标由 metrics.py 纯函数复算，不依赖被测代码内部状态；
3. 所有数字都能用 `python scripts/run_eval.py` 在本机重跑复现，
   provider（embedding/rerank/llm）会写进报告，避免不同硬件/模型间横向比较。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

from ..core.config import Settings
from ..core.logging import get_logger
from ..services.container import AppContext
from . import metrics
from .dataset import MCQCase, RetrievalCase

log = get_logger(__name__)


@dataclass
class EvalReport:
    name: str
    config: dict
    metrics: dict
    latency_ms: dict
    n: int
    notes: list[str] = field(default_factory=list)


DEFAULT_ABLATIONS: dict[str, dict] = {
    "dense_only": {"use_bm25": False, "use_graph": False},
    "bm25_only": {"use_dense": False, "use_graph": False},
    "hybrid_rrf": {"use_graph": False},
    "hybrid_weighted": {"use_graph": False, "fusion": "weighted"},
    "hybrid_rrf_graph": {},
}

# 排序质量组：关闭「实体命中即限定人物」的产品行为，否则带人名的查询会被过滤到饱和
QUALITY_ABLATIONS: dict[str, dict] = {
    "dense_only_nofilter": {"use_bm25": False, "use_graph": False, "use_entity_filter": False},
    "bm25_only_nofilter": {"use_dense": False, "use_graph": False, "use_entity_filter": False},
    "hybrid_rrf_nofilter": {"use_graph": False, "use_entity_filter": False},
    "hybrid_weighted_nofilter": {"use_graph": False, "fusion": "weighted",
                                 "use_entity_filter": False},
    "hybrid_rrf_graph_nofilter": {"use_entity_filter": False},
}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def evaluate_retrieval(ctx: AppContext, cases: list[RetrievalCase], overrides: dict,
                       name: str = "retrieval", top_k: int = 10) -> EvalReport:
    base_cfg = ctx.retriever.cfg
    rows: list[dict] = []
    latencies: list[float] = []
    changes = {k: v for k, v in overrides.items() if k != "top_k"}
    try:
        ctx.retriever.cfg = replace(base_cfg, **changes) if changes else base_cfg
        for case in cases:
            result = ctx.retriever.retrieve(case.query, top_k=top_k)
            latencies.append(result.total_ms)
            rows.append({"ranked_persons": [h.metadata.get("person_name", "") for h in result.hits],
                         "gold": case.gold_person})
    finally:
        ctx.retriever.cfg = base_cfg

    return EvalReport(
        name=name,
        config={"overrides": overrides, "top_k": top_k,
                "embedder": ctx.embedder.name, "reranker": ctx.reranker.name},
        metrics=metrics.summarize(rows, ks=(1, 3, 5, 10)),
        latency_ms={"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95),
                    "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0},
        n=len(rows),
    )


def evaluate_ablations(ctx: AppContext, cases: list[RetrievalCase],
                       matrix: Optional[dict[str, dict]] = None,
                       top_k: int = 10) -> list[EvalReport]:
    matrix = matrix or DEFAULT_ABLATIONS
    reports = []
    for name, overrides in matrix.items():
        report = evaluate_retrieval(ctx, cases, overrides, name=name, top_k=top_k)
        reports.append(report)
        log.info("消融 %s → %s", name, report.metrics)
    return reports


def evaluate_mcq(ctx: AppContext, cases: list[MCQCase], method: str = "retrieval-grounded") -> EvalReport:
    """retrieval-grounded：把「人物 + 候选项」当查询，取命中该人物块的最高分作为候选得分。

    这是不消耗 LLM 额度的可复现基线；llm 方法需要配置真实 provider。
    """
    correct = 0
    details = []
    latencies = []
    for case in cases:
        t0 = time.perf_counter()
        if method == "retrieval-grounded":
            scores = []
            for option in case.options:
                result = ctx.retriever.retrieve(f"{case.person} {option}", top_k=5)
                own = [h for h in result.hits if h.metadata.get("person_name") == case.person]
                best = max((max(h.score, h.rerank_score or 0.0) for h in own), default=0.0)
                scores.append(best)
            pick = max(range(len(scores)), key=lambda i: scores[i])
        else:
            prompt = _mcq_prompt(case)
            answer = ctx.llm.chat([{"role": "user", "content": prompt}]).text
            ctx.tracker.record(_noop_result(answer, ctx))
            pick = _parse_option(answer, len(case.options))
        latencies.append(round((time.perf_counter() - t0) * 1000, 2))
        details.append({"id": case.id, "pick": pick, "answer": case.answer,
                        "ok": pick == case.answer})
        correct += int(pick == case.answer)

    return EvalReport(
        name=f"mcq:{method}",
        config={"method": method, "embedder": ctx.embedder.name, "llm": ctx.llm.name},
        metrics={"accuracy": round(correct / (len(cases) or 1), 4), "correct": correct},
        latency_ms={"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        n=len(cases),
        notes=[f"detail={sum(d['ok'] for d in details)}/{len(details)}"],
    )


def _mcq_prompt(case: MCQCase) -> str:
    lines = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(case.options))
    return f"{case.question}\n{lines}\n只输出选项字母。"


def _parse_option(text: str, n: int) -> int:
    import re

    m = re.search(r"[A-Z]", text or "")
    idx = ord(m.group(0)) - 65 if m else -1
    return idx if 0 <= idx < n else -1


def _noop_result(text, ctx):
    from ..llm.base import LLMResult, Usage

    return LLMResult(text=text, model=ctx.llm.name, provider=ctx.llm.name, usage=Usage())


def save_reports(db, settings: Settings, reports: list[EvalReport]) -> None:
    from ..db.models import EvalRun

    with db.session() as session:
        for report in reports:
            session.add(EvalRun(name=report.name, config=report.config,
                                metrics={**report.metrics, "latency_ms": report.latency_ms,
                                         "n": report.n},
                                notes="; ".join(report.notes)))
    log.info("评测结果写入 eval_runs: %d 条", len(reports))


def write_json(reports: list[EvalReport], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in reports]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
