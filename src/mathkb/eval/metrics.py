"""检索与生成指标：纯函数实现，便于单测与复算。"""
from __future__ import annotations

import math
from typing import Sequence


def is_relevant(hit_person: str, gold_person: str) -> bool:
    return bool(gold_person) and hit_person == gold_person


def recall_at_k(ordered_persons: Sequence[str], gold_person: str, k: int) -> float:
    """gold 定义为「该人物的任一块」，因此 top-k 中命中任一即记 1（人物级召回）。"""
    top = list(ordered_persons)[:k]
    return 1.0 if any(is_relevant(p, gold_person) for p in top) else 0.0


def precision_at_k(ordered_persons: Sequence[str], gold_person: str, k: int) -> float:
    top = list(ordered_persons)[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if is_relevant(p, gold_person)) / k


def reciprocal_rank(ordered_persons: Sequence[str], gold_person: str) -> float:
    for i, person in enumerate(ordered_persons, 1):
        if is_relevant(person, gold_person):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ordered_persons: Sequence[str], gold_person: str, k: int) -> float:
    """二值相关性下的 nDCG：IDCG 按「返回列表中相关项个数」的理想排列计算，保证 ≤ 1。

    注意：本评测集的相关性是人物级弱标注（同一人物的块都算相关），
    因此 nDCG 衡量的是「gold 人物是否被排到前面」，不是段落级精准度。
    """
    top = list(ordered_persons)[:k]
    relevances = [1.0 if is_relevant(p, gold_person) else 0.0 for p in top]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def summarize(rows: list[dict], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """rows: [{"ranked_persons": [...], "gold": "..."}]"""
    n = len(rows) or 1
    out: dict[str, float] = {"n": len(rows)}
    for k in ks:
        out[f"recall@{k}"] = round(sum(recall_at_k(r["ranked_persons"], r["gold"], k)
                                        for r in rows) / n, 4)
        out[f"precision@{k}"] = round(sum(precision_at_k(r["ranked_persons"], r["gold"], k)
                                          for r in rows) / n, 4)
        out[f"ndcg@{k}"] = round(sum(ndcg_at_k(r["ranked_persons"], r["gold"], k)
                                     for r in rows) / n, 4)
    out["mrr"] = round(sum(reciprocal_rank(r["ranked_persons"], r["gold"]) for r in rows) / n, 4)
    return out
