from __future__ import annotations

from mathkb.eval import metrics


def test_recall_and_mrr_on_perfect_ranking():
    rows = [{"ranked_persons": ["华罗庚", "陈省身"], "gold": "华罗庚"}]
    summary = metrics.summarize(rows, ks=(1, 2))
    assert summary["recall@1"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["precision@1"] == 1.0


def test_mrr_uses_first_hit_position():
    assert metrics.reciprocal_rank(["a", "b", "gold", "gold"], "gold") == 1 / 3


def test_miss_scores_zero():
    assert metrics.recall_at_k(["a", "b"], "gold", 2) == 0.0
    assert metrics.ndcg_at_k(["a", "b"], "gold", 2) == 0.0


def test_ndcg_stays_within_unit_interval():
    """多条 gold 相关时 nDCG 仍应 ≤ 1（曾按单条理想值归一，算出 4.5 的荒谬值）。"""
    rows = [{"ranked_persons": ["gold"] * 10, "gold": "gold"}]
    assert metrics.summarize(rows, ks=(5,))["ndcg@5"] == 1.0


def test_ndcg_penalizes_late_hits():
    early = metrics.ndcg_at_k(["gold", "x", "x", "x", "x"], "gold", 5)
    late = metrics.ndcg_at_k(["x", "x", "x", "x", "gold"], "gold", 5)
    assert early > late
