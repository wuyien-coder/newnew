from __future__ import annotations

from mathkb.rag.kg_store import GraphPerson, GraphStore


def build() -> GraphStore:
    persons = [
        GraphPerson("熊庆来", hometown="云南弥勒", birth_year="1893", fields=["复变函数论"]),
        GraphPerson("华罗庚", hometown="江苏金坛", birth_year="1910",
                    fields=["解析数论", "多复变函数论"]),
        GraphPerson("陈省身", hometown="浙江嘉兴", birth_year="1911", fields=["微分几何"]),
        GraphPerson("段学复", hometown="江苏苏州", birth_year="1914", fields=["数论"]),
    ]
    relations = [("华罗庚", "熊庆来", "teacher"), ("段学复", "熊庆来", "student")]
    return GraphStore(persons, relations)


def test_neighbors_direction_and_label():
    store = build()
    triples = store.neighbors("华罗庚")
    assert ("出", "师从", "熊庆来") in triples


def test_triples_are_prompt_ready():
    lines = build().triples_for(["华罗庚"])
    assert any(line.startswith("[图谱-实体]") for line in lines)
    assert any("师从" in line for line in lines)


def test_by_field_matches_field_inside_query():
    store = build()
    assert "华罗庚" in store.by_field("研究数论的有哪些人")


def test_by_hometown_strips_ji_suffix():
    store = build()
    assert set(store.by_hometown("江苏籍的数学家")) == {"华罗庚", "段学复"}


def test_path_between_two_entities():
    store = build()
    assert store.path("华罗庚", "段学复") == ["华罗庚", "熊庆来", "段学复"]


def test_unknown_entity_is_safe():
    store = build()
    assert store.neighbors("不存在的人") == []
    assert store.path("华罗庚", "不存在") == []
