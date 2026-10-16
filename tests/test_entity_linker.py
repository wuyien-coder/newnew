from __future__ import annotations

from mathkb.rag.entity_linker import EntityLinker, core_name, normalize_name


def test_name_helpers_strip_parenthesis_and_fullwidth():
    assert normalize_name("曾 昭安（字箴益）") == "曾昭安(字箴益)"
    assert core_name("华罗庚（字浩昆）") == "华罗庚"


def test_links_full_name_and_core_name():
    linker = EntityLinker(["华罗庚", "陈省身", "苏步青"])
    assert linker.link("华罗庚在数论上的工作") == ["华罗庚"]
    assert linker.link("陈省身和苏步青谁更早留学？") == ["陈省身", "苏步青"] or \
        set(linker.link("陈省身和苏步青谁更早留学？")) == {"陈省身", "苏步青"}


def test_prefers_longer_name_over_substring():
    linker = EntityLinker(["王元", "王元深"])
    assert linker.link("王元深的学生是谁") == ["王元深"]


def test_no_match_returns_empty():
    linker = EntityLinker(["华罗庚"])
    assert linker.link("微分几何在中国的发展") == []


def test_add_registers_new_candidate():
    linker = EntityLinker(["华罗庚"])
    linker.add("吴文俊")
    assert "吴文俊" in linker.link("吴文俊的拓扑研究")
