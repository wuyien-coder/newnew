"""从论文工程（只读）生成知识库种子与可复现评测集。

两种规模：
  --mode sample（默认）裁剪 14 位人物到 data/seed/，传记复制进本仓库，方便演示与提交；
  --mode full     全量 265 位人物：图谱与评测集写到 data/full/，
                  **传记原文不复制**（仍留在论文目录，由 MATHKB_BIOGRAPHIES_DIR 指过去），
                  避免把整本传记的文本带进 git。

用法：
    PYTHONPATH=src python scripts/extract_seed.py --source "<论文工程目录>" --mode full
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 演示样本：覆盖数论/微分几何/概率统计/计算数学等领域，传记篇幅适中
SAMPLE_PERSONS = [
    "华罗庚", "陈省身", "苏步青", "吴文俊", "段学复", "关肇直", "冯康", "柯召",
    "李国平", "江泽涵", "熊庆来", "许宝騄", "陈建功", "傅种孙",
]

# 聚合类问句里的通用词
_GENERIC_RE = re.compile(r"(哪些|多少人|几个人|从事|研究|领域|方向|籍贯|哪里|什么|请问|介绍一下|有|的|是|在|和|与|了|吗|呢|[，。？！、\s])")


def copy_biographies(source: Path, out: Path, names: list[str]) -> list[str]:
    src_dir = source / "text_organized"
    dst_dir = out / "biographies"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    copied = []
    for name in names:
        candidate = src_dir / f"{name}.txt"
        if candidate.exists():
            shutil.copyfile(candidate, dst_dir / f"{name}.txt")
            copied.append(name)
    return copied


def _names(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    out = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name") or item.get("person") or ""
        item = str(item).strip()
        if item and item not in ("无", "/", "无正式数学老师"):
            out.append(item)
    return out


def _year(text: str) -> str:
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(text))
    return m.group(1) if m else ""


def extract_graph(source: Path, out: Path, bio_names: list[str],
                  full: bool) -> tuple[list[dict], list[dict]]:
    kg = json.loads((source / "kg_cache" / "knowledge_graph.json").read_text(encoding="utf-8"))
    persons = kg["persons"]
    if full:
        keep = set(persons)
    else:
        keep = {n for n in bio_names if n in persons}
        for name in list(keep):  # 1 跳邻居：保证师承查询在样本内可答
            for rel_key in ("teachers", "students", "advisor"):
                for other in _names(persons[name].get(rel_key)):
                    if other in persons:
                        keep.add(other)

    subset = []
    for name in sorted(keep):
        p = persons[name]
        subset.append({
            "name": name,
            "hometown": p.get("hometown", ""),
            "birth_year": _year(p.get("birth_date", "")),
            "death_year": "",
            "fields": [f for f in p.get("research_fields", []) if f],
            "institutions": [i for i in p.get("institutions", []) if i],
            "achievements": (p.get("achievements") or "")[:400],
        })

    relations, seen = [], set()
    for name in sorted(keep):
        p = persons[name]
        for teacher in _names(p.get("teachers")) + _names(p.get("advisor")):
            if teacher in keep and (name, teacher, "teacher") not in seen:
                seen.add((name, teacher, "teacher"))
                relations.append({"from": name, "to": teacher, "type": "teacher"})
        for student in _names(p.get("students")):
            if student in keep and (name, student, "student") not in seen:
                seen.add((name, student, "student"))
                relations.append({"from": name, "to": student, "type": "student"})

    (out / "kg").mkdir(parents=True, exist_ok=True)
    (out / "kg" / "persons.json").write_text(json.dumps(subset, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    (out / "kg" / "relations.json").write_text(json.dumps(relations, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    return subset, relations


def _tokens(text: str) -> list[str]:
    stripped = _GENERIC_RE.sub(" ", text or "")
    parts = [p for p in stripped.split() if len(p) >= 2]
    return list(dict.fromkeys([p for part in parts for p in (part, re.sub(r"[籍省市]$", "", part))
                               if len(p) >= 2]))


def build_eval_sets(persons: list[dict], bio_dir: Path, out: Path, copied: list[str]) -> None:
    """三个可复现评测集（生成规则写在 docs/ARCHITECTURE.md §5）：

    - qa/retrieval.jsonl        查询含人名，gold=该人物（弱标注，验证产品行为，会饱和）
    - qa/retrieval_hard.jsonl   去名化传记原句，gold=该人物（排序质量集）
    - qa/mcq.jsonl              由结构化字段生成的四选一，干扰项取他人字段值
    """
    by_name = {p["name"]: p for p in persons}
    rng = random.Random(20261016)
    qa_dir = out / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    named, hard, mcq = [], [], []
    for name in copied:
        p = by_name.get(name)
        if not p:
            continue
        if p["fields"]:
            named.append({"id": f"r{len(named) + 1}", "query": f"{name}的主要学术贡献是什么",
                          "gold_person": name, "gold_source": "KG结构化字段"})
            named.append({"id": f"r{len(named) + 1}", "query": f"{name}在{p['fields'][0]}方面的工作",
                          "gold_person": name, "gold_source": "KG结构化字段"})
        if p["hometown"]:
            named.append({"id": f"r{len(named) + 1}", "query": f"{name}是哪里人",
                          "gold_person": name, "gold_source": "KG结构化字段"})

        path = bio_dir / f"{name}.txt"
        if path.exists():
            body = path.read_text(encoding="utf-8").split("=" * 20)[-1]
            sentences = [s.strip() for s in re.split(r"(?<=[。！？])", body) if 30 <= len(s.strip()) <= 110]
            picks = [s for s in sentences if re.search(r"1[89]\d{2}|20\d{2}", s)][:2] or sentences[10:12]
            for sent in picks:
                query = re.sub(r"\s+", "", sent.replace(name, ""))
                if len(query) >= 15:
                    hard.append({"id": f"h{len(hard) + 1}", "query": query[:90],
                                 "gold_person": name, "gold_source": "去名化传记原句（自动生成，人工可复核）"})

        if p["fields"]:
            others = [o for n, o in by_name.items() if n != name and o["fields"]]
            if others:
                distract = rng.sample(others, min(3, len(others)))
                options = [p["fields"][0]] + [d["fields"][0] for d in distract[:3]]
                rng.shuffle(options)
                mcq.append({"id": f"m{len(mcq) + 1}", "person": name,
                            "question": f"根据传记语料，{name}的主要研究方向属于以下哪一项？",
                            "options": options, "answer": options.index(p["fields"][0]),
                            "gold_field": p["fields"][0]})

    _write_jsonl(qa_dir / "retrieval.jsonl", named)
    _write_jsonl(qa_dir / "retrieval_hard.jsonl", hard)
    _write_jsonl(qa_dir / "mcq.jsonl", mcq)
    print(f"[seed] 评测集: named={len(named)} hard={len(hard)} mcq={len(mcq)}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="论文工程根目录（只读）")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--max-persons", type=int, default=0, help="full 模式下限制人物数（0=全部）")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"源目录不存在: {source}")
        return 1

    full = args.mode == "full"
    out = ROOT / "data" / ("full" if full else "seed")

    if full:
        bio_dir = source / "text_organized"
        names = sorted({p.stem for p in bio_dir.glob("*.txt")})
        names = [n for n in names if not n.endswith("_2")]
        if args.max_persons:
            names = names[: args.max_persons]
        copied = names
    else:
        copied = copy_biographies(source, out, SAMPLE_PERSONS)
        bio_dir = out / "biographies"
        names = copied

    subset, relations = extract_graph(source, out, names, full)
    build_eval_sets(subset, bio_dir, out, names)

    print(f"[seed] mode={args.mode} 传记={len(names)} 图谱人物={len(subset)} 关系={len(relations)}")
    print(f"[seed] 输出目录: {out}")
    if full:
        print("[seed] 传记原文未复制，启动前请设置 "
              f"MATHKB_BIOGRAPHIES_DIR={bio_dir} 与 MATHKB_DATA_DIR={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
