"""Check every shipped full-corpus file and report actual biography/graph coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def inspect_corpus(directory: Path) -> dict:
    from mathkb.rag.chunking import load_directory

    files = sorted(p for p in directory.rglob("*") if p.is_file()
                   and p.parts[-2] != "uploads" and p.suffix in {".txt", ".json", ".jsonl"}
                   and p.name != "manifest.json")
    chunks = load_directory(directory / "biographies")
    persons = json.loads((directory / "kg/persons.json").read_text(encoding="utf-8"))
    relations = json.loads((directory / "kg/relations.json").read_text(encoding="utf-8"))
    biography_names = {c.person_name for c in chunks}
    graph_names = {p["name"] for p in persons}
    return {
        "biographies": len(list((directory / "biographies").glob("*.txt"))),
        "biography_persons": len(biography_names),
        "chunks": len(chunks),
        "graph_persons": len(persons),
        "relations": len(relations),
        "graph_only_persons": sorted(graph_names - biography_names),
        "biography_only_persons": sorted(biography_names - graph_names),
        "files": {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in files},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-dir", type=Path, help="also compare against the original local full corpus")
    args = parser.parse_args()
    directory = ROOT / "data/full"
    expected = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    actual = inspect_corpus(directory)
    if actual != expected:
        print("FAIL: full corpus does not match manifest", file=sys.stderr)
        return 1
    if args.compare_dir and actual != inspect_corpus(args.compare_dir):
        print("FAIL: full corpus differs from local reference", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", **{k: v for k, v in actual.items() if k != "files"},
                      "verified_files": len(actual["files"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
