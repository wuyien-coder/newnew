"""把 data/seed 下的种子语料与图谱写入关系库（幂等，按主键 upsert）。"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..core.logging import get_logger
from ..rag.chunking import load_directory
from .models import Chunk, Document, Person, PersonField, Relation

log = get_logger(__name__)


def seed_knowledgebase(session: Session, settings: Settings, force: bool = False) -> dict:
    data_dir = Path(settings.data_dir)
    existing = session.scalar(select(Document).limit(1))
    if existing and not force:
        return {"skipped": True}

    counts = {"documents": 0, "chunks": 0, "persons": 0, "relations": 0}

    bio_dir = Path(settings.biographies_dir) if settings.biographies_dir else data_dir / "biographies"
    if bio_dir.is_dir():
        s = settings.retrieval
        for chunk in load_directory(bio_dir, chunk_size=s.chunk_size,
                                   chunk_overlap=s.chunk_overlap):
            doc = session.scalar(select(Document).where(Document.file_path == chunk.file_path))
            if not doc:
                doc = Document(person_name=chunk.person_name, source=chunk.source,
                               page=chunk.page, file_path=chunk.file_path)
                session.add(doc)
                session.flush()
                counts["documents"] += 1
            if session.scalar(select(Chunk).where(Chunk.id == chunk.chunk_id)):
                continue
            session.add(Chunk(id=chunk.chunk_id, document_id=doc.id,
                              person_name=chunk.person_name, section_title=chunk.section_title,
                              chunk_index=chunk.chunk_index, text=chunk.text,
                              token_count=len(chunk.text) // 2, metadata_=chunk.metadata()))
            doc.char_count += len(chunk.text)
            counts["chunks"] += 1
        session.flush()

    counts.update(seed_graph(session, data_dir / "kg"))
    log.info("种子数据入库: %s", counts)
    return counts


def seed_graph(session: Session, kg_dir: Path) -> dict:
    persons_file = kg_dir / "persons.json"
    relations_file = kg_dir / "relations.json"
    result = {"persons": 0, "relations": 0}
    if not persons_file.exists():
        log.warning("缺少图谱种子: %s", persons_file)
        return result

    persons = json.loads(persons_file.read_text(encoding="utf-8"))
    name_to_id: dict[str, int] = {}
    for item in persons:
        person = session.scalar(select(Person).where(Person.name == item["name"]))
        if not person:
            person = Person(
                name=item["name"], hometown=item.get("hometown", ""),
                birth_year=item.get("birth_year", ""), death_year=item.get("death_year", ""),
                achievements=item.get("achievements", ""),
                institutions=item.get("institutions", []),
            )
            session.add(person)
            session.flush()
            for field in item.get("fields", []):
                session.add(PersonField(person_id=person.id, field=field))
            result["persons"] += 1
        name_to_id[person.name] = person.id
    session.flush()

    if relations_file.exists():
        for item in json.loads(relations_file.read_text(encoding="utf-8")):
            exists = session.scalar(
                select(Relation).where(
                    Relation.from_name == item["from"], Relation.to_name == item["to"],
                    Relation.rel_type == item["type"],
                )
            )
            if exists:
                continue
            session.add(Relation(from_name=item["from"], to_name=item["to"],
                                 rel_type=item["type"]))
            result["relations"] += 1
        session.flush()
    return result
