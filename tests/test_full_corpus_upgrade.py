"""A seed database must gain full-corpus records when explicitly synchronized."""
from pathlib import Path
import shutil

from sqlalchemy import func, select

from mathkb.db.models import Document, Person
from mathkb.db.seed import seed_knowledgebase
from mathkb.services.container import AppContext
from tests.conftest import make_settings


def test_sync_adds_missing_biographies_and_graph_without_losing_existing_data(tmp_path):
    settings = make_settings(tmp_path)
    ctx = AppContext.create(settings)
    try:
        with ctx.db.session() as session:
            original_docs = session.scalar(select(func.count(Document.id)))
            original_people = session.scalar(select(func.count(Person.id)))
            session.add(Document(person_name="用户上传", file_path="memory://user-upload"))
        full = tmp_path / "full"
        shutil.copytree(Path(settings.data_dir), full)
        (full / "biographies/新增人物.txt").write_text(
            "人物：新增人物\n来源：升级回归测试\n====\n新增人物研究代数几何，并有完整传记内容。", encoding="utf-8")
        import json
        persons_file = full / "kg/persons.json"
        persons = json.loads(persons_file.read_text(encoding="utf-8"))
        persons.append({"name": "新增人物", "fields": ["代数几何"]})
        persons_file.write_text(json.dumps(persons, ensure_ascii=False), encoding="utf-8")
        # Keep existing paths as in a deployed data directory and add one new file.
        shutil.copy2(full / "biographies/新增人物.txt", Path(settings.data_dir) / "biographies/新增人物.txt")
        shutil.copy2(persons_file, Path(settings.data_dir) / "kg/persons.json")
        with ctx.db.session() as session:
            changes = seed_knowledgebase(session, settings, force=True)
        assert changes["documents"] == 1
        assert changes["persons"] == 1
        with ctx.db.session() as session:
            assert session.scalar(select(func.count(Document.id))) == original_docs + 2
            assert session.scalar(select(func.count(Person.id))) == original_people + 1
            again = seed_knowledgebase(session, settings, force=True)
        assert all(again[k] == 0 for k in ("documents", "chunks", "persons", "relations"))
    finally:
        ctx.close()
