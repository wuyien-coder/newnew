"""Load UTF-8 text/Markdown and text-based PDFs for asynchronous ingestion."""
from pathlib import Path
import re

from .chunking import Document, parse_document


def load_document(path: Path) -> Document:
    if path.suffix.lower() != ".pdf":
        return parse_document(path)
    from pypdf import PdfReader
    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF已加密，请先解密后上传")
    if len(reader.pages) > 300:
        raise ValueError("单个PDF最多支持300页，请拆分后上传")
    sections = []
    total = 0
    for number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        total += len(text)
        if total > 1_500_000:
            raise ValueError("PDF正文过长，请拆分后上传")
        if text:
            sections.append((f"PDF第{number}页", text))
    if not sections:
        raise ValueError("PDF没有可提取的文字，请先对扫描件进行OCR")
    title = re.sub(r"_[0-9a-f]{24}$", "", path.stem)
    return Document(title, meta={"source": f"用户上传 PDF：{title}",
                                "file_path": str(path.resolve()), "format": "pdf"},
                    sections=sections)
