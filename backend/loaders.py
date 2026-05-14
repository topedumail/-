"""קריאת מסמכים מתיקיית הידע. תומך ב-txt, md, docx, pdf."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    source: str
    text: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_docx(path: Path) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def _read_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts: list[str] = []
    for slide_num, slide in enumerate(prs.slides, 1):
        slide_parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        slide_parts.append(text)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        slide_parts.append(" | ".join(cells))
        if slide_parts:
            parts.append(f"[שקופית {slide_num}]\n" + "\n".join(slide_parts))
    return "\n\n".join(parts)


_LOADERS = {
    ".txt": _read_text,
    ".md": _read_text,
    ".markdown": _read_text,
    ".docx": _read_docx,
    ".pdf": _read_pdf,
    ".pptx": _read_pptx,
}


def load_documents(folder: Path) -> list[Document]:
    """קורא את כל הקבצים הנתמכים בתיקייה ובתת-תיקיות.

    מסנן כפילויות לפי תוכן (אם אותו קובץ קיים בשתי תיקיות, רק אחד נכלל).
    """
    import hashlib

    seen_hashes: dict[str, str] = {}  # hash -> source שכבר נכלל
    docs: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        try:
            text = loader(path).strip()
        except Exception as exc:
            print(f"[loaders] לא ניתן לקרוא את {path.name}: {exc}")
            continue
        if not text:
            continue
        rel = path.relative_to(folder).as_posix()
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            print(f"[loaders] מדלג על כפילות: {rel} (זהה ל-{seen_hashes[content_hash]})")
            continue
        seen_hashes[content_hash] = rel
        docs.append(Document(source=rel, text=text))
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """פיצול טקסט לחלקים. עובד היטב בעברית — מפצל לפי פסקאות עם הצפה."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) > chunk_size:
                # פסקה ארוכה מאוד — פצל גס לפי גודל
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks


def build_chunks(docs: list[Document]) -> list[tuple[str, str]]:
    """מחזיר רשימה של (source, chunk_text)."""
    result: list[tuple[str, str]] = []
    for doc in docs:
        for chunk in chunk_text(doc.text):
            result.append((doc.source, chunk))
    return result
