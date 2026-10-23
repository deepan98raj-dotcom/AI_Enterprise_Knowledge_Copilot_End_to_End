import os, re, json, uuid
from pathlib import Path
import fitz

BASE = Path(os.getenv("DATA_DIR", "data"))
UPLOADS = BASE / "uploads"
INDEX = BASE / "index"
UPLOADS.mkdir(parents=True, exist_ok=True)
INDEX.mkdir(parents=True, exist_ok=True)

def safe_name(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:150] or "document"

def extract_pages(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = []
        with fitz.open(path) as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page": i + 1, "text": text})
        return pages
    if suffix in {".txt", ".md"}:
        return [{"page": 1, "text": path.read_text(encoding="utf-8", errors="ignore")}]
    raise ValueError("Only PDF, TXT, and MD files are supported.")

def chunk_text(text: str, size: int = 900, overlap: int = 140):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(" ", start + int(size * .65), end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def index_document(filename: str, stored_path: Path):
    pages = extract_pages(stored_path)
    records = []
    for page_item in pages:
        for idx, chunk in enumerate(chunk_text(page_item["text"])):
            records.append({
                "chunk_id": str(uuid.uuid4()),
                "document": filename,
                "page": page_item["page"],
                "text": chunk
            })
    if not records:
        raise ValueError("No extractable text found. If this is a scanned PDF, add OCR support.")
    out = INDEX / f"{stored_path.stem}.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"document": filename, "chunks": len(records), "pages_with_text": len(pages)}

def load_records():
    records = []
    for p in INDEX.glob("*.json"):
        try:
            records.extend(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return records

def list_documents():
    names = sorted({r["document"] for r in load_records()})
    return [{"name": n, "chunks": sum(1 for r in load_records() if r["document"] == n)} for n in names]

def get_document_records(name: str):
    return [r for r in load_records() if r["document"] == name]
