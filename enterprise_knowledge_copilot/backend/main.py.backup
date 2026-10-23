import os, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .schemas import ChatRequest, ChatResponse
from .services.documents import UPLOADS, safe_name, index_document, list_documents, get_document_records
from .services.retrieval import retrieve
from .services.llm import generate_answer
from .services.query_rewriter import rewrite_query
from .services.quality import check_citations, check_grounding
from .services.agentic_rag import run_agentic_rag


load_dotenv()
app = FastAPI(title="AI Enterprise Knowledge Copilot", version="1.0.0",
              description="Document RAG assistant with source references.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "enterprise-knowledge-copilot"}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    name = safe_name(file.filename or "document.txt")
    if Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise HTTPException(400, "Only PDF, TXT, and MD files are supported.")
    max_mb = int(os.getenv("MAX_UPLOAD_MB", "20"))
    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(content) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {max_mb} MB limit.")
    # Prefix stored filename to avoid collisions while preserving display name in metadata.
    stored = UPLOADS / f"{uuid.uuid4().hex[:8]}_{name}"
    stored.write_bytes(content)
    try:
        result = index_document(name, stored)
        return {"status": "indexed", **result}
    except ValueError as e:
        stored.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    except Exception as e:
        stored.unlink(missing_ok=True)
        raise HTTPException(500, f"Could not process document: {e}")

@app.get("/documents")
def documents():
    return {"documents": list_documents()}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    # 1. Rewrite follow-up questions for document retrieval.
    retrieval_query = rewrite_query(
        payload.question,
        payload.history,
    )

    # 2. Retrieve passages using hybrid search and reranking.
    sources = retrieve(retrieval_query, top_k=5)

    # 3. Generate an answer using the original user question.
    answer, mode = generate_answer(
        payload.question,
        sources,
        payload.history,
    )

    # 4. Check whether citation numbers refer to retrieved sources.
    citation_check = check_citations(
        answer,
        len(sources),
    )

    # 5. Optionally run an additional LLM grounding assessment.
    grounding_check = check_grounding(
        payload.question,
        answer,
        sources,
    )

    clean_sources = [
        {
            "document": source["document"],
            "chunk_id": source["chunk_id"],
            "page": source.get("page"),
            "text": source["text"],
            "score": round(float(source["score"]), 4),
        }
        for source in sources
    ]

    return {
        "answer": answer,
        "sources": clean_sources,
        "mode": mode,
        "retrieval_query": retrieval_query,
        "citation_check": citation_check,
        "grounding_check": grounding_check,
    }


@app.post("/documents/{document_name}/summary")
def summarize(document_name: str):
    records = get_document_records(document_name)
    if not records:
        raise HTTPException(404, "Document not found.")
    text = "\n".join(r["text"] for r in records)
    answer, mode = generate_answer(
        f"Summarize this document. Organize the summary into purpose, key points, decisions/policies, and action items. Document text: {text[:18000]}",
        [{"document": document_name, "page": None, "text": text[:18000], "score": 1.0}], []
    )
    return {"document": document_name, "summary": answer, "mode": mode}

@app.post("/documents/compare")
def compare_documents(payload: dict):
    a, b = payload.get("document_a"), payload.get("document_b")
    if not a or not b:
        raise HTTPException(400, "Provide document_a and document_b.")
    ra, rb = get_document_records(a), get_document_records(b)
    if not ra or not rb:
        raise HTTPException(404, "One or both documents were not found.")
    context = [
        {"document": a, "page": None, "text": "\n".join(r["text"] for r in ra)[:9000], "score": 1.0},
        {"document": b, "page": None, "text": "\n".join(r["text"] for r in rb)[:9000], "score": 1.0}
    ]
    answer, mode = generate_answer(
        f"Compare these two documents. Show common points, differences, conflicts, and unanswered questions. Document A: {a}; Document B: {b}.",
        context, []
    )
    return {"document_a": a, "document_b": b, "comparison": answer, "mode": mode}
