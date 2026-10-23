# AI Enterprise Knowledge Intelligence & Decision Copilot

An end-to-end portfolio project for an AI Engineer. Upload PDF/TXT documents, index their text, ask questions, receive source references, summarize documents, and compare files.

## Architecture
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Document extraction:** PyMuPDF (PDF), plain text
- **Retrieval:** TF-IDF similarity (runs locally; no embedding API required)
- **LLM generation:** OpenAI Responses API when `OPENAI_API_KEY` is configured; otherwise a grounded extractive fallback
- **Storage:** Local files + JSON metadata (starter implementation)

> This is a portfolio starter, not a production-ready enterprise system. For production, add authentication/RBAC, persistent vector storage, malware scanning, encrypted storage, rate limits, audit logging, and stronger prompt-injection defenses.

## Features
- Upload PDF/TXT files through the UI or API
- Extract and chunk document text
- Retrieve relevant chunks and show source/page references
- Ask questions with conversational follow-up context
- Summarize a document
- Compare two documents
- Health endpoint and API docs

## Quick start (Windows PowerShell)
```powershell
cd enterprise_knowledge_copilot
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Optionally add your OpenAI API key to `.env`:
```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```
Without a key, upload, retrieval, sources, and extractive answers still work.

Terminal 1 — backend:
```powershell
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000/docs

Terminal 2 — frontend:
```powershell
streamlit run frontend/app.py
```
Open the local Streamlit URL shown in the terminal.

## API endpoints
- `GET /health`
- `POST /documents/upload` — multipart field `file`
- `GET /documents` — indexed documents
- `POST /chat` — `{ "question": "...", "history": [] }`
- `POST /documents/{document_id}/summary`
- `POST /documents/compare` — `{ "document_a": "name.pdf", "document_b": "other.pdf" }`

## Example workflow
1. Upload a policy PDF or TXT file.
2. Ask: "What is the leave approval process?"
3. Inspect the sources returned with the answer.
4. Ask a follow-up question; the UI sends recent chat history.
5. Summarize one document or compare two documents.

## Folder structure
```text
enterprise_knowledge_copilot/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── __init__.py
│       ├── documents.py
│       ├── retrieval.py
│       └── llm.py
├── frontend/
│   └── app.py
├── data/
│   ├── uploads/.gitkeep
│   └── index/.gitkeep
├── tests/
│   └── test_api.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## Interview explanation
**Problem:** employees waste time finding answers in organizational documents.
**Solution:** parse documents, chunk text, retrieve relevant chunks, and generate grounded answers with source references.
**RAG flow:** Upload → Extract → Chunk → Retrieve → Prompt/LLM → Answer + Sources.
**Evaluation ideas:** retrieval hit-rate, answer faithfulness, citation correctness, latency, and cost per question.
