
import os
import re
from functools import lru_cache
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from sentence_transformers import CrossEncoder

from .documents import BASE, load_records


# Persistent vector storage
CHROMA_DIR = BASE / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "enterprise_docs"

# Models can be changed through environment variables.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


@lru_cache(maxsize=1)
def get_collection():
    """Create or open the persistent ChromaDB collection."""
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def get_reranker():
    """Load the Cross-Encoder only when reranking is needed."""
    return CrossEncoder(
        RERANKER_MODEL,
        device="cpu",
        max_length=512,
    )


def tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25 keyword retrieval."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def sync_documents(collection, records):
    """
    Add JSON-indexed chunks to ChromaDB.
    This migrates documents indexed before semantic search was added.
    """
    existing_ids = set(
        collection.get(include=[])["ids"]
    )

    missing = [
        record
        for record in records
        if record.get("chunk_id")
        and record["chunk_id"] not in existing_ids
        and record.get("text", "").strip()
    ]

    batch_size = 100

    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]

        collection.upsert(
            ids=[r["chunk_id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[
                {
                    "document": str(r["document"]),
                    "page": int(r.get("page") or 0),
                }
                for r in batch
            ],
        )


def dense_search(query: str, collection, limit: int):
    """Retrieve candidates using semantic vector similarity."""
    total = collection.count()

    if total == 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(limit, total),
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    texts = result.get("documents", [[]])[0]
    metadata = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    candidates = []

    for rank, (chunk_id, text, meta, distance) in enumerate(
        zip(ids, texts, metadata, distances),
        start=1,
    ):
        meta = meta or {}

        candidates.append({
            "chunk_id": chunk_id,
            "document": meta.get("document", "Unknown document"),
            "page": meta.get("page") or None,
            "text": text or "",
            "dense_distance": float(distance),
            "rank": rank,
        })

    return candidates


def keyword_search(query: str, records, limit: int):
    """Retrieve candidates using BM25 keyword matching."""
    if not records:
        return []

    corpus = [
        tokenize(record.get("text", ""))
        for record in records
    ]
    query_tokens = tokenize(query)

    if not query_tokens or not any(corpus):
        return []

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(records)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )

    candidates = []

    for idx in ranked_indices:
        score = float(scores[idx])

        # Do not add passages with no keyword match.
        if score <= 0:
            continue

        candidates.append({
            **records[idx],
            "bm25_score": score,
        })

        if len(candidates) >= limit:
            break

    return candidates


def reciprocal_rank_fusion(
    dense_candidates,
    keyword_candidates,
    records_by_id,
    limit: int,
    rrf_constant: int = 60,
):
    """
    Combine dense and BM25 rankings using Reciprocal Rank Fusion.
    RRF scores are used for candidate ordering, not as probabilities.
    """
    combined = defaultdict(float)

    for rank, candidate in enumerate(dense_candidates, start=1):
        chunk_id = candidate["chunk_id"]
        combined[chunk_id] += 1.0 / (rrf_constant + rank)

    for rank, candidate in enumerate(keyword_candidates, start=1):
        chunk_id = candidate["chunk_id"]
        combined[chunk_id] += 1.0 / (rrf_constant + rank)

    ranked_ids = sorted(
        combined,
        key=lambda chunk_id: combined[chunk_id],
        reverse=True,
    )

    fused = []

    for chunk_id in ranked_ids[:limit]:
        record = records_by_id.get(chunk_id)

        if record is None:
            record = next(
                (
                    candidate
                    for candidate in dense_candidates
                    if candidate["chunk_id"] == chunk_id
                ),
                None,
            )

        if record is None:
            record = next(
                (
                    candidate
                    for candidate in keyword_candidates
                    if candidate["chunk_id"] == chunk_id
                ),
                None,
            )

        if record:
            fused.append({
                **record,
                "rrf_score": combined[chunk_id],
            })

    return fused


def rerank(query: str, candidates, top_k: int):
    """
    Rerank the fused candidates with a Cross-Encoder.
    If the reranker cannot load, use the RRF order as a fallback.
    """
    if not candidates:
        return []

    try:
        model = get_reranker()

        pairs = [
            (query, candidate.get("text", ""))
            for candidate in candidates
        ]

        scores = model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=16,
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = []

        for candidate, score in ranked[:top_k]:
            item = {
                key: value
                for key, value in candidate.items()
                if key not in {
                    "dense_distance",
                    "bm25_score",
                    "rrf_score",
                    "rank",
                }
            }
            item["score"] = float(score)
            results.append(item)

        return results

    except Exception as exc:
        print(
            "Cross-Encoder reranking unavailable; "
            f"using RRF order instead: {exc}"
        )

        results = []

        for candidate in candidates[:top_k]:
            item = {
                key: value
                for key, value in candidate.items()
                if key not in {
                    "dense_distance",
                    "bm25_score",
                    "rrf_score",
                    "rank",
                }
            }
            item["score"] = float(
                candidate.get("rrf_score", 0.0)
            )
            results.append(item)

        return results


def retrieve(query: str, top_k: int = 5):
    """
    End-to-end hybrid retrieval:
    1. Sync legacy JSON chunks.
    2. Retrieve semantic candidates.
    3. Retrieve BM25 candidates.
    4. Fuse rankings with RRF.
    5. Rerank and return the best passages.
    """
    if not query or not query.strip() or top_k < 1:
        return []

    query = query.strip()
    records = load_records()

    if not records:
        return []

    records_by_id = {
        r["chunk_id"]: r
        for r in records
        if r.get("chunk_id")
    }

    pool_size = max(20, top_k * 4)

    try:
        collection = get_collection()
        sync_documents(collection, records)

        dense_candidates = dense_search(
            query,
            collection,
            pool_size,
        )

    except Exception as exc:
        print(f"ChromaDB semantic search unavailable: {exc}")
        dense_candidates = []

    keyword_candidates = keyword_search(
        query,
        records,
        pool_size,
    )

    fused_candidates = reciprocal_rank_fusion(
        dense_candidates,
        keyword_candidates,
        records_by_id,
        limit=pool_size,
    )

    # Graceful fallback when neither retrieval route returns results.
    if not fused_candidates:
        return []

    return rerank(
        query,
        fused_candidates,
        top_k=top_k,
    )
