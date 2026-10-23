
import re
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .query_rewriter import rewrite_query
from .retrieval import retrieve
from .llm import generate_answer
from .quality import check_citations, check_grounding


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict[str, Any]]
    retrieval_query: str
    sources: list[dict[str, Any]]
    retry_count: int
    weak_evidence: bool
    evidence_reason: str
    answer: str
    mode: str
    citation_check: dict[str, Any]
    grounding_check: dict[str, Any]


STOPWORDS = {
    "what", "when", "where", "which", "who", "why", "how",
    "the", "and", "for", "from", "with", "that", "this",
    "are", "was", "were", "is", "does", "did", "can",
    "could", "should", "would", "you", "your", "have",
    "has", "had", "about", "into", "their", "they",
    "tell", "explain", "summarize", "please",
}


def rewrite_node(state: RAGState) -> dict:
    question = state["question"]
    query = rewrite_query(question, state.get("history", []))

    return {
        "retrieval_query": query or question,
        "retry_count": 0,
    }


def retrieve_node(state: RAGState) -> dict:
    query = state.get("retrieval_query") or state["question"]

    return {
        "sources": retrieve(query, top_k=5)
    }


def assess_evidence_node(state: RAGState) -> dict:
    """Basic heuristic; not a guarantee of answer correctness."""
    sources = state.get("sources", [])

    query_text = (
        state.get("question", "") + " " +
        state.get("retrieval_query", "")
    ).lower()

    terms = {
        term
        for term in re.findall(r"\w+", query_text)
        if len(term) > 2 and term not in STOPWORDS
    }

    evidence_text = " ".join(
        source.get("text", "") for source in sources
    ).lower()

    matched_terms = {
        term for term in terms if term in evidence_text
    }

    weak = not sources or (bool(terms) and not matched_terms)

    if not sources:
        reason = "No passages were retrieved."
    elif weak:
        reason = "No query keywords matched the retrieved evidence."
    else:
        reason = "Some query keywords appear in retrieved evidence."

    return {
        "weak_evidence": weak,
        "evidence_reason": reason,
    }


def route_after_assessment(state: RAGState) -> str:
    if (
        state.get("weak_evidence", False)
        and state.get("retry_count", 0) < 1
    ):
        return "retry"

    return "generate"


def refine_query_node(state: RAGState) -> dict:
    question = state["question"]
    current_query = state.get("retrieval_query") or question

    history = list(state.get("history", []))
    history.append({
        "role": "assistant",
        "content": (
            "The previous retrieval appeared weak. "
            "Try a more specific search query for the user's question."
        ),
    })

    refined = rewrite_query(question, history).strip()

    if not refined or refined.casefold() == current_query.casefold():
        refined = (
            f"{question} relevant policy procedure requirements "
            "reporting steps"
        )

    return {
        "retrieval_query": refined,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def generate_answer_node(state: RAGState) -> dict:
    question = state["question"]
    sources = state.get("sources", [])

    answer, mode = generate_answer(
        question,
        sources,
        state.get("history", []),
    )

    return {
        "answer": answer,
        "mode": mode,
        "citation_check": check_citations(answer, len(sources)),
        "grounding_check": check_grounding(
            question,
            answer,
            sources,
        ),
    }


@lru_cache(maxsize=1)
def get_agentic_rag_graph():
    builder = StateGraph(RAGState)

    builder.add_node("rewrite_query", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("assess_evidence", assess_evidence_node)
    builder.add_node("refine_query", refine_query_node)
    builder.add_node("generate_answer", generate_answer_node)

    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "assess_evidence")

    builder.add_conditional_edges(
        "assess_evidence",
        route_after_assessment,
        {
            "retry": "refine_query",
            "generate": "generate_answer",
        },
    )

    builder.add_edge("refine_query", "retrieve")
    builder.add_edge("generate_answer", END)

    return builder.compile()


def run_agentic_rag(
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> dict:
    graph = get_agentic_rag_graph()

    result = graph.invoke({
        "question": question,
        "history": history or [],
    })

    sources = [
        {
            "document": source.get("document", "Unknown document"),
            "chunk_id": source.get("chunk_id", ""),
            "page": source.get("page"),
            "text": source.get("text", ""),
            "score": round(float(source.get("score", 0.0)), 4),
        }
        for source in result.get("sources", [])
    ]

    return {
        "answer": result.get("answer", ""),
        "sources": sources,
        "mode": result.get("mode", "unknown"),
        "retrieval_query": result.get("retrieval_query", question),
        "citation_check": result.get("citation_check", {}),
        "grounding_check": result.get("grounding_check", {}),
    }
