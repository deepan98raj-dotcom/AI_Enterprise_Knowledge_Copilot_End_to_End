
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, Any]] = Field(default_factory=list)


class Source(BaseModel):
    document: str
    chunk_id: str
    page: int | None = None
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str
    retrieval_query: str = ""
    citation_check: dict[str, Any] = Field(default_factory=dict)
    grounding_check: dict[str, Any] = Field(default_factory=dict)
