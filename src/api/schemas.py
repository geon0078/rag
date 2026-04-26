"""Pydantic request / response schemas for the RAG API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    campus: str | None = None
    use_cache: bool = True


class SourceItem(BaseModel):
    doc_id: str | None = None
    category: str | None = None
    subcategory: str | None = None
    campus: str | None = None
    source_collection: str | None = None
    title: str | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    verdict: str | None = None
    sources: list[SourceItem] = Field(default_factory=list)
    retry: bool = False
    cached: bool = False
    similarity: float | None = None
    # Surfaces the campus actually used for filtering. ``campus_was_inferred``
    # is True when the router fell back to the configured default campus
    # (settings.default_campus) because the query had no explicit signal.
    resolved_campus: str | None = None
    campus_was_inferred: bool = False
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    redis: bool
    components: dict[str, bool]


class StatsResponse(BaseModel):
    qdrant: dict[str, Any] = Field(default_factory=dict)
    bm25: dict[str, Any] = Field(default_factory=dict)
    cache: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
