"""FastAPI server: non-streaming + SSE streaming RAG endpoints with Redis semantic cache."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from src.cache.redis_cache import (
    TTL_CLASS_CALENDAR,
    TTL_CLASS_DEFAULT,
    SemanticCache,
)
from src.config import settings
from src.generation.prompts import format_context
from src.pipeline.rag_pipeline import RagPipeline, _sources
from src.retrieval.router import route
from src.utils.logger import get_logger
from src.utils.telemetry import record_query

log = get_logger(__name__)

CALENDAR_COLLECTION = "학사일정"


def _ttl_class_for(query: str) -> str:
    """Pick a cache TTL class from the YAML-driven router (no hardcoded keywords)."""
    decision = route(query)
    if CALENDAR_COLLECTION in decision.boosts:
        return TTL_CLASS_CALENDAR
    return TTL_CLASS_DEFAULT


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup: warming pipeline + cache")
    app.state.pipeline = RagPipeline()
    app.state.cache = SemanticCache()
    redis_ok = await app.state.cache.ping()
    log.info(f"startup ready: redis={redis_ok}")
    try:
        yield
    finally:
        log.info("shutdown: closing cache connection")
        await app.state.cache.close()


app = FastAPI(
    title="을지대 RAG API",
    description="EulJi University Korean RAG chatbot (Solar + Qdrant + BM25-Okt + bge-reranker)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _pipeline() -> RagPipeline:
    pipe = getattr(app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return pipe


def _cache() -> SemanticCache:
    cache = getattr(app.state, "cache", None)
    if cache is None:
        raise HTTPException(status_code=503, detail="cache not ready")
    return cache


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cache = _cache()
    redis_ok = await cache.ping()
    pipe = getattr(app.state, "pipeline", None)
    components = {
        "pipeline": pipe is not None,
        "retriever": pipe is not None and pipe.retriever is not None,
        "reranker": pipe is not None and pipe.reranker is not None,
        "llm": pipe is not None and pipe.llm is not None,
        "groundedness": pipe is not None and pipe.groundedness is not None,
    }
    healthy = redis_ok and all(components.values())
    return HealthResponse(
        status="ok" if healthy else "degraded",
        redis=redis_ok,
        components=components,
    )


@app.get("/api/v1/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    pipe = _pipeline()
    cache = _cache()

    qdrant_info: dict = {}
    try:
        qdrant_info = {
            "collection": pipe.retriever.store.collection,
            "doc_count": pipe.retriever.store.count(),
        }
    except Exception as exc:
        qdrant_info = {"error": str(exc)}

    bm25_info: dict = {
        "loaded": pipe.retriever.bm25.bm25 is not None,
        "doc_count": len(getattr(pipe.retriever.bm25, "doc_ids", []) or []),
        "tokenizer": settings.bm25_tokenizer,
    }

    cache_info = await cache.stats()

    config_info = {
        "top_k_dense": settings.top_k_dense,
        "top_k_sparse": settings.top_k_sparse,
        "top_k_rerank_final": settings.top_k_rerank_final,
        "hybrid_method": settings.hybrid_method,
        "hybrid_cc_weight": settings.hybrid_cc_weight,
        "reranker_model": settings.reranker_model,
        "reranker_enabled": settings.reranker_enabled,
        "llm_model": settings.llm_model_pro,
        "cache_similarity_threshold": settings.cache_similarity_threshold,
    }

    return StatsResponse(
        qdrant=qdrant_info,
        bm25=bm25_info,
        cache=cache_info,
        config=config_info,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    pipe = _pipeline()
    cache = _cache()
    t0 = time.time()

    if req.use_cache:
        hit = await cache.lookup(req.query)
        if hit is not None:
            elapsed_ms = int((time.time() - t0) * 1000)
            record_query(
                req.query,
                {
                    "sources": hit.get("sources", []),
                    "grounded": hit["grounded"],
                    "verdict": hit.get("verdict"),
                    "retry": False,
                    "elapsed_ms": elapsed_ms,
                },
                cached=True,
                similarity=hit.get("similarity"),
            )
            return QueryResponse(
                answer=hit["answer"],
                grounded=hit["grounded"],
                verdict=hit.get("verdict"),
                sources=hit.get("sources", []),
                retry=False,
                cached=True,
                similarity=hit.get("similarity"),
                elapsed_ms=elapsed_ms,
            )

    result = await pipe.run(req.query)
    record_query(req.query, result, cached=False, similarity=None)

    if result.get("grounded"):
        await cache.store(
            req.query,
            payload={
                "answer": result["answer"],
                "sources": result["sources"],
                "grounded": result["grounded"],
                "verdict": result.get("verdict"),
            },
            ttl_class=_ttl_class_for(req.query),
        )

    return QueryResponse(
        answer=result["answer"],
        grounded=result["grounded"],
        verdict=result.get("verdict"),
        sources=result.get("sources", []),
        retry=result.get("retry", False),
        cached=False,
        similarity=None,
        resolved_campus=result.get("resolved_campus"),
        campus_was_inferred=result.get("campus_was_inferred", False),
        elapsed_ms=result.get("elapsed_ms", int((time.time() - t0) * 1000)),
    )


async def _stream_events(query: str, use_cache: bool) -> AsyncIterator[dict]:
    pipe = _pipeline()
    cache = _cache()
    t0 = time.time()

    if use_cache:
        hit = await cache.lookup(query)
        if hit is not None:
            elapsed_ms = int((time.time() - t0) * 1000)
            yield {
                "event": "meta",
                "data": json.dumps(
                    {"cached": True, "similarity": hit.get("similarity")},
                    ensure_ascii=False,
                ),
            }
            yield {"event": "token", "data": hit["answer"]}
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "grounded": hit["grounded"],
                        "verdict": hit.get("verdict"),
                        "sources": hit.get("sources", []),
                        "elapsed_ms": elapsed_ms,
                        "cached": True,
                    },
                    ensure_ascii=False,
                ),
            }
            record_query(
                query,
                {
                    "sources": hit.get("sources", []),
                    "grounded": hit["grounded"],
                    "verdict": hit.get("verdict"),
                    "retry": False,
                    "elapsed_ms": elapsed_ms,
                },
                cached=True,
                similarity=hit.get("similarity"),
            )
            return

    candidates, _decision = await pipe._retrieve_then_rerank(
        query,
        hybrid_top_k=settings.top_k_dense,
        rerank_top_k=settings.top_k_rerank_final,
    )

    yield {
        "event": "meta",
        "data": json.dumps({"cached": False, "candidates": len(candidates)}),
    }

    chunks: list[str] = []
    async for delta in pipe.llm.stream(query, candidates):
        chunks.append(delta)
        yield {"event": "token", "data": delta}

    answer = "".join(chunks).strip()
    verdict = await pipe.groundedness.verify(format_context(candidates), answer)
    sources = _sources(candidates)
    grounded = verdict == "grounded"

    if grounded:
        await cache.store(
            query,
            payload={
                "answer": answer,
                "sources": sources,
                "grounded": True,
                "verdict": verdict,
            },
            ttl_class=_ttl_class_for(query),
        )

    elapsed_ms = int((time.time() - t0) * 1000)
    record_query(
        query,
        {
            "sources": sources,
            "grounded": grounded,
            "verdict": verdict,
            "retry": False,
            "elapsed_ms": elapsed_ms,
        },
        cached=False,
        similarity=None,
    )

    yield {
        "event": "done",
        "data": json.dumps(
            {
                "grounded": grounded,
                "verdict": verdict,
                "sources": sources,
                "elapsed_ms": elapsed_ms,
                "cached": False,
            },
            ensure_ascii=False,
        ),
    }


@app.post("/api/v1/query/stream")
async def query_stream(req: QueryRequest) -> EventSourceResponse:
    return EventSourceResponse(_stream_events(req.query, req.use_cache))
