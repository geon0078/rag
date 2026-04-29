"""POST /api/preview/* (운영웹통합명세서 §5 + §8.4).

HyDE 가상 답변 + retrieval + 답변 + Groundedness 결과를 한 번에 노출. 운영자가
편집 후 발행 전 검증하는 데 사용된다.

내부적으로 ``src.pipeline.RagPipeline`` 을 재사용 (docker-compose 의 PYTHONPATH
가 /workspace 를 포함하므로 backend 컨테이너에서 src/* import 가능).

Endpoints:
  POST /api/preview/search   — HyDE doc + retrieval top-K 청크
  POST /api/preview/answer   — 위 + 답변 생성 + verdict + grounded
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/preview", tags=["preview"])


_pipeline = None


def _get_pipeline():
    """Lazy import — backend 컨테이너 외부에서 backend 모듈만 테스트할 때 fail 안 하도록."""
    global _pipeline
    if _pipeline is None:
        try:
            from src.pipeline.rag_pipeline import RagPipeline  # type: ignore
        except ImportError as exc:
            raise HTTPException(
                503,
                detail=f"RagPipeline import 실패 — PYTHONPATH/볼륨 확인: {exc}",
            )
        _pipeline = RagPipeline()
    return _pipeline


class PreviewQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class PreviewSearchResp(BaseModel):
    query: str
    hyde_doc: str | None = None
    candidates: list[dict[str, Any]]


class PreviewAnswerResp(BaseModel):
    query: str
    answer: str
    grounded: bool
    verdict: str
    retry: bool
    sources: list[dict[str, Any]]
    elapsed_ms: int


@router.post("/search", response_model=PreviewSearchResp)
async def preview_search(req: PreviewQuery) -> PreviewSearchResp:
    """HyDE 가상 답변 + retrieval 결과 (생성/Groundedness 미수행)."""
    pipeline = _get_pipeline()
    hyde_doc: str | None = None
    try:
        hyde_doc = await pipeline.llm.hyde_expand(req.query)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        hyde_doc = None

    try:
        expanded = f"{req.query}\n\n{hyde_doc}" if hyde_doc else req.query
        candidates, _decision = await pipeline._retrieve_then_rerank(  # type: ignore[attr-defined]
            expanded,
            hybrid_top_k=30,
            rerank_top_k=req.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"retrieval 실패: {exc}")

    out = []
    for c in candidates:
        payload = c.get("payload") or {}
        out.append({
            "doc_id": c.get("doc_id"),
            "score": c.get("score"),
            "title": payload.get("title"),
            "category": payload.get("category") or payload.get("source_collection"),
            "campus": payload.get("campus"),
            "snippet": (c.get("contents") or payload.get("contents") or "")[:300],
        })

    return PreviewSearchResp(query=req.query, hyde_doc=hyde_doc, candidates=out)


@router.post("/answer", response_model=PreviewAnswerResp)
async def preview_answer(req: PreviewQuery) -> PreviewAnswerResp:
    """Full pipeline.run — 답변 + Groundedness."""
    pipeline = _get_pipeline()
    try:
        result = await pipeline.run(req.query)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"pipeline.run 실패: {exc}")

    return PreviewAnswerResp(
        query=req.query,
        answer=result.get("answer", ""),
        grounded=bool(result.get("grounded", False)),
        verdict=str(result.get("verdict", "")),
        retry=bool(result.get("retry", False)),
        sources=[
            {
                "doc_id": s.get("doc_id"),
                "score": s.get("score"),
                "category": s.get("category") or s.get("source_collection"),
                "campus": s.get("campus"),
            }
            for s in (result.get("sources") or [])[:5]
        ],
        elapsed_ms=int(result.get("elapsed_ms", 0)),
    )
