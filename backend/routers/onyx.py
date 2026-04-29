"""Onyx Frontend 호환 어댑터 (onyx+docmost 개발.md §5).

학생 챗봇 (Onyx fork 또는 자체 chat UI) 가 이 엔드포인트들을 호출한다.
우리 RagPipeline 위에 Onyx 형식 변환 레이어만 추가.

Endpoints:
  POST /api/onyx/chat             — 채팅 메시지 (JSON 응답)
  POST /api/onyx/chat/stream      — 채팅 메시지 SSE 스트리밍
  POST /api/onyx/search           — retrieval 만 (답변 없음)
  GET  /api/onyx/sessions         — 채팅 세션 목록
  POST /api/onyx/sessions/create  — 새 채팅 세션
  GET  /api/onyx/persona          — 단일 persona (을지대 학사 도우미)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/onyx", tags=["onyx"])


_pipeline = None
_sessions: dict[str, dict[str, Any]] = {}  # in-mem session store; PostgreSQL TODO


def _get_pipeline():
    """Lazy import — 백엔드 워크스페이스 마운트가 있어야 동작."""
    global _pipeline
    if _pipeline is None:
        try:
            from src.pipeline.rag_pipeline import RagPipeline  # type: ignore
        except ImportError as exc:
            raise HTTPException(503, detail=f"RagPipeline import 실패: {exc}")
        _pipeline = RagPipeline()
    return _pipeline


# ────────────────────────────────────────────────────────────
# Pydantic models — Onyx-style request/response shapes
# ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    chat_session_id: str | None = None
    persona_id: int | None = 0


class CitationDoc(BaseModel):
    document_id: str
    link: str | None = None
    source_type: str | None = None
    semantic_identifier: str | None = None
    blurb: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    chat_session_id: str
    message_id: str
    answer: str
    citations: list[CitationDoc] = []
    grounded: bool = True
    verdict: str = "grounded"
    elapsed_ms: int = 0


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[CitationDoc]


class SessionCreateResponse(BaseModel):
    chat_session_id: str
    persona_id: int = 0


class SessionListResponse(BaseModel):
    sessions: list[dict[str, Any]]


class Persona(BaseModel):
    id: int
    name: str
    description: str
    is_default: bool = True


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────


def _docmost_link(doc_id: str) -> str | None:
    """청크 doc_id 가 docmost_page_id 메타를 가지면 docmost 링크 생성.

    onyx+docmost 개발.md §5 Day 5: '출처 클릭 시 Docmost 페이지로 이동'.
    환경변수 DOCMOST_BASE_URL 가 없으면 None.
    """
    base = os.environ.get("DOCMOST_BASE_URL", "http://localhost:3001")
    return f"{base}/p/{doc_id}" if base else None


def _to_citation(source: dict[str, Any]) -> CitationDoc:
    payload = source.get("payload") or {}
    doc_id = source.get("doc_id") or payload.get("doc_id") or "?"
    return CitationDoc(
        document_id=str(doc_id),
        link=_docmost_link(str(doc_id)),
        source_type=payload.get("source_collection") or payload.get("category"),
        semantic_identifier=payload.get("title") or payload.get("path"),
        blurb=(source.get("contents") or payload.get("contents") or "")[:200] or None,
        score=source.get("score"),
    )


# ────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────


@router.get("/persona")
async def get_persona() -> Persona:
    """단일 default persona — 을지대 학사 도우미."""
    return Persona(
        id=0,
        name="을지대 학사 도우미",
        description="을지대학교 학사 정보 RAG 챗봇 (Solar Pro + HyDE)",
        is_default=True,
    )


@router.post("/sessions/create", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "id": sid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    return SessionCreateResponse(chat_session_id=sid)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    items = sorted(
        (
            {"id": s["id"], "created_at": s["created_at"], "n_messages": len(s["messages"])}
            for s in _sessions.values()
        ),
        key=lambda x: x["created_at"],
        reverse=True,
    )
    return SessionListResponse(sessions=items)


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Retrieval only — 답변 생성 없음."""
    pipeline = _get_pipeline()
    try:
        candidates, _decision = await pipeline._retrieve_then_rerank(  # type: ignore[attr-defined]
            req.query,
            hybrid_top_k=30,
            rerank_top_k=req.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"retrieval 실패: {exc}")
    return SearchResponse(query=req.query, results=[_to_citation(c) for c in candidates])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """단일 응답 채팅 — 답변 + 인용. 세션에 메시지 추가."""
    pipeline = _get_pipeline()
    sid = req.chat_session_id or str(uuid.uuid4())
    if sid not in _sessions:
        _sessions[sid] = {
            "id": sid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
        }

    t0 = time.time()
    try:
        result = await pipeline.run(req.message)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"pipeline.run 실패: {exc}")

    msg_id = str(uuid.uuid4())
    citations = [_to_citation(s) for s in (result.get("sources") or [])[:5]]
    resp = ChatResponse(
        chat_session_id=sid,
        message_id=msg_id,
        answer=result.get("answer", ""),
        citations=citations,
        grounded=bool(result.get("grounded", False)),
        verdict=str(result.get("verdict", "")),
        elapsed_ms=int((time.time() - t0) * 1000),
    )

    _sessions[sid]["messages"].append({
        "id": msg_id,
        "role": "user",
        "content": req.message,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    _sessions[sid]["messages"].append({
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": resp.answer,
        "citations": [c.model_dump() for c in citations],
        "grounded": resp.grounded,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    return resp


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 스트리밍 — Onyx 형식 호환.

    이벤트 흐름:
      event: message    data: {chat_session_id, message_id}    (시작)
      event: citations  data: [{...}, ...]                     (검색 결과)
      event: token      data: "<chunk>"                        (토큰 스트림)
      event: done       data: {grounded, verdict, elapsed_ms}  (종료)
    """
    pipeline = _get_pipeline()
    sid = req.chat_session_id or str(uuid.uuid4())
    if sid not in _sessions:
        _sessions[sid] = {
            "id": sid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
        }

    async def gen():
        t0 = time.time()
        msg_id = str(uuid.uuid4())
        yield f"event: message\ndata: {json.dumps({'chat_session_id': sid, 'message_id': msg_id})}\n\n"

        try:
            result = await pipeline.run(req.message)
        except Exception as exc:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return

        answer = result.get("answer", "")
        citations = [_to_citation(s).model_dump() for s in (result.get("sources") or [])[:5]]

        yield f"event: citations\ndata: {json.dumps(citations, ensure_ascii=False)}\n\n"

        # 토큰 단위 (간이 분할 — 진짜 streaming 은 LLM 레벨에서 future work).
        chunk_size = 30
        for i in range(0, len(answer), chunk_size):
            piece = answer[i : i + chunk_size]
            yield f"event: token\ndata: {json.dumps(piece, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)

        done = {
            "grounded": bool(result.get("grounded", False)),
            "verdict": str(result.get("verdict", "")),
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
        yield f"event: done\ndata: {json.dumps(done)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
