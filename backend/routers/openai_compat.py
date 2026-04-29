"""OpenAI-compatible /v1/chat/completions endpoint.

Onyx 가 이 endpoint 를 LLM provider 로 호출 (env GEN_AI_API_BASE=http://api:8000/v1).
요청 형식은 OpenAI 표준, 응답은 OpenAI 표준 + 비표준 `citations` 필드.

Endpoints:
  POST /v1/chat/completions   — non-streaming + streaming (SSE)
  GET  /v1/models             — 모델 목록 (Onyx 호환용)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["openai-compat"])


_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from src.pipeline.rag_pipeline import RagPipeline  # type: ignore
        except ImportError as exc:
            raise HTTPException(503, detail=f"RagPipeline import 실패: {exc}")
        _pipeline = RagPipeline()
    return _pipeline


# ────────────────────────────────────────────────────────────
# Request/Response models — OpenAI Chat Completions 표준
# ────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "solar-pro"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = 1
    stop: str | list[str] | None = None
    user: str | None = None
    # Onyx 가 보낼 수 있는 비표준 필드 (무시)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CitationOut(BaseModel):
    id: int
    doc_id: str
    title: str | None = None
    score: float | None = None
    outline_url: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    # 비표준 확장 — Onyx 가 읽을 수 있음 (없으면 무시)
    citations: list[CitationOut] = []
    grounded: bool = True
    verdict: str = "grounded"


# ────────────────────────────────────────────────────────────
# Models endpoint (Onyx LLM-provider 헬스체크용)
# ────────────────────────────────────────────────────────────


@router.get("/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "solar-pro",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "eulji-rag",
            },
            {
                "id": "solar-mini",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "eulji-rag",
            },
        ],
    }


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────


_REMINDER_MARKERS = (
    "<system-reminder>",
    "search_eulji_corpus",
    "internal_search",
    "검색 없이 일반 지식만으로",
)


def _is_reminder(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(m.lower() in t for m in _REMINDER_MARKERS) or t.startswith("[chat_history")


def _extract_query(messages: list[ChatMessage]) -> str:
    """user 메시지 중 system-reminder/persona 지시문이 아닌 가장 마지막 메시지를 query 로 채택.

    Onyx 는 persona system_prompt + task_prompt 를 메시지 끝에 추가하기도 하므로
    그 패턴을 필터링.
    """
    for m in reversed(messages):
        if m.role == "user" and m.content and not _is_reminder(m.content):
            return m.content
    # fallback — 모두 reminder 였으면 가장 처음 user 메시지
    for m in messages:
        if m.role == "user" and m.content:
            return m.content
    return ""


def _to_citation_list(sources: list[dict[str, Any]]) -> list[CitationOut]:
    out: list[CitationOut] = []
    for i, s in enumerate(sources[:5], start=1):
        payload = s.get("payload") or {}
        doc_id = str(s.get("doc_id") or payload.get("doc_id") or "")
        out.append(CitationOut(
            id=i,
            doc_id=doc_id,
            title=payload.get("title") or payload.get("topic_name"),
            score=s.get("score") or s.get("rerank_score"),
            outline_url=f"http://localhost:3002/doc/{doc_id}" if doc_id else None,
        ))
    return out


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 3)


# ────────────────────────────────────────────────────────────
# Citation → Outline 클릭 가능 markdown 링크 변환
# ────────────────────────────────────────────────────────────


_URL_MAP: dict[str, dict[str, str]] | None = None


def _load_url_map() -> dict[str, dict[str, str]]:
    """tmp/outline_url_map.json 캐시 로드."""
    global _URL_MAP
    if _URL_MAP is not None:
        return _URL_MAP
    candidates = [
        Path("/workspace/data/outline_url_map.json"),
        Path("/workspace/tmp/outline_url_map.json"),
        Path(__file__).resolve().parent.parent.parent / "tmp" / "outline_url_map.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                _URL_MAP = json.loads(p.read_text(encoding="utf-8"))
                return _URL_MAP
            except Exception:
                continue
    _URL_MAP = {"by_corpus_doc_id": {}, "by_title": {}, "by_topic_id": {}}
    return _URL_MAP


# [출처: <breadcrumb>] (단일 필드, 새 형식)
_CITATION_RE_BREADCRUMB = re.compile(r"\[출처\s*[::]\s*([^\]]+)\]")
# [출처: doc_id, 카테고리, 캠퍼스]  3-tuple (구 형식)
_CITATION_RE_TUPLE = re.compile(r"\[출처\s*[::]\s*([^,\]]+)\s*,\s*([^,\]]+)\s*,\s*([^\]]+)\]")


def _resolve_url_by_breadcrumb(breadcrumb: str, sources: list[dict[str, Any]]) -> str | None:
    """sources 의 payload.breadcrumb 와 일치하는 chunk 의 outline_url 반환."""
    target = breadcrumb.strip()
    for s in sources:
        payload = s.get("payload") or {}
        if (payload.get("breadcrumb") or "").strip() == target:
            url = payload.get("outline_url")
            if url:
                return url
    last = target.split(">")[-1].strip() if ">" in target else target
    url_map = _load_url_map()
    by_title = url_map.get("by_title") or {}
    if last in by_title:
        return by_title[last]
    return None


def _resolve_url(doc_id: str, sources: list[dict[str, Any]]) -> str | None:
    """corpus doc_id 또는 sources payload 로부터 outline URL 찾기 (구 형식 fallback)."""
    url_map = _load_url_map()
    by_doc = url_map.get("by_corpus_doc_id") or {}
    if doc_id in by_doc:
        return by_doc[doc_id]
    by_title = url_map.get("by_title") or {}
    for s in sources:
        payload = s.get("payload") or {}
        if str(s.get("doc_id", "")) == doc_id or str(payload.get("doc_id", "")) == doc_id:
            url = payload.get("outline_url")
            if url:
                return url
            title = payload.get("title") or payload.get("topic_name")
            if title and title in by_title:
                return by_title[title]
    return None


def _replace_citations_with_links(answer: str, sources: list[dict[str, Any]]) -> str:
    """답변 내 [출처: ...] → [출처: ...](outline_url) markdown 링크 변환."""
    if not answer:
        return answer

    # 1) 구 형식 (3-tuple) 먼저 처리 — 더 specific
    def _sub_tuple(m: re.Match[str]) -> str:
        doc_id = m.group(1).strip()
        url = _resolve_url(doc_id, sources)
        return f"{m.group(0)}({url})" if url else m.group(0)

    out = _CITATION_RE_TUPLE.sub(_sub_tuple, answer)

    # 2) breadcrumb (새 형식) — 단일 필드 처리
    def _sub_bc(m: re.Match[str]) -> str:
        body = m.group(1).strip()
        # 콤마 있으면 3-tuple 미매칭 케이스 — skip
        if "," in body:
            return m.group(0)
        url = _resolve_url_by_breadcrumb(body, sources)
        return f"{m.group(0)}({url})" if url else m.group(0)

    out = _CITATION_RE_BREADCRUMB.sub(_sub_bc, out)
    return out


# ────────────────────────────────────────────────────────────
# /v1/chat/completions
# ────────────────────────────────────────────────────────────


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    pipeline = _get_pipeline()
    query = _extract_query(req.messages)
    if not query.strip():
        raise HTTPException(400, detail="No user message provided")

    if req.stream:
        return StreamingResponse(
            _stream_response(pipeline, req, query),
            media_type="text/event-stream",
        )

    try:
        result = await pipeline.run(query)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"pipeline.run 실패: {exc}")

    answer = result.get("answer", "")
    sources = result.get("sources") or []
    answer = _replace_citations_with_links(answer, sources)
    cit = _to_citation_list(sources)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(content=answer),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=_approx_tokens(query),
            completion_tokens=_approx_tokens(answer),
            total_tokens=_approx_tokens(query) + _approx_tokens(answer),
        ),
        citations=cit,
        grounded=bool(result.get("grounded", True)),
        verdict=str(result.get("verdict", "grounded")),
    )


async def _stream_response(pipeline, req: ChatCompletionRequest, query: str):
    """SSE 스트리밍 — pipeline 은 non-streaming 이므로 결과를 chunked 로 분할 송신."""
    try:
        result = await pipeline.run(query)
    except Exception as exc:  # noqa: BLE001
        err_chunk = {"error": {"message": str(exc), "type": "internal_error"}}
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    answer = result.get("answer", "")
    sources = result.get("sources") or []
    answer = _replace_citations_with_links(answer, sources)
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    chunk_size = 20
    for i in range(0, len(answer), chunk_size):
        piece = answer[i : i + chunk_size]
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.02)

    last = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(last)}\n\n"
    yield "data: [DONE]\n\n"
