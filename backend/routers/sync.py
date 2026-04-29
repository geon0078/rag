"""Docmost ↔ RAG 동기화 webhook (onyx+docmost 개발.md §5 Day 3).

Docmost 페이지가 편집되면 Docmost 가 본 엔드포인트로 webhook 을 보내고,
우리 RAG 가 페이지 마크다운을 다시 청킹 → 메타 v3 변환 → chunks 테이블
upsert → Celery 인덱싱 큐에 등록한다.

Endpoints:
  POST /api/sync/docmost              — webhook 수신 (page.created/updated/deleted)
  POST /api/sync/docmost/reindex/{page_id}  — 수동 재인덱싱 (Docmost API fetch)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from models import Chunk, IndexingJob

router = APIRouter(prefix="/api/sync", tags=["sync"])


class DocmostPage(BaseModel):
    id: str
    title: str
    content_md: str = ""
    space_id: str | None = None
    slug: str | None = None
    breadcrumb: list[str] = []


class DocmostWebhook(BaseModel):
    event: str  # "page.created" | "page.updated" | "page.deleted"
    page: DocmostPage
    timestamp: str | None = None


_HEADER_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _split_markdown_sections(md: str) -> list[tuple[str, str]]:
    """## 헤더 단위 분할. (heading, body) 튜플 리스트.

    헤더 없으면 [("(본문)", 전체)] 1개 반환.
    """
    if not md or not md.strip():
        return []
    headers = list(_HEADER_RE.finditer(md))
    if not headers:
        return [("(본문)", md.strip())]
    out: list[tuple[str, str]] = []
    if headers[0].start() > 0:
        intro = md[: headers[0].start()].strip()
        if intro:
            out.append(("(intro)", intro))
    for i, m in enumerate(headers):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(md)
        body = md[body_start:body_end].strip()
        if body:
            out.append((heading, body))
    return out


def _build_chunk_row(
    page: DocmostPage,
    section_idx: int,
    section_count: int,
    heading: str,
    body: str,
) -> dict[str, Any]:
    """메타 v3 평탄 dict 구성 (METADATA_AND_WEB_SPEC §3 호환)."""
    doc_id = f"docmost_{page.id}_s{section_idx}"
    parent_doc_id = f"docmost_{page.id}"
    breadcrumb = list(page.breadcrumb) + (
        [heading] if heading not in ("(본문)", "(intro)") else []
    )
    path = "/".join(breadcrumb) if breadcrumb else page.title
    metadata = {
        "doc_id": doc_id,
        "parent_doc_id": parent_doc_id,
        "path": path,
        "breadcrumb": breadcrumb,
        "schema_version": "v3",
        "source_collection": "Docmost",
        "category": page.space_id or "Docmost",
        "title": f"{page.title} — {heading}" if heading else page.title,
        "campus": "전체",
        "language": "ko",
        "chunk_index": section_idx,
        "chunk_count": section_count,
        "depth": len(breadcrumb),
        "topic_id": parent_doc_id,
        "topic_name": page.title,
        "topic_section": heading,
        "topic_section_order": section_idx,
        "docmost_page_id": page.id,
        "docmost_space_id": page.space_id,
        "docmost_slug": page.slug,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "version": 1,
        "status": "Indexed",
    }
    return {
        "doc_id": doc_id,
        "parent_doc_id": parent_doc_id,
        "path": path,
        "schema_version": "v3",
        "source_collection": "Docmost",
        "metadata": metadata,
        "contents": body,
        "raw_content": body,
        "status": "Indexed",
    }


async def _archive_page_chunks(page_id: str, db: AsyncSession) -> int:
    """page.deleted 시 해당 page 의 모든 청크를 status='Archived' 표시."""
    stmt = (
        update(Chunk)
        .where(Chunk.chunk_metadata["docmost_page_id"].astext == page_id)
        .values(status="Archived")
    )
    res = await db.execute(stmt)
    await db.commit()
    return res.rowcount or 0


@router.post("/docmost")
async def docmost_webhook(
    payload: DocmostWebhook = Body(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Docmost 페이지 변경 webhook.

    event=page.deleted: 모든 청크 status='Archived'.
    event=page.created/updated: 마크다운 ## 단위 청킹 → upsert + Celery 큐 등록.
    """
    page = payload.page
    if payload.event == "page.deleted":
        archived = await _archive_page_chunks(page.id, db)
        return {"ok": True, "event": payload.event, "archived": archived}

    sections = _split_markdown_sections(page.content_md)
    if not sections:
        return {
            "ok": True,
            "event": payload.event,
            "page_id": page.id,
            "chunks": 0,
            "note": "empty",
        }

    rows = [
        _build_chunk_row(page, idx, len(sections), heading, body)
        for idx, (heading, body) in enumerate(sections)
    ]

    stmt = pg_insert(Chunk.__table__).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["doc_id"],
        set_={
            "path": stmt.excluded.path,
            "metadata": stmt.excluded.metadata,
            "contents": stmt.excluded.contents,
            "raw_content": stmt.excluded.raw_content,
            "status": stmt.excluded.status,
        },
    )
    await db.execute(stmt)

    job = IndexingJob(job_type="incremental", status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        from worker import run_incremental  # type: ignore
        run_incremental.delay(job.id)
    except ImportError:
        pass

    return {
        "ok": True,
        "event": payload.event,
        "page_id": page.id,
        "chunks_upserted": len(rows),
        "indexing_job_id": job.id,
    }


@router.post("/docmost/reindex/{page_id}")
async def manual_reindex(
    page_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """수동 재인덱싱 — Docmost API 에서 페이지를 가져와 webhook 처럼 처리.

    DOCMOST_API_URL + DOCMOST_API_KEY 환경변수 필요.
    """
    base = os.environ.get("DOCMOST_API_URL", "http://docmost:3000")
    key = os.environ.get("DOCMOST_API_KEY", "")
    if not key:
        raise HTTPException(503, detail="DOCMOST_API_KEY 미설정")

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"{base}/api/v1/pages/{page_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        if r.status_code != 200:
            raise HTTPException(r.status_code, detail=f"docmost fetch 실패: {r.text}")
        data = r.json()

    page = DocmostPage(
        id=page_id,
        title=data.get("title", ""),
        content_md=data.get("content_md") or data.get("content", ""),
        space_id=data.get("space_id"),
        slug=data.get("slug"),
        breadcrumb=data.get("breadcrumb", []),
    )
    return await docmost_webhook(
        DocmostWebhook(event="page.updated", page=page),
        db,
    )
