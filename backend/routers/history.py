"""GET /api/history/* — 전역 변경 이력 (운영웹통합명세서 §11 Day 7).

전역 활동 피드 — chunk 별 이력은 ``/api/chunks/{doc_id}/history`` (chunks.py).

Endpoints:
  GET /api/history/recent  — 최근 변경 이력 (across all chunks)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from models import Chunk, ChunkHistory

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/recent")
async def list_recent(
    limit: int = Query(50, ge=1, le=500),
    collection: str | None = Query(None, description="source_collection 필터 (옵션)"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """최근 변경 이력 — chunks JOIN 으로 source_collection 도 같이 반환."""
    title_expr = Chunk.chunk_metadata["title"].astext.label("title")
    stmt = (
        select(ChunkHistory, Chunk.source_collection, Chunk.path, title_expr)
        .join(Chunk, Chunk.doc_id == ChunkHistory.doc_id)
        .order_by(ChunkHistory.changed_at.desc())
        .limit(limit)
    )
    if collection:
        stmt = stmt.where(Chunk.source_collection == collection)

    rows = (await db.execute(stmt)).all()
    return {
        "items": [
            {
                "id": h.id,
                "doc_id": h.doc_id,
                "version": h.version,
                "changed_at": h.changed_at.isoformat() if h.changed_at else None,
                "diff": h.diff,
                "source_collection": coll,
                "path": path,
                "title": title,
            }
            for h, coll, path, title in rows
        ]
    }
