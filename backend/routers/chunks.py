"""GET/PATCH /api/chunks + validate + related (운영웹통합명세서 §8.1, §8.2, §8.6).

Day 3 (read-only):
  GET   /api/chunks                — 목록, 필터(collection/campus/status/q)
  GET   /api/chunks/{doc_id}       — 단일 조회
  GET   /api/chunks/{doc_id}/related — 같은 부모(parent_doc_id) + 인접 청크
Day 4 (write):
  PATCH /api/chunks/{doc_id}       — metadata/contents 부분 갱신 + 변경 이력
  POST  /api/chunks/validate       — 저장하지 않고 metadata v3 검증만
  GET   /api/chunks/{doc_id}/history — 변경 이력 목록
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, ValidationError
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from models import Chunk, ChunkHistory
from schemas.metadata_v3 import MetadataV3

router = APIRouter(prefix="/api/chunks", tags=["chunks"])


class ChunkPatch(BaseModel):
    """부분 갱신 입력. None 인 필드는 변경 안 함."""

    metadata: dict[str, Any] | None = None
    contents: str | None = None
    raw_content: str | None = None
    status: str | None = None
    expected_version: int | None = None  # 낙관적 락 (Optional)


class ValidateInput(BaseModel):
    metadata: dict[str, Any]


class ChunkCreate(BaseModel):
    """신규 청크 작성 입력 (운영웹통합명세서 §8.1)."""

    doc_id: str
    parent_doc_id: str | None = None
    path: str
    source_collection: str
    metadata: dict[str, Any]
    contents: str
    raw_content: str | None = None
    status: str = "Draft"


def _diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[Any]]:
    keys = set(before) | set(after)
    return {
        k: [before.get(k), after.get(k)]
        for k in keys
        if before.get(k) != after.get(k)
    }


def _to_dict(c: Chunk) -> dict[str, Any]:
    return {
        "doc_id": c.doc_id,
        "parent_doc_id": c.parent_doc_id,
        "path": c.path,
        "schema_version": c.schema_version,
        "source_collection": c.source_collection,
        "metadata": c.chunk_metadata,
        "contents": c.contents,
        "raw_content": c.raw_content,
        "status": c.status,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


@router.get("")
async def list_chunks(
    collection: str | None = Query(None, alias="collection"),
    campus: str | None = Query(None),
    status: str | None = Query(None),
    q: str | None = Query(None, description="제목 또는 내용 부분 매칭"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    stmt = select(Chunk)
    if collection:
        stmt = stmt.where(Chunk.source_collection == collection)
    if status:
        stmt = stmt.where(Chunk.status == status)
    if campus:
        stmt = stmt.where(Chunk.chunk_metadata["campus"].astext == campus)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(
            Chunk.contents.ilike(like),
            Chunk.path.ilike(like),
            Chunk.chunk_metadata["title"].astext.ilike(like),
        ))
    stmt = stmt.order_by(Chunk.path).limit(limit).offset(offset)

    rows = (await db.execute(stmt)).scalars().all()
    return {
        "items": [_to_dict(c) for c in rows],
        "limit": limit,
        "offset": offset,
        "count": len(rows),
    }


@router.get("/{doc_id}")
async def get_chunk(doc_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    chunk = await db.get(Chunk, doc_id)
    if not chunk:
        raise HTTPException(404, f"chunk not found: {doc_id}")
    return _to_dict(chunk)


@router.get("/{doc_id}/related")
async def get_related(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    chunk = await db.get(Chunk, doc_id)
    if not chunk:
        raise HTTPException(404, f"chunk not found: {doc_id}")

    siblings: list[Chunk] = []
    if chunk.parent_doc_id:
        stmt = (
            select(Chunk)
            .where(Chunk.parent_doc_id == chunk.parent_doc_id)
            .where(Chunk.doc_id != doc_id)
            .order_by(Chunk.path)
            .limit(20)
        )
        siblings = list((await db.execute(stmt)).scalars().all())

    children_stmt = (
        select(Chunk)
        .where(Chunk.parent_doc_id == doc_id)
        .order_by(Chunk.path)
        .limit(20)
    )
    children = list((await db.execute(children_stmt)).scalars().all())

    return {
        "self": _to_dict(chunk),
        "siblings": [_to_dict(s) for s in siblings],
        "children": [_to_dict(c) for c in children],
    }


@router.post("/validate")
async def validate_metadata(payload: ValidateInput) -> dict[str, Any]:
    """저장 X — metadata v3 검증만."""
    try:
        MetadataV3(**payload.metadata)
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors())
    return {"ok": True}


@router.post("")
async def create_chunk(
    payload: ChunkCreate = Body(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """신규 청크 생성 (FAQ 마법사 등). status 기본=Draft, version=1.

    운영웹통합명세서 §8.1: ``POST /api/chunks  Body: { metadata, contents, raw_content }``
    """
    if await db.get(Chunk, payload.doc_id):
        raise HTTPException(409, f"doc_id already exists: {payload.doc_id}")

    meta = dict(payload.metadata)
    meta["version"] = 1
    try:
        MetadataV3(**meta)
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors())

    chunk = Chunk(
        doc_id=payload.doc_id,
        parent_doc_id=payload.parent_doc_id,
        path=payload.path,
        schema_version="v3",
        source_collection=payload.source_collection,
        chunk_metadata=meta,
        contents=payload.contents,
        raw_content=payload.raw_content or payload.contents,
        status=payload.status,
    )
    db.add(chunk)
    db.add(ChunkHistory(doc_id=payload.doc_id, version=1, diff={"_created": [None, "v1"]}))
    await db.commit()
    await db.refresh(chunk)
    return _to_dict(chunk)


@router.patch("/{doc_id}")
async def patch_chunk(
    doc_id: str,
    payload: ChunkPatch = Body(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    chunk = await db.get(Chunk, doc_id)
    if not chunk:
        raise HTTPException(404, f"chunk not found: {doc_id}")

    before_meta = dict(chunk.chunk_metadata or {})

    # 낙관적 락 (선택): expected_version 이 다르면 409
    if payload.expected_version is not None:
        cur_version = int(before_meta.get("version", 1))
        if cur_version != payload.expected_version:
            raise HTTPException(
                409,
                detail=f"version mismatch: server={cur_version} vs expected={payload.expected_version}",
            )

    # metadata merge + v3 검증
    after_meta = before_meta.copy()
    if payload.metadata is not None:
        after_meta.update({k: v for k, v in payload.metadata.items() if k != "version"})
    after_meta["version"] = int(before_meta.get("version", 1)) + 1
    try:
        MetadataV3(**after_meta)
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors())

    # 컬럼 변경 적용
    chunk.chunk_metadata = after_meta
    if payload.contents is not None:
        chunk.contents = payload.contents
    if payload.raw_content is not None:
        chunk.raw_content = payload.raw_content
    if payload.status is not None:
        chunk.status = payload.status

    # 변경 이력 기록
    diff = _diff(before_meta, after_meta)
    if payload.contents is not None and payload.contents != chunk.contents:
        diff["contents"] = ["<old>", "<new>"]
    history = ChunkHistory(
        doc_id=doc_id,
        version=after_meta["version"],
        diff=diff,
    )
    db.add(history)
    await db.commit()
    await db.refresh(chunk)

    return _to_dict(chunk)


@router.get("/{doc_id}/history")
async def get_history(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    stmt = (
        select(ChunkHistory)
        .where(ChunkHistory.doc_id == doc_id)
        .order_by(ChunkHistory.version.desc())
        .limit(limit)
    )
    rows = list((await db.execute(stmt)).scalars().all())
    return {
        "doc_id": doc_id,
        "items": [
            {
                "id": r.id,
                "version": r.version,
                "changed_at": r.changed_at.isoformat() if r.changed_at else None,
                "diff": r.diff,
            }
            for r in rows
        ],
    }
