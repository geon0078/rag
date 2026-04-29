"""GET /api/tree (운영웹통합명세서 §6.3, §8.2).

평탄 청크 리스트를 path/breadcrumb 로 트리 구축한 결과 반환. 좌측 Sidebar
의 react-arborist 가 직접 렌더링할 수 있는 형태.

Lazy-loading 미사용 (Day 3 단순 구현). 컬렉션 단위 분리로 트리 크기 통제:
  - ?collection=학칙_조항 → 해당 컬렉션만
  - ?collection 미지정 → 컬렉션 단위 최상위 레벨만 (자식은 expand 시 별도 호출)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from models import Chunk

router = APIRouter(prefix="/api/tree", tags=["tree"])


def _build_tree(rows: list[Chunk]) -> list[dict[str, Any]]:
    """평탄 청크 리스트를 path 기반 트리로 변환 (react-arborist 호환)."""
    root: dict[str, dict[str, Any]] = {}

    def _ensure(cursor: dict[str, dict[str, Any]], key: str, full_id: str, depth: int) -> dict[str, Any]:
        return cursor.setdefault(key, {
            "id": full_id,
            "name": key,
            "children": {},
            "doc_id": None,
            "depth": depth,
            "status": None,
        })

    for c in rows:
        meta = c.chunk_metadata or {}
        breadcrumb: list[str] = list(meta.get("breadcrumb") or [])
        if not breadcrumb:
            breadcrumb = c.path.split("/") if c.path else [c.source_collection, c.doc_id]

        cursor = root
        full_path = ""
        last_node: dict[str, Any] | None = None
        for depth, part in enumerate(breadcrumb):
            full_path = f"{full_path}/{part}" if full_path else part
            node = _ensure(cursor, part, full_path, depth)
            last_node = node
            cursor = node["children"]
        if last_node is not None:
            last_node["doc_id"] = c.doc_id
            last_node["status"] = c.status

    def _flatten(d: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _, node in sorted(d.items(), key=lambda kv: kv[0]):
            children = _flatten(node["children"])
            out.append({
                "id": node["id"],
                "name": node["name"],
                "doc_id": node.get("doc_id"),
                "depth": node.get("depth"),
                "status": node.get("status"),
                "children": children,
            })
        return out

    return _flatten(root)


@router.get("")
async def get_tree(
    collection: str | None = Query(None),
    limit: int = Query(2500, ge=1, le=10_000),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if collection:
        stmt = (
            select(Chunk)
            .where(Chunk.source_collection == collection)
            .order_by(Chunk.path)
            .limit(limit)
        )
        rows = list((await db.execute(stmt)).scalars().all())
        return {"collection": collection, "tree": _build_tree(rows), "count": len(rows)}

    stmt = (
        select(Chunk.source_collection, func.count(Chunk.doc_id))
        .group_by(Chunk.source_collection)
    )
    counts = (await db.execute(stmt)).all()
    tree = [
        {
            "id": sc,
            "name": sc,
            "doc_id": None,
            "depth": 0,
            "status": None,
            "children": [],
            "count": int(n),
        }
        for sc, n in sorted(counts)
    ]
    return {"collection": None, "tree": tree, "count": sum(int(n) for _, n in counts)}
