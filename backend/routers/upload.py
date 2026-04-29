"""POST /api/upload/csv — CSV 일괄 업로드 (운영웹통합명세서 §11 Day 7).

Multipart `file` 업로드 → CSV 파싱 → Chunk UPSERT (status=Draft).

CSV 필수 컬럼:
  - doc_id, path, contents

선택 컬럼:
  - parent_doc_id, raw_content, status, source_collection
  - 그 외 컬럼은 모두 metadata 로 들어감 (title, question, answer, campus, ...).

UPSERT 정책:
  - 동일 doc_id 가 존재하면 metadata + contents 병합, version+1.
  - 동일 doc_id 가 없으면 신규 INSERT, ChunkHistory v1.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from models import Chunk, ChunkHistory
from schemas.metadata_v3 import MetadataV3

router = APIRouter(prefix="/api/upload", tags=["upload"])

REQUIRED_COLS = {"doc_id", "path", "contents"}
RESERVED_COLS = {
    "doc_id",
    "parent_doc_id",
    "path",
    "contents",
    "raw_content",
    "status",
    "source_collection",
}


@router.post("/csv")
async def upload_csv(
    file: UploadFile = File(...),
    collection: str = Query(..., description="source_collection 기본값 (CSV 에 없으면)"),
    skip_validation: bool = Query(False, description="MetadataV3 검증 스킵"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "CSV 파일이 아닙니다")

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp949")
        except UnicodeDecodeError as exc:
            raise HTTPException(400, f"인코딩 실패 (utf-8/cp949): {exc}")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(400, "CSV 헤더가 비어있습니다")
    missing = REQUIRED_COLS - set(reader.fieldnames)
    if missing:
        raise HTTPException(400, f"필수 컬럼 누락: {sorted(missing)}")

    created = 0
    updated = 0
    errors: list[dict[str, Any]] = []

    for row_idx, row in enumerate(reader, start=2):  # 1=header
        doc_id = (row.get("doc_id") or "").strip()
        if not doc_id:
            errors.append({"row": row_idx, "error": "doc_id 비어있음"})
            continue

        meta: dict[str, Any] = {}
        for k, v in row.items():
            if k in RESERVED_COLS:
                continue
            if v is None or str(v).strip() == "":
                continue
            meta[k] = str(v).strip()

        meta.setdefault("source_collection", row.get("source_collection") or collection)
        meta.setdefault("status", row.get("status") or "Draft")

        existing = await db.get(Chunk, doc_id)
        if existing:
            before = dict(existing.chunk_metadata or {})
            after = {**before, **meta}
            after["version"] = int(before.get("version", 1)) + 1
            if not skip_validation:
                try:
                    MetadataV3(**after)
                except ValidationError as exc:
                    errors.append({"row": row_idx, "doc_id": doc_id, "error": exc.errors()})
                    continue
            existing.chunk_metadata = after
            if row.get("contents"):
                existing.contents = row["contents"]
            if row.get("raw_content"):
                existing.raw_content = row["raw_content"]
            db.add(
                ChunkHistory(
                    doc_id=doc_id,
                    version=after["version"],
                    diff={"_csv_upsert": [None, "v" + str(after["version"])]},
                )
            )
            updated += 1
        else:
            meta["version"] = 1
            if not skip_validation:
                try:
                    MetadataV3(**meta)
                except ValidationError as exc:
                    errors.append({"row": row_idx, "doc_id": doc_id, "error": exc.errors()})
                    continue
            chunk = Chunk(
                doc_id=doc_id,
                parent_doc_id=row.get("parent_doc_id") or None,
                path=row["path"],
                schema_version="v3",
                source_collection=row.get("source_collection") or collection,
                chunk_metadata=meta,
                contents=row["contents"],
                raw_content=row.get("raw_content") or row["contents"],
                status=row.get("status") or "Draft",
            )
            db.add(chunk)
            db.add(ChunkHistory(doc_id=doc_id, version=1, diff={"_created": [None, "v1"]}))
            created += 1

    await db.commit()
    return {
        "ok": True,
        "filename": file.filename,
        "collection": collection,
        "created": created,
        "updated": updated,
        "errors": errors[:50],
        "error_count": len(errors),
    }
