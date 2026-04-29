"""Corpus → PostgreSQL ingest (운영웹통합명세서 §11 Day 2 + Track C).

- ``data/corpus.parquet``  → ``chunks`` (v3 평탄 dict, breadcrumb·path·depth 자동 생성)
- ``data/qa_adversarial.parquet`` 또는 ``data/golden_curated_v1.parquet``
  도 같이 ingest 가능 (선택: --include-golden)
- 모든 행은 backend.schemas.metadata_v3.MetadataV3 검증 통과 필요

Run inside backend container (DB_URL 환경변수 자동 로드):
    python ingest_corpus.py [--corpus /data/corpus.parquet] [--include-golden]

Run from project root:
    python -m backend.ingest_corpus
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # backend on path

from db import SessionLocal, engine  # noqa: E402
from models import Base, Chunk  # noqa: E402
from schemas.metadata_v3 import MetadataV3  # noqa: E402


def _coerce_metadata_v3(row: pd.Series) -> dict[str, Any]:
    """기존 v2 metadata + 청크 본문에서 v3 평탄 dict 구성."""
    meta = row.get("metadata") or {}
    if hasattr(meta, "tolist"):
        meta = meta.tolist()
    if not isinstance(meta, dict):
        meta = {}

    sc = meta.get("source_collection") or meta.get("category") or "기타"
    title = meta.get("title") or row.get("doc_id", "(제목 없음)")
    campus = meta.get("campus") or "전체"
    category = meta.get("category") or sc
    subcategory = meta.get("subcategory")

    breadcrumb = [sc]
    if subcategory:
        breadcrumb.append(str(subcategory))
    breadcrumb.append(str(title))
    path = "/".join(breadcrumb)

    payload: dict[str, Any] = {
        "doc_id": row["doc_id"],
        "parent_doc_id": meta.get("parent_doc_id"),
        "path": path,
        "breadcrumb": breadcrumb,
        "schema_version": "v3",
        "source_collection": sc,
        "category": category,
        "subcategory": subcategory,
        "title": str(title),
        "campus": campus if campus in ("성남", "의정부", "대전", "전체") else "전체",
        "language": "ko",
        "chunk_index": int(meta.get("chunk_index", 0) or 0),
        "chunk_count": int(meta.get("chunk_count", 1) or 1),
        "depth": len(breadcrumb),
    }

    for key in ("chapter", "chapter_title", "article_number", "article_title",
                "paragraph", "start_date", "end_date", "semester", "event_type",
                "lecture_id", "lecture_title", "section", "subject_area",
                "is_required", "phone", "building", "floor", "facility_type",
                "department", "low_confidence"):
        if key in meta and meta[key] is not None:
            payload[key] = meta[key]

    payload["created_at"] = meta.get("created_at") or datetime.now(timezone.utc).isoformat()
    payload["indexed_at"] = meta.get("indexed_at")
    payload["confidence"] = meta.get("confidence") or "medium"

    if payload["source_collection"] == "학칙_조항" and not payload.get("article_number"):
        m = re.search(r"제\s*(\d+)\s*조", str(title))
        payload["article_number"] = f"제{m.group(1)}조" if m else f"unknown_{row['doc_id']}"
    if payload["source_collection"] == "강의평가" and not payload.get("lecture_id"):
        payload["lecture_id"] = str(row["doc_id"]).split("_c")[0]
    if payload["source_collection"] == "시설_연락처" and not payload.get("phone"):
        payload["phone"] = "(미상)"
    if payload["source_collection"] == "학사일정" and not payload.get("start_date"):
        payload["start_date"] = "1970-01-01"

    return payload


async def _create_tables_if_needed() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def ingest_parquet(path: Path, status: str = "Indexed") -> int:
    df = pd.read_parquet(path)
    print(f"[ingest] read {len(df)} rows from {path}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            v3 = _coerce_metadata_v3(row)
            MetadataV3(**v3)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if skipped <= 5:
                print(f"[ingest] skip {row.get('doc_id')!r}: {exc}")
            continue

        rows.append({
            "doc_id": row["doc_id"],
            "parent_doc_id": v3.get("parent_doc_id"),
            "path": v3["path"],
            "schema_version": "v3",
            "source_collection": v3["source_collection"],
            "metadata": json.loads(json.dumps(v3, default=str)),
            "contents": str(row["contents"]),
            "raw_content": str(row.get("raw_content") or row["contents"]),
            "status": status,
        })

    if not rows:
        print("[ingest] no valid rows; abort")
        return 0

    print(f"[ingest] inserting {len(rows)} rows (skipped={skipped}) ...")
    # Use Chunk.__table__ (Core) instead of the ORM class; the column key
    # ``metadata`` collides with SQLAlchemy's ``MetaData`` attribute when
    # going through the ORM-level pg_insert (AttributeError on bulk insert).
    async with SessionLocal() as session:
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
        await session.execute(stmt)
        await session.commit()
    print(f"[ingest] done: {len(rows)} upserted")
    return len(rows)


async def main_async(args: argparse.Namespace) -> int:
    await _create_tables_if_needed()

    total = 0
    corpus = Path(args.corpus)
    if corpus.exists():
        total += await ingest_parquet(corpus, status="Indexed")
    else:
        print(f"[ingest] corpus missing: {corpus}")
        return 1

    if args.include_golden:
        for p in [
            PROJECT_ROOT / "data" / "golden_curated_v1.parquet",
            PROJECT_ROOT / "data" / "golden_candidates_v1.parquet",
        ]:
            if p.exists():
                total += await ingest_parquet(p, status="Draft")

    async with SessionLocal() as session:
        all_chunks = (await session.execute(select(Chunk))).scalars().all()
    print(f"[ingest] DB now has {len(all_chunks)} chunks (total upserted this run: {total})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(PROJECT_ROOT / "data" / "corpus.parquet"))
    p.add_argument("--include-golden", action="store_true")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
