"""POST /api/indexing/* + SSE 진행률 (운영웹통합명세서 §8.5 + §11 Day 6).

Endpoints:
  POST /api/indexing/incremental  — status=Draft/Published 청크만 재인덱싱
  POST /api/indexing/full         — 전체 재인덱싱
  GET  /api/indexing/jobs         — 작업 큐 목록
  GET  /api/indexing/jobs/{id}    — 단일 조회
  GET  /api/indexing/jobs/{id}/stream  — SSE (1초마다 진행률 push)
  POST /api/indexing/jobs/{id}/cancel  — 취소 (status=cancelled)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import SessionLocal, get_db
from models import IndexingJob

router = APIRouter(prefix="/api/indexing", tags=["indexing"])


def _to_dict(j: IndexingJob) -> dict[str, Any]:
    return {
        "id": j.id,
        "job_type": j.job_type,
        "status": j.status,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        "chunks_total": j.chunks_total,
        "chunks_processed": j.chunks_processed,
        "error_message": j.error_message,
    }


def _enqueue(job_type: str, job_id: int) -> None:
    """Celery task enqueue (worker 가 import 가능할 때만)."""
    try:
        from worker import run_incremental, run_full  # type: ignore
    except ImportError:
        return
    if job_type == "incremental":
        run_incremental.delay(job_id)
    elif job_type == "full":
        run_full.delay(job_id)


@router.post("/incremental")
async def trigger_incremental(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    job = IndexingJob(job_type="incremental", status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    _enqueue("incremental", job.id)
    return _to_dict(job)


@router.post("/full")
async def trigger_full(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    job = IndexingJob(job_type="full", status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    _enqueue("full", job.id)
    return _to_dict(job)


@router.get("/jobs")
async def list_jobs(db: AsyncSession = Depends(get_db), limit: int = 30) -> dict[str, Any]:
    stmt = select(IndexingJob).order_by(IndexingJob.id.desc()).limit(limit)
    rows = list((await db.execute(stmt)).scalars().all())
    return {"items": [_to_dict(r) for r in rows]}


@router.get("/jobs/{job_id}")
async def get_job(job_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    j = await db.get(IndexingJob, job_id)
    if not j:
        raise HTTPException(404, f"job not found: {job_id}")
    return _to_dict(j)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    j = await db.get(IndexingJob, job_id)
    if not j:
        raise HTTPException(404, f"job not found: {job_id}")
    if j.status in ("success", "failed", "cancelled"):
        return _to_dict(j)
    j.status = "cancelled"
    j.completed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(j)
    return _to_dict(j)


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: int) -> StreamingResponse:
    """SSE — 1초마다 IndexingJob 행을 push."""

    async def gen():
        terminal = {"success", "failed", "cancelled"}
        while True:
            async with SessionLocal() as session:
                j = await session.get(IndexingJob, job_id)
                if not j:
                    yield f"event: error\ndata: job_not_found_{job_id}\n\n"
                    return
                payload = _to_dict(j)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if j.status in terminal:
                    return
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")
