"""Celery worker (운영웹통합명세서 §4.2 + §11 Day 6).

Redis 를 broker + result backend 로 사용. 인덱싱 작업을 비동기로 실행하면서
PostgreSQL `indexing_jobs` 테이블에 진행률을 기록 (SSE 가 read).

Tasks:
  - run_incremental(job_id) : status=Draft|Published 인 청크만 Qdrant 재인덱싱
  - run_full(job_id)        : 모든 청크 재인덱싱

Notes:
  - 본 파일은 placeholder 단계 — 실 인덱싱 호출은 src/ 의 인덱서를 재사용해야
    하며, 그 부분은 후속 단계에서 wired in. 현재는 chunks_total/processed
    카운터만 갱신하여 SSE/UI 흐름 검증.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from celery import Celery
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from models import Chunk, IndexingJob


REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
SYNC_DB_URL = os.environ.get(
    "DB_URL", "postgresql+psycopg://rag:rag@postgres:5432/euljiu_rag"
)

celery_app = Celery("euljiu_worker", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=True,
)


def _sync_engine():
    return create_engine(SYNC_DB_URL, pool_pre_ping=True)


def _set_status(session: Session, job_id: int, **fields) -> None:
    session.execute(update(IndexingJob).where(IndexingJob.id == job_id).values(**fields))
    session.commit()


@celery_app.task(name="indexing.run_incremental")
def run_incremental(job_id: int) -> dict:
    eng = _sync_engine()
    with Session(eng) as session:
        _set_status(
            session,
            job_id,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        rows = (
            session.execute(
                select(Chunk).where(Chunk.status.in_(["Draft", "Published"]))
            )
            .scalars()
            .all()
        )
        total = len(rows)
        _set_status(session, job_id, chunks_total=total)
        for i, _c in enumerate(rows, start=1):
            time.sleep(0.05)
            if i % 5 == 0 or i == total:
                _set_status(session, job_id, chunks_processed=i)
        _set_status(
            session,
            job_id,
            status="success",
            completed_at=datetime.now(timezone.utc),
            chunks_processed=total,
        )
    return {"job_id": job_id, "total": total}


@celery_app.task(name="indexing.run_full")
def run_full(job_id: int) -> dict:
    eng = _sync_engine()
    with Session(eng) as session:
        _set_status(
            session,
            job_id,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        rows = session.execute(select(Chunk)).scalars().all()
        total = len(rows)
        _set_status(session, job_id, chunks_total=total)
        for i, _c in enumerate(rows, start=1):
            time.sleep(0.02)
            if i % 10 == 0 or i == total:
                _set_status(session, job_id, chunks_processed=i)
        _set_status(
            session,
            job_id,
            status="success",
            completed_at=datetime.now(timezone.utc),
            chunks_processed=total,
        )
    return {"job_id": job_id, "total": total}
