"""SQLAlchemy 모델 (운영웹통합명세서 §9).

세 테이블:
  - chunks            — 청크 메타·콘텐츠 (v3 평탄 dict는 ``metadata`` JSONB 컬럼)
  - chunk_history     — 변경 이력 (낙관적 락 + diff)
  - indexing_jobs     — Celery 인덱싱 작업 큐 추적

지표 / GIN 인덱스:
  - status, source_collection, parent_doc_id, path 단일 컬럼 인덱스
  - metadata JSONB 위에 GIN 인덱스 (campus·effective_end 등 임의 키 필터)

ORM 주의: SQLAlchemy 의 ``metadata`` 는 예약 속성이므로 파이썬 속성명은
``chunk_metadata`` 로, DB 컬럼명은 ``metadata`` 로 매핑한다.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    __tablename__ = "chunks"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_doc_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    path: Mapped[str] = mapped_column(String, nullable=False, index=True)
    schema_version: Mapped[str] = mapped_column(String, default="v3")
    source_collection: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False)
    contents: Mapped[str] = mapped_column(Text, nullable=False)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="Draft", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    history: Mapped[list["ChunkHistory"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_chunks_metadata_gin", "metadata", postgresql_using="gin"),
    )


class ChunkHistory(Base):
    __tablename__ = "chunk_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String, ForeignKey("chunks.doc_id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    diff: Mapped[dict] = mapped_column(JSONB, nullable=False)

    chunk: Mapped["Chunk"] = relationship(back_populates="history")

    __table_args__ = (
        Index("idx_chunk_history_doc_version", "doc_id", "version"),
    )


class IndexingJob(Base):
    __tablename__ = "indexing_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="queued")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    chunks_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
