"""Metadata v3 평탄 dict 스키마 (운영웹통합명세서 §3).

3-Layer 구조:
  Layer 1 Core        — 모든 청크 필수
  Layer 2 Domain      — 컬렉션별 조건부 (다른 컬렉션은 None / key 부재)
  Layer 3 Operations  — 운영 추적 (검색·LLM 모두 제외)

Pydantic v2 BaseModel 단일 클래스로 정의 (평탄 dict 원칙). 컬렉션별 도메인
필드는 모두 Optional 로 선언하고, ``model_validator`` 로 ``source_collection``
값에 따른 필수 필드 검증을 수행한다.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


VALID_SOURCE_COLLECTIONS = (
    "강의평가",
    "학칙_조항",
    "학사정보",
    "FAQ",
    "시설_연락처",
    "학사일정",
    "장학금",
    "교육과정",
    "학과정보",
    "기타",
)
VALID_CAMPUSES = ("성남", "의정부", "대전", "전체")
VALID_CONFIDENCE = ("high", "medium", "low")
VALID_LANGUAGES = ("ko", "en")
VALID_STATUS = ("Draft", "Published", "Indexed", "Archived")


SourceCollection = Literal[
    "강의평가",
    "학칙_조항",
    "학사정보",
    "FAQ",
    "시설_연락처",
    "학사일정",
    "장학금",
    "교육과정",
    "학과정보",
    "기타",
]
Campus = Literal["성남", "의정부", "대전", "전체"]
Confidence = Literal["high", "medium", "low"]
Language = Literal["ko", "en"]
Status = Literal["Draft", "Published", "Indexed", "Archived"]


class MetadataV3(BaseModel):
    """Layer 1 Core + Layer 2 Domain + Layer 3 Operations 평탄 dict."""

    model_config = ConfigDict(extra="allow")  # unknown 필드 보존 (마이그레이션 안전)

    # === Layer 1: Core ===
    doc_id: str
    parent_doc_id: str | None = None
    path: str
    breadcrumb: list[str] = Field(default_factory=list)
    schema_version: Literal["v3"] = "v3"
    source_collection: SourceCollection
    category: str
    subcategory: str | None = None
    title: str
    campus: Campus = "전체"
    language: Language = "ko"
    chunk_index: int = 0
    chunk_count: int = 1
    depth: int = 1

    # === Layer 2: Domain (컬렉션별) ===
    chapter: str | None = None
    chapter_title: str | None = None
    article_number: str | None = None
    article_title: str | None = None
    paragraph: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    semester: str | None = None
    event_type: str | None = None
    lecture_id: str | None = None
    lecture_title: str | None = None
    section: str | None = None
    subject_area: str | None = None
    is_required: bool | None = None
    question_canonical: str | None = None
    question_variants: list[str] | None = None
    keywords: list[str] | None = None
    negative_examples: list[str] | None = None
    phone: str | None = None
    building: str | None = None
    floor: str | None = None
    facility_type: str | None = None
    scholarship_type: str | None = None
    application_period_start: date | None = None
    application_period_end: date | None = None
    eligibility_grade: float | None = None
    department: str | None = None
    low_confidence: bool | None = None

    # === Layer 3: Operations ===
    effective_start: date | None = None
    effective_end: date | None = None
    created_at: datetime | None = None
    indexed_at: datetime | None = None
    last_verified_at: date | None = None
    version: int = 1
    owner: str | None = None
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def _check_collection_required(self) -> "MetadataV3":
        sc = self.source_collection
        if sc == "학사일정" and self.start_date is None:
            raise ValueError("source_collection=학사일정 인 경우 start_date 필수")
        if sc == "학칙_조항" and not self.article_number:
            raise ValueError("source_collection=학칙_조항 인 경우 article_number 필수")
        if sc == "시설_연락처" and not self.phone:
            raise ValueError("source_collection=시설_연락처 인 경우 phone 필수")
        if sc == "강의평가" and not self.lecture_id:
            raise ValueError("source_collection=강의평가 인 경우 lecture_id 필수")
        if self.depth <= 0 and self.breadcrumb:
            object.__setattr__(self, "depth", len(self.breadcrumb))
        if self.effective_start and self.effective_end and self.effective_end < self.effective_start:
            raise ValueError("effective_end 가 effective_start 보다 빠를 수 없음")
        return self


# === 임베딩·LLM 노출 키 (Layer 1+2 일부) ===
EMBED_INCLUDE_FIELDS: tuple[str, ...] = (
    "title", "category", "subcategory", "campus",
    "section", "keywords", "question_canonical", "question_variants",
    "article_number", "article_title", "lecture_title",
)

LLM_INCLUDE_FIELDS: tuple[str, ...] = (
    "title", "category", "campus", "breadcrumb",
    "article_number", "article_title", "section",
    "start_date", "end_date", "phone",
)

OPERATIONS_FIELDS: tuple[str, ...] = (
    "effective_start", "effective_end", "created_at", "indexed_at",
    "last_verified_at", "version", "owner",
    "confidence", "schema_version",
)


def filter_for_embedding(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k in EMBED_INCLUDE_FIELDS and v is not None}


def filter_for_llm(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k in LLM_INCLUDE_FIELDS and v is not None}
