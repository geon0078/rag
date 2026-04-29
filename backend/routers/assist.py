"""POST /api/chunks/* assist endpoints — Solar Pro 자동 보조 (운영웹통합명세서 §8.3 + §13).

운영자가 FAQ/학칙 작성 중 [✨] 버튼으로 호출하는 자동 생성 엔드포인트들.
모두 사람 검수가 마지막 단계 — LLM 출력은 단지 초안.

내부적으로 ``src.generation.solar_llm.SolarLLM`` 을 재사용 (PYTHONPATH /workspace).

Endpoints (§8.3):
  POST /api/chunks/{doc_id}/generate-variants
       Response: { variants: ["...", "..."] }            — 5개 변형 질문
  POST /api/chunks/{doc_id}/generate-keywords
       Response: { keywords: ["...", "..."] }            — 키워드 5개
  POST /api/chunks/parse-article
       Body: { text: "제31조 1항 ..." }
       Response: { chapter: "...", article_number: "..." }   — 학칙 파싱

추가 helper (§13):
  POST /api/chunks/{doc_id}/generate-negatives  — negative_examples 제안
  POST /api/chunks/{doc_id}/generate-summary    — answer_short 1문장 요약
"""

from __future__ import annotations

import json
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from models import Chunk

router = APIRouter(prefix="/api/chunks", tags=["assist"])


_llm = None


def _get_llm():
    """Lazy import — Solar 의존성이 없는 환경에서 import 시점 폭발 방지."""
    global _llm
    if _llm is None:
        try:
            from src.generation.solar_llm import SolarLLM  # type: ignore
        except ImportError as exc:
            raise HTTPException(
                503,
                detail=f"SolarLLM import 실패 — PYTHONPATH/볼륨 확인: {exc}",
            )
        _llm = SolarLLM()
    return _llm


async def _solar_chat(messages: list[dict], max_tokens: int | None = None) -> str:
    """SolarLLM._chat 단일 호출 헬퍼."""
    llm = _get_llm()
    resp = await llm._chat(messages, stream=False, max_tokens=max_tokens)  # type: ignore[attr-defined]
    return (resp.choices[0].message.content or "").strip()


def _strip_code_fence(text: str) -> str:
    """LLM 이 ```json ...``` 으로 감싼 응답을 풀어준다."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_array(text: str) -> list[str]:
    """LLM 응답에서 string array 추출. 실패 시 줄바꿈 fallback."""
    s = _strip_code_fence(text)
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    lines = [re.sub(r"^\s*[\-\*\d\.\)]+\s*", "", ln).strip() for ln in s.splitlines()]
    return [ln for ln in lines if ln]


async def _load_chunk(doc_id: str, db: AsyncSession) -> Chunk:
    c = await db.get(Chunk, doc_id)
    if not c:
        raise HTTPException(404, f"chunk not found: {doc_id}")
    return c


def _faq_question_answer(c: Chunk) -> tuple[str, str]:
    """FAQ chunk 에서 (question, answer) 추출."""
    md = c.chunk_metadata or {}
    question = md.get("question") or md.get("title") or ""
    answer = md.get("answer") or c.contents or ""
    return question, answer


# ─────────────────────────────────────────────────────────────────
# variants
# ─────────────────────────────────────────────────────────────────


class VariantsResp(BaseModel):
    variants: list[str]


@router.post("/{doc_id}/generate-variants", response_model=VariantsResp)
async def generate_variants(doc_id: str, db: AsyncSession = Depends(get_db)) -> VariantsResp:
    """FAQ chunk 의 question 을 5개 변형질문으로 확장 (§13)."""
    c = await _load_chunk(doc_id, db)
    question, answer = _faq_question_answer(c)
    if not question:
        raise HTTPException(400, "FAQ chunk 에 question/title 이 없습니다")

    prompt = (
        "다음 FAQ 의 표준 질문을 학생들이 실제로 검색할 5개의 변형 질문으로 바꿔주세요.\n"
        "- 줄임말, 구어체, 관련 키워드를 자연스럽게 섞어 다양하게.\n"
        "- 의미는 동일하게 유지.\n"
        '- 결과는 JSON 배열로만 출력. 예: ["...", "...", "...", "...", "..."]\n\n'
        f"표준 질문: {question}\n"
        f"답변(맥락): {answer[:500]}"
    )
    text = await _solar_chat(
        [
            {"role": "system", "content": "당신은 한국어 검색 쿼리 변형 도우미입니다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
    )
    variants = _parse_json_array(text)[:5]
    if not variants:
        raise HTTPException(502, "Solar 응답을 파싱하지 못했습니다")
    return VariantsResp(variants=variants)


# ─────────────────────────────────────────────────────────────────
# keywords
# ─────────────────────────────────────────────────────────────────


class KeywordsResp(BaseModel):
    keywords: list[str]


@router.post("/{doc_id}/generate-keywords", response_model=KeywordsResp)
async def generate_keywords(doc_id: str, db: AsyncSession = Depends(get_db)) -> KeywordsResp:
    """답변에서 핵심 키워드 5개 추출 (§13)."""
    c = await _load_chunk(doc_id, db)
    question, answer = _faq_question_answer(c)
    body = answer or c.contents or ""
    if not body.strip():
        raise HTTPException(400, "답변/본문이 비어 키워드를 추출할 수 없습니다")

    prompt = (
        "다음 글에서 검색 키워드 5개를 한국어 명사 중심으로 추출하세요.\n"
        "- 단일 단어 또는 짧은 명사구.\n"
        "- 동의어/약어가 자연스러우면 포함.\n"
        '- 결과는 JSON 배열로만 출력. 예: ["...", "...", "...", "...", "..."]\n\n'
        f"질문: {question}\n"
        f"본문: {body[:1500]}"
    )
    text = await _solar_chat(
        [
            {"role": "system", "content": "당신은 한국어 키워드 추출 도우미입니다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )
    keywords = _parse_json_array(text)[:5]
    if not keywords:
        raise HTTPException(502, "Solar 응답을 파싱하지 못했습니다")
    return KeywordsResp(keywords=keywords)


# ─────────────────────────────────────────────────────────────────
# negative_examples
# ─────────────────────────────────────────────────────────────────


class NegativesResp(BaseModel):
    negative_examples: list[str]


@router.post("/{doc_id}/generate-negatives", response_model=NegativesResp)
async def generate_negatives(doc_id: str, db: AsyncSession = Depends(get_db)) -> NegativesResp:
    """비슷한 단어를 사용하지만 답이 다른 질문 3개 생성 (§13)."""
    c = await _load_chunk(doc_id, db)
    question, answer = _faq_question_answer(c)
    if not question:
        raise HTTPException(400, "FAQ chunk 에 question/title 이 없습니다")

    prompt = (
        "다음 FAQ 와 키워드는 비슷하지만 답이 다른 '오인 가능 질문' 3개를 만들어주세요.\n"
        "- 학생이 검색했을 때 이 FAQ 가 잘못 매칭될 수 있는 함정 질문.\n"
        "- 의미는 다르지만 어휘는 겹치도록.\n"
        '- 결과는 JSON 배열로만 출력. 예: ["...", "...", "..."]\n\n'
        f"표준 질문: {question}\n"
        f"답변: {answer[:500]}"
    )
    text = await _solar_chat(
        [
            {"role": "system", "content": "당신은 검색 함정 질문을 설계하는 한국어 QA 도우미입니다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
    )
    items = _parse_json_array(text)[:3]
    if not items:
        raise HTTPException(502, "Solar 응답을 파싱하지 못했습니다")
    return NegativesResp(negative_examples=items)


# ─────────────────────────────────────────────────────────────────
# answer_short
# ─────────────────────────────────────────────────────────────────


class SummaryResp(BaseModel):
    answer_short: str


@router.post("/{doc_id}/generate-summary", response_model=SummaryResp)
async def generate_summary(doc_id: str, db: AsyncSession = Depends(get_db)) -> SummaryResp:
    """답변을 1문장 요약 (§13)."""
    c = await _load_chunk(doc_id, db)
    _question, answer = _faq_question_answer(c)
    body = answer or c.contents or ""
    if not body.strip():
        raise HTTPException(400, "답변/본문이 비어 요약할 수 없습니다")

    prompt = (
        "다음 답변을 한 문장(70자 이내)으로 한국어 요약해주세요. "
        "답변 외 설명/접두어 없이 본문만 출력.\n\n"
        f"답변: {body[:1500]}"
    )
    text = await _solar_chat(
        [
            {"role": "system", "content": "당신은 한국어 한 문장 요약기입니다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
    )
    short = _strip_code_fence(text).splitlines()[0].strip()
    if not short:
        raise HTTPException(502, "Solar 응답이 비어있습니다")
    return SummaryResp(answer_short=short)


# ─────────────────────────────────────────────────────────────────
# parse-article (학칙)
# ─────────────────────────────────────────────────────────────────


class ParseArticleReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class ParseArticleResp(BaseModel):
    chapter: str | None
    article_number: str | None
    section: str | None = None
    raw_match: str | None = None


_RE_ARTICLE = re.compile(r"제\s*(\d+)\s*조")
_RE_CHAPTER = re.compile(r"제\s*(\d+)\s*장")
_RE_SECTION = re.compile(r"제\s*(\d+)\s*절")


@router.post("/parse-article", response_model=ParseArticleResp)
async def parse_article(req: ParseArticleReq) -> ParseArticleResp:
    """학칙 본문에서 제N장/제N조/제N절 파싱 (§13). 정규식 1차 + LLM 검증 2차."""
    text = req.text
    m_ch = _RE_CHAPTER.search(text)
    m_art = _RE_ARTICLE.search(text)
    m_sec = _RE_SECTION.search(text)

    chapter = m_ch.group(0).replace(" ", "") if m_ch else None
    article = m_art.group(0).replace(" ", "") if m_art else None
    section = m_sec.group(0).replace(" ", "") if m_sec else None

    if chapter and article:
        return ParseArticleResp(
            chapter=chapter,
            article_number=article,
            section=section,
            raw_match=(m_ch.group(0) if m_ch else "") + " / " + (m_art.group(0) if m_art else ""),
        )

    prompt = (
        "다음 학칙 본문에서 장/조/절 번호를 JSON 으로 추출. "
        "필드: chapter (예 '제2장'), article_number (예 '제31조'), section (예 '제1절', 없으면 null). "
        "발견되지 않은 필드는 null. JSON 외 문구 금지.\n\n"
        f"본문: {text[:1500]}"
    )
    raw = await _solar_chat(
        [
            {"role": "system", "content": "당신은 한국어 법령/학칙 구조 파서입니다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )
    try:
        data = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, ValueError):
        data = {}

    return ParseArticleResp(
        chapter=chapter or (data.get("chapter") if isinstance(data, dict) else None),
        article_number=article or (data.get("article_number") if isinstance(data, dict) else None),
        section=section or (data.get("section") if isinstance(data, dict) else None),
        raw_match=None,
    )
