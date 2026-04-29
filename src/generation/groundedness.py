"""Groundedness verifier using Solar LLM as a judge.

Upstage retired the dedicated `solar-1-mini-groundedness-check` model.
We replicate the same contract — verdict in {grounded, notGrounded, notSure} —
by prompting `solar-pro2` to judge whether the answer is supported by the context.
"""

from __future__ import annotations

import re
from typing import Literal

from openai import AsyncOpenAI, OpenAI

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


GroundednessResult = Literal["grounded", "notGrounded", "notSure"]
_JUDGE_MODEL = "solar-pro3"

_JUDGE_SYSTEM_PROMPT = """당신은 RAG 응답의 사실 일치성을 평가하는 심사관입니다.
주어진 [참고 문서]를 근거로, [답변]의 핵심 주장(main claims)이 문서에 의해 뒷받침되는지 판정하세요.

판정 원칙 (관대한 기준 — 검색이 정답을 가져왔다면 답변은 대부분 grounded):
- 답변의 핵심 주장이 문서에 직접 등장하거나 paraphrase·요약 형태로 존재하면 grounded
- **답변이 다루는 주제가 문서와 같고, 답변 내용이 문서로부터 합리적으로 도출 가능하면 grounded** (정확히 일치하지 않아도 됨)
- 여러 문서를 종합한 합리적 추론, 명시된 정보로부터의 직접적 계산, 그리고 단순 적용·재해석은 모두 grounded
- 표현 차이, 단어 선택, 어순, 문장 분할 차이는 모두 무시
- "정보 없음" 등 거부 답변도 문서와 모순되지 않으면 grounded
- **notGrounded는 오직 답변이 문서와 명백히 모순되거나 문서에 없는 새로운 구체적 사실(고유명사/숫자/날짜)을 발명한 경우에만**
- notSure는 답변이 핵심에서 크게 벗어나거나 모호한 경우에만 보수적으로 사용

응답 형식: 반드시 다음 셋 중 하나의 단어만 출력하세요. 다른 설명이나 부가 텍스트를 포함하지 마세요.
grounded
notGrounded
notSure"""


_JUDGE_USER_TEMPLATE = """[참고 문서]
{context}

[답변]
{answer}

[판정]"""


_VALID = {"grounded", "notGrounded", "notSure"}


def _normalize(raw: str) -> GroundednessResult:
    text = (raw or "").strip()
    if text in _VALID:
        return text  # type: ignore[return-value]
    # Try to find any of the valid tokens inside a longer string (case-insensitive variants)
    lowered = text.lower()
    for token in ("notgrounded", "grounded", "notsure"):
        if token in lowered:
            mapping = {"notgrounded": "notGrounded", "grounded": "grounded", "notsure": "notSure"}
            return mapping[token]  # type: ignore[return-value]
    # Some judges may emit Korean labels — map common variants
    if re.search(r"근거\s*없음|뒷받침\s*안", text):
        return "notGrounded"
    if re.search(r"근거\s*있음|뒷받침\s*됨", text):
        return "grounded"
    log.warning(f"unexpected groundedness verdict: {text!r}; treating as notSure")
    return "notSure"


def _build_messages(context: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(context=context, answer=answer)},
    ]


class GroundednessChecker:
    def __init__(self, model: str = _JUDGE_MODEL) -> None:
        self.model = model
        self.async_client = AsyncOpenAI(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )
        self.sync_client = OpenAI(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )

    async def verify(self, context: str, answer: str) -> GroundednessResult:
        resp = await self.async_client.chat.completions.create(
            model=self.model,
            messages=_build_messages(context, answer),
            temperature=0.0,
            max_tokens=10,
        )
        verdict = _normalize(resp.choices[0].message.content or "")
        log.info(f"groundedness: {verdict}")
        return verdict

    def verify_sync(self, context: str, answer: str) -> GroundednessResult:
        resp = self.sync_client.chat.completions.create(
            model=self.model,
            messages=_build_messages(context, answer),
            temperature=0.0,
            max_tokens=10,
        )
        return _normalize(resp.choices[0].message.content or "")
