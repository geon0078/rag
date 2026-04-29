"""처방 2 — 쿼리 재작성 + 의도 분해 (Adversarial T2/T4 회복용).

Adversarial 평가 (`reports/eval_adversarial.md`) 에서:
  T2_vague (recall@5 0.580): "이거 어떻게 해?" 같은 모호 쿼리는 retrieval
    이 무작위로 매칭됨.
  T4_multi_intent (recall@5 0.610, retry 52%): "수강신청 언제 시작이고
    어디서 해?" 처럼 두 의도가 섞이면 한 번 검색으로 둘 다 못 찾음.

본 모듈은 retrieval 직전에 Solar Pro 1회 호출로 쿼리를 분류·재작성한다.

  type=single|normal → rewrites = [original]              (재작성 안 함)
  type=multi         → rewrites = [sub_q1, sub_q2, ...]   (분해)
  type=vague         → rewrites = [original, best_guess]  (둘 다 retrieve)

JSON 파싱 실패 / Solar 오류 / timeout 시에는 [original] 로 안전 fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from src.generation.solar_llm import SolarLLM
from src.utils.logger import get_logger

log = get_logger(__name__)

IntentType = Literal["single", "multi", "vague", "normal"]


@dataclass
class RewriteResult:
    type: IntentType
    rewrites: list[str]
    original: str = ""
    raw: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {"type": self.type, "rewrites": self.rewrites, "note": self.note}


_SYSTEM = (
    "당신은 한국어 대학 학사 챗봇의 쿼리 분석기입니다. "
    "사용자 입력 쿼리를 분석해 검색에 더 유리한 형태로 재작성합니다."
)

_USER_TEMPLATE = """다음 쿼리의 의도를 분류하고 검색 가능한 형태로 재작성하세요.

분류 기준:
- "single": 명확한 단일 의도. 그대로 검색 가능.
- "multi": 두 개 이상의 독립적인 질문이 결합됨 (예: "X는 언제이고 Y는 어디?"). 분해 필요.
- "vague": 지시어("이거", "그거"), 짧은 구어체, 의도 불명. 명료화 필요.
- "normal": single 과 같음. 단어 길이가 정상이고 모호하지 않음.

규칙:
1. multi 인 경우: 독립적인 서브쿼리 2-3개로 분해. 각 서브쿼리는 그 자체로 검색 가능.
2. vague 인 경우: 가장 가능성 높은 의도로 best-guess 재작성 1개.
3. single/normal 인 경우: rewrites = [원본] 그대로.

JSON 으로만 출력 (다른 설명 금지):
{{"type": "single|multi|vague|normal", "rewrites": ["...", "..."]}}

쿼리: {query}"""


def _strip_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


class QueryRewriter:
    def __init__(self, llm: SolarLLM | None = None) -> None:
        self.llm = llm or SolarLLM()

    async def rewrite(self, query: str) -> RewriteResult:
        original = (query or "").strip()
        if not original:
            return RewriteResult(type="single", rewrites=[""], original="")

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER_TEMPLATE.format(query=original)},
        ]
        try:
            resp = await self.llm._chat(messages, stream=False, max_tokens=300)
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"rewriter Solar error → fallback to original: {exc}")
            return RewriteResult(
                type="single",
                rewrites=[original],
                original=original,
                note=f"solar_error: {exc}",
            )

        try:
            data = json.loads(_strip_fence(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning(f"rewriter JSON parse fail → fallback: {exc}")
            return RewriteResult(
                type="single",
                rewrites=[original],
                original=original,
                raw=raw,
                note=f"parse_error: {exc}",
            )

        intent = str(data.get("type", "single")).lower()
        if intent not in {"single", "multi", "vague", "normal"}:
            intent = "single"
        rewrites_raw = data.get("rewrites") or []
        if not isinstance(rewrites_raw, list):
            rewrites_raw = [str(rewrites_raw)]
        rewrites = [str(r).strip() for r in rewrites_raw if str(r).strip()]

        if intent == "multi":
            if len(rewrites) < 2:
                rewrites = [original]
                intent = "single"
            else:
                rewrites = rewrites[:3]
        elif intent == "vague":
            best = rewrites[0] if rewrites else original
            rewrites = [original, best] if best != original else [original]
        else:
            rewrites = [original]

        return RewriteResult(
            type=intent,  # type: ignore[arg-type]
            rewrites=rewrites,
            original=original,
            raw=raw,
        )
