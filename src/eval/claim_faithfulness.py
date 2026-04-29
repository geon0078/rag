"""Claim-level faithfulness — RAGChecker 패턴 (평가명세서 §5).

기존 RAGAS / Solar Groundedness 는 답변 전체에 단일 라벨을 주지만, 5문장 답변
중 1문장만 환각이어도 사고가 됩니다. 본 모듈은 답변을 atomic claim 으로
분해한 뒤 각 claim 을 컨텍스트에 대해 entailment 검증해서 부분 환각도 잡아냅니다.

흐름:
  1. ``extract_claims(answer)`` — Solar 가 답변을 사실 단위 list 로 분해
  2. ``verify_claim(claim, context)`` — Solar 가 단일 claim 의 supported / not_supported / partial 판정
  3. ``score_answer(answer, context)`` — 두 단계를 묶어 0..1 faithfulness + 상세 결과 반환

Mode E 안전장치 정책 (§5.4) 은 본 모듈을 사용하는 호출자(파이프라인)가 적용:
  - groundedness=grounded → 통과
  - groundedness=notSure + claim_faithfulness >= 0.8 → 통과
  - 그 외 → 차단
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from openai import APIError, AsyncOpenAI, RateLimitError

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

JUDGE_MODEL = "solar-pro3"
TEMPERATURE = 0.0
MAX_TOKENS_EXTRACT = 600
MAX_TOKENS_VERIFY = 80
RETRY_MAX = 3
RETRY_BASE = 1.5


ClaimVerdict = Literal["supported", "not_supported", "partial"]


@dataclass
class ClaimResult:
    claim: str
    verdict: ClaimVerdict
    rationale: str = ""


@dataclass
class FaithfulnessResult:
    score: float                          # supported 비율 (partial = 0.5)
    n_claims: int
    claims: list[ClaimResult] = field(default_factory=list)


_EXTRACT_SYSTEM = """다음 한국어 답변을 atomic 사실 단위(claim)로 쪼개세요.

규칙:
1. 각 claim 은 한 문장의 짧은 사실 진술 (예: "졸업학점은 130학점이다").
2. 한 claim 안에 여러 사실이 들어가지 않게 (and 로 묶지 말 것).
3. 출처 표기 (`[출처: ...]`), 인사말, 메타 코멘트는 제외.
4. 빈 답변·거부 답변("정보를 찾을 수 없습니다" 등)은 빈 list 반환.

반드시 다음 JSON 한 줄만 출력. 다른 텍스트 금지:
{"claims": ["claim 1", "claim 2", ...]}"""


_VERIFY_SYSTEM = """주어진 [참고 문서]에 비추어 [주장] 이 사실로 뒷받침되는지 판정하세요.

판정 기준:
- "supported"   = 주장이 문서에 직접 등장하거나 paraphrase·요약으로 존재
- "partial"     = 일부만 뒷받침되거나, 주장 일부에 표현 차이가 있어 불확실
- "not_supported" = 문서와 모순되거나 문서에 전혀 없는 새로운 정보

반드시 다음 JSON 한 줄만 출력. 다른 텍스트 금지:
{"verdict": "supported|partial|not_supported", "rationale": "<한 문장>"}"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_codefence(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw


def _extract_json(raw: str) -> dict | None:
    raw = _strip_codefence(raw)
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class ClaimFaithfulnessChecker:
    """Two-step Solar judge for claim-level faithfulness."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.upstage_api_key, base_url=settings.upstage_base_url
        )

    async def _chat(self, system: str, user: str, max_tokens: int) -> str:
        attempt = 0
        delay = RETRY_BASE
        while True:
            attempt += 1
            try:
                resp = await self.client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError:
                if attempt >= RETRY_MAX:
                    raise
                log.warning(f"claim_faithfulness: 429 retry {attempt}/{RETRY_MAX}")
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)
            except APIError:
                if attempt >= 2:
                    raise
                await asyncio.sleep(delay)

    async def extract_claims(self, answer: str) -> list[str]:
        if not answer or not answer.strip():
            return []
        raw = await self._chat(_EXTRACT_SYSTEM, f"[답변]\n{answer}\n\n[출력]", MAX_TOKENS_EXTRACT)
        parsed = _extract_json(raw)
        if not parsed or "claims" not in parsed:
            log.warning(f"extract_claims: parse failed {raw[:120]!r}")
            return []
        claims = parsed["claims"]
        if not isinstance(claims, list):
            return []
        return [str(c).strip() for c in claims if str(c).strip()]

    async def verify_claim(self, claim: str, context: str) -> ClaimResult:
        user = f"[참고 문서]\n{context}\n\n[주장]\n{claim}\n\n[판정]"
        raw = await self._chat(_VERIFY_SYSTEM, user, MAX_TOKENS_VERIFY)
        parsed = _extract_json(raw)
        if not parsed or parsed.get("verdict") not in {"supported", "partial", "not_supported"}:
            return ClaimResult(claim=claim, verdict="partial",
                               rationale=f"unparseable: {raw[:80]!r}")
        return ClaimResult(
            claim=claim,
            verdict=parsed["verdict"],
            rationale=str(parsed.get("rationale", "")),
        )

    async def score_answer(self, answer: str, context: str) -> FaithfulnessResult:
        """답변 전체에 대해 claim-level faithfulness 계산.

        score = (supported * 1.0 + partial * 0.5) / n_claims
        n_claims == 0 (예: fallback "정보 없음") → score = 1.0 (안전 기본값)
        """
        claims = await self.extract_claims(answer)
        if not claims:
            return FaithfulnessResult(score=1.0, n_claims=0)
        results = await asyncio.gather(*(self.verify_claim(c, context) for c in claims))
        weight = {"supported": 1.0, "partial": 0.5, "not_supported": 0.0}
        score = sum(weight[r.verdict] for r in results) / max(1, len(results))
        return FaithfulnessResult(score=score, n_claims=len(results), claims=list(results))
