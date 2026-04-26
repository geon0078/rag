"""LLM-based intent classifier: decides whether a query is answerable from the
EulJi University knowledge base before any retrieval/generation happens.

Replaces hardcoded keyword/regex matching (forbidden per CLAUDE.md). The judge
is asked to return a single token verdict, so latency stays ~300ms per query.

Coverage targets (negative QAs that previously slipped past the relaxed
groundedness judge):
  - 사적 정보 요청 ("내 비밀번호 뭐였지?")
  - 모호/공백 입력 ("그거 어떻게 해?", "...", "?")
  - 외부 서비스 ("버스 시간표", "오늘 날씨", "삼성전자 채용")
  - 일반 상식이거나 코퍼스에 없는 사실
"""

from __future__ import annotations

from typing import Literal

from openai import AsyncOpenAI, OpenAI

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


IntentResult = Literal["answerable", "unanswerable"]
_INTENT_MODEL = "solar-pro3"


_INTENT_SYSTEM_PROMPT = """당신은 을지대학교 RAG 챗봇의 입력 필터입니다.
사용자 질문이 "을지대학교에 관한 질문"이면 무조건 answerable로 분류하세요.
"학교와 명백히 무관"하거나 "개인 비밀 정보"를 묻거나 "의미 없는 입력"인 경우만 unanswerable입니다.

[answerable — 학교에 관한 모든 질문] (기본값)
- 학과·전공·교육과정·학칙·학사일정·졸업·입학·수강신청·장학금·휴복학
- 캠퍼스(성남/대전/의정부), 학교 시설·연락처·전화번호, 강의평가, 도서관, 학식
- 학교 행정 절차, 증명서 발급, 동아리, 학생회
- "수강신청 언제 해?", "간호학과 졸업 학점?", "성남에 OO학과 있어?" 같은 일반적 학교 질문은 모두 answerable
- 답이 코퍼스에 없을 가능성이 있어도, 학교에 관한 질문이면 answerable

[unanswerable — 다음 4가지만]
1. 개인 비밀 정보 요청: "내 비밀번호", "내 학번 알려줘", "내 시간표"
2. 의미 없는/너무 짧은 입력: "안녕", "그거", "?", "...", "테스트", "ㅁㄴㅇㄹ"
3. 학교와 명백히 무관한 외부 정보: 일반 시내버스 시간표, 날씨, 타사 채용, 정치, 연예인, 일반 상식
4. 사실과 다른 거짓 전제 (확인된 경우): "수강신청 12월 1일 맞아?" (실제 학사일정과 다른 명백한 거짓)

판정이 애매하면 answerable을 선택하세요. RAG가 알아서 처리합니다.

응답 형식: 반드시 다음 둘 중 하나의 단어만 출력하세요. 다른 설명을 포함하지 마세요.
answerable
unanswerable"""


_INTENT_USER_TEMPLATE = "[질문]\n{query}\n\n[판정]"


_VALID = {"answerable", "unanswerable"}


def _normalize(raw: str) -> IntentResult:
    text = (raw or "").strip().lower()
    if text in _VALID:
        return text  # type: ignore[return-value]
    if "unanswerable" in text:
        return "unanswerable"
    if "answerable" in text:
        return "answerable"
    log.warning(f"unexpected intent verdict: {text!r}; defaulting to answerable")
    return "answerable"


def _build_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": _INTENT_USER_TEMPLATE.format(query=query)},
    ]


class IntentClassifier:
    def __init__(self, model: str = _INTENT_MODEL) -> None:
        self.model = model
        self.async_client = AsyncOpenAI(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )
        self.sync_client = OpenAI(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )

    async def classify(self, query: str) -> IntentResult:
        resp = await self.async_client.chat.completions.create(
            model=self.model,
            messages=_build_messages(query),
            temperature=0.0,
            max_tokens=5,
        )
        verdict = _normalize(resp.choices[0].message.content or "")
        log.info(f"intent: {verdict}")
        return verdict

    def classify_sync(self, query: str) -> IntentResult:
        resp = self.sync_client.chat.completions.create(
            model=self.model,
            messages=_build_messages(query),
            temperature=0.0,
            max_tokens=5,
        )
        return _normalize(resp.choices[0].message.content or "")

    async def is_answerable(self, query: str) -> bool:
        return (await self.classify(query)) == "answerable"
