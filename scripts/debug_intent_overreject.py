"""A/B harness for intent classifier prompt variants.

Builds a mini eval set (over-rejected from latest eval_supplementary.json,
true negatives, control positives) and runs all variants in parallel.
Saves comparison to reports/intent_prompt_ab.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT / "data" / "qa.parquet"
EVAL = ROOT / "reports" / "eval_supplementary.json"
OUT = ROOT / "reports" / "intent_prompt_ab.json"

PROMPTS = {
    "V6_blocklist_only": """당신은 을지대학교 RAG 챗봇의 입력 필터입니다. 답은 두 단어 중 하나입니다: answerable 또는 unanswerable.

기본값은 answerable. 아래 4가지 정확한 패턴 중 하나에 일치할 때만 unanswerable.

UNANSWERABLE 4가지 패턴:
1. "내 X" 형식의 본인만 아는 정보: "내 비밀번호", "내 학번", "내 시간표", "내 성적", "내 이메일"
2. 5자 이하 + 학교 단어 없음: "안녕", "그거", "?", "...", "테스트", "ㅁㄴㅇㄹ", "ㅎㅎ"
3. 외부 정보 키워드: "버스 시간표", "오늘 날씨", "삼성/LG/현대 채용", 정치인 이름, 연예인 이름, 일반 시사
4. 학사일정 거짓 단언: "수강신청 12월 1일 맞아?" (실제 일정과 다른 단정)

위 4가지 중 하나에 정확히 일치하지 않으면 → answerable.
질문이 길거나, 시나리오가 있거나, 조건이 붙어있거나, 잘 모르는 단어가 있어도 → answerable.
"외국인 유학생 장학금", "기숙사 와이파이", "강의 추천", "택배 접수", "교차수강", "전자책 신청" 등 → 모두 answerable.

응답: answerable 또는 unanswerable""",
    "V1_current": """당신은 을지대학교 RAG 챗봇의 입력 필터입니다.
사용자 질문이 "을지대학교에 관한 질문"이면 무조건 answerable로 분류하세요.
"학교와 명백히 무관"하거나 "개인 비밀 정보"를 묻거나 "의미 없는 입력"인 경우만 unanswerable입니다.

[answerable — 학교에 관한 모든 질문] (기본값)
- 학과·전공·교육과정·학칙·학사일정·졸업·입학·수강신청·장학금·휴복학
- 캠퍼스(성남/대전/의정부), 학교 시설·연락처·전화번호, 강의평가, 도서관, 학식
- 학교 행정 절차, 증명서 발급, 동아리, 학생회

[unanswerable — 다음 4가지만]
1. 개인 비밀 정보: "내 비밀번호", "내 학번", "내 시간표"
2. 의미 없는 입력: "안녕", "그거", "?", "...", "테스트"
3. 학교와 무관한 외부 정보: 시내버스, 날씨, 타사 채용, 정치, 연예인
4. 거짓 전제 (확인된 경우)

판정이 애매하면 answerable.

응답: answerable 또는 unanswerable 중 하나만 출력.""",
    "V2_explicit_categories": """당신은 을지대학교 RAG 챗봇의 입력 필터입니다.
**기본값은 answerable**입니다. 아래 4가지 unanswerable 카테고리에 명백히 해당할 때만 unanswerable로 답하세요.

[answerable — 학교에 관한 모든 질문]
모든 학교 관련 주제 (예시일 뿐, 더 많은 주제 포함):
- 학과·전공·교육과정·학칙·학사일정·졸업·입학·수강신청·장학금·휴복학
- 캠퍼스(성남/대전/의정부), 학교 시설·연락처·전화번호
- **강의평가** (강의 만족도, 교수 평가, 수업 후기, 강의 추천 등)
- **시설_연락처** (도서관·식당·행정실·학생회실 전화번호, 위치, 운영시간)
- **FAQ** (자주 묻는 질문, 일반 안내, 처음 학교 다니는 학생 질문)
- 학교 행정 절차, 증명서 발급, 동아리, 학생회, 학식, 도서관, 기숙사

답이 코퍼스에 없을 가능성이 있어도, 학교에 관한 것이면 무조건 answerable입니다.

[unanswerable — 4가지 카테고리만, 명백한 경우만]
1. 개인 비밀: "내 비밀번호", "내 학번", "내 시간표" 같이 본인만 아는 정보
2. 의미 없음: "안녕", "그거", "?", "...", "테스트", "ㅁㄴㅇㄹ" 같이 질문 형식 미달
3. 외부 정보: 시내버스 시간표, 날씨, 타사 채용, 정치, 연예인, 일반 상식
4. 거짓 전제: 확인된 학사일정과 명백히 다른 날짜 단언 ("수강신청 12월 1일 맞아?")

확신이 없으면 무조건 answerable. RAG가 알아서 처리합니다.

응답: answerable 또는 unanswerable 중 하나만 출력.""",
    "V3_strict_unanswerable": """당신은 을지대학교 RAG 챗봇의 입력 필터입니다.

**규칙: unanswerable은 다음 4가지 좁은 케이스에만 적용. 그 외 모든 입력은 answerable.**

[unanswerable — 이 4가지 카테고리에 정확히 일치할 때만]
1. 개인 비밀 정보: 본인만 아는 정보 요청 (비밀번호/학번/시간표/성적)
2. 의미 없는 입력: 5자 미만의 인사말, 빈 입력, 키보드 난타, "그거", "..." 등
3. 외부 정보: 학교와 무관한 일반 시내버스/날씨/타사/정치/연예인 질문
4. 거짓 전제: 사실 확인된 정보와 다른 단언

[answerable — 위 4가지에 해당하지 않는 모든 입력]
학교 관련 주제이면 answerable:
- 학과/전공/교육과정/학칙/학사일정/졸업/입학/수강신청/장학금
- 강의평가 (교수평가, 수업후기, 강의추천)
- 시설 연락처 (도서관/식당/학생회/행정실 전화번호, 위치, 운영시간)
- FAQ (자주 묻는 질문, 처음 학교 다니는 학생의 일반 안내)
- 캠퍼스, 동아리, 학생회, 학식, 기숙사, 휴복학

답이 코퍼스에 없을 가능성이 있어도 answerable.
질문이 짧고 모호해도, 학교 관련 단서가 조금이라도 있으면 answerable.

응답: answerable 또는 unanswerable 중 하나만 출력.""",
    "V5_strict_neg_with_anchors": """당신은 을지대학교 RAG 챗봇의 입력 필터입니다.

**중요한 규칙 두 가지:**
1. unanswerable은 아래 4가지 케이스에만 정확히 적용. 그 외 모든 입력은 answerable.
2. 학교 시설/학사/장학금/규정/도서관/식당/기숙사 등 학교 관련 단어가 하나라도 있으면 무조건 answerable.

[answerable 예시 — 학교 관련은 모두 포함]
- "택배 접수 어디?" → answerable (시설)
- "멀티미디어실 좌석 예약 어떻게?" → answerable (시설)
- "기숙사 와이파이 설정 방법" → answerable (기숙사)
- "국외 전자책 신청 절차" → answerable (도서관)
- "교차수강 폐강 시 처리는?" → answerable (학칙)
- "외국인 유학생 장학금 종류" → answerable (장학금)
- "HUMAN 영역 과목 분류" → answerable (교육과정)
- "이 수업에서 주의해야 할 점은?" → answerable (강의평가, 컨텍스트는 RAG가 처리)
- "시험 준비 어떻게 해?" → answerable (학습)
- "두 강의 비교해줘" → answerable (강의평가)

[unanswerable — 4가지에 정확히 일치할 때만]
1. 개인 비밀: "내 비밀번호", "내 학번", "내 시간표", "내 성적" (본인만 아는 정보)
2. 의미 없음: "안녕", "그거", "?", "...", "ㅁㄴㅇㄹ", "테스트" (질문 형식 미달)
3. 외부 정보: "버스 시간표", "오늘 날씨", "삼성전자 채용", 정치/연예인/일반상식
4. 거짓 전제: "성남에 X학과 있지?" 같이 사실과 다른 단언 (확인된 경우)

[애매한 경우 처리]
- 학교 키워드(수업/강의/시설/캠퍼스/학과/장학금/도서관/식당/기숙사)가 있으면 → answerable
- 키워드 없이 "그거", "이거", "어떻게" 만으로 끝나면 → unanswerable
- 모르겠으면 → answerable (RAG가 처리)

응답: answerable 또는 unanswerable 중 하나만 출력.""",
    "V4_few_shot_anchors": """당신은 을지대학교 RAG 챗봇의 입력 필터입니다.

다음 예시들이 정답입니다. 패턴을 학습하세요.

[answerable 예시]
- "수강신청 언제 해?" → answerable
- "간호학과 졸업 학점은?" → answerable
- "성남캠퍼스 도서관 전화번호" → answerable (시설_연락처)
- "OOO 교수님 강의평가 어때?" → answerable (강의평가)
- "휴학 어떻게 해?" → answerable (FAQ)
- "장학금 신청 방법" → answerable
- "교양 추천해줘" → answerable
- "학식 메뉴 알려줘" → answerable
- "졸업 요건 뭐야?" → answerable
- "성남에 OO학과 있어?" → answerable

[unanswerable 예시]
- "내 비밀번호 뭐였지?" → unanswerable (개인 비밀)
- "내 학번 알려줘" → unanswerable (개인 비밀)
- "버스 시간표 알려줘" → unanswerable (외부 정보)
- "오늘 날씨" → unanswerable (외부)
- "삼성전자 채용" → unanswerable (외부)
- "..." → unanswerable (의미 없음)
- "?" → unanswerable (의미 없음)
- "그거 어떻게 해?" → unanswerable (의미 없음)
- "안녕" → unanswerable (의미 없음)

규칙:
- 학교 관련이면 무조건 answerable
- 개인 비밀, 외부 정보, 의미 없는 입력만 unanswerable
- 애매하면 answerable

응답: answerable 또는 unanswerable 중 하나만 출력.""",
}


async def classify(client: AsyncOpenAI, system_prompt: str, query: str) -> str:
    resp = await client.chat.completions.create(
        model="solar-pro3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[질문]\n{query}\n\n[판정]"},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    if "unanswerable" in text:
        return "unanswerable"
    if "answerable" in text:
        return "answerable"
    return "answerable"


def build_eval_set() -> list[dict]:
    df = pd.read_parquet(QA)
    report = json.load(open(EVAL, encoding="utf-8"))

    over_rejected_qids = [
        f["qid"] for f in report["routing"]["failures"] if not f["got_top3"]
    ]
    over_rej_rows = df[df["qid"].isin(over_rejected_qids)]
    over_rej = [
        {"qid": r.qid, "query": r.query, "expected": "answerable", "group": "over_rejected"}
        for r in over_rej_rows.itertuples()
    ]

    negatives = [
        ("qa_neg_004", "버스 시간표 알려줘"),
        ("qa_neg_011", "그거 어떻게 해?"),
        ("qa_neg_014", "..."),
        ("qa_neg_024", "내 비밀번호 뭐였지?"),
        ("qa_neg_028", "성남캠퍼스에 간호학과 있지?"),
    ]
    neg = [
        {"qid": q, "query": qy, "expected": "unanswerable", "group": "true_negative"}
        for q, qy in negatives
    ]

    controls = [
        "수강신청 언제 시작해?",
        "간호학과 졸업 학점은?",
        "성남캠퍼스 도서관 전화번호 알려줘",
        "장학금 신청 절차 알려줘",
        "휴학 신청은 어떻게 해?",
    ]
    ctrl = [
        {"qid": f"ctrl_{i}", "query": q, "expected": "answerable", "group": "control"}
        for i, q in enumerate(controls)
    ]

    return over_rej + neg + ctrl


async def run_variant(
    client: AsyncOpenAI, name: str, prompt: str, items: list[dict]
) -> dict:
    sem = asyncio.Semaphore(8)

    async def one(item: dict) -> dict:
        async with sem:
            verdict = await classify(client, prompt, item["query"])
            return {**item, "got": verdict, "correct": verdict == item["expected"]}

    results = await asyncio.gather(*[one(i) for i in items])
    by_group: dict[str, dict] = {}
    for g in {r["group"] for r in results}:
        sub = [r for r in results if r["group"] == g]
        by_group[g] = {"n": len(sub), "correct": sum(1 for r in sub if r["correct"])}
    return {
        "name": name,
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "by_group": by_group,
        "errors": [
            {"qid": r["qid"], "query": r["query"], "expected": r["expected"], "got": r["got"]}
            for r in results
            if not r["correct"]
        ],
    }


async def main() -> None:
    items = build_eval_set()
    print(
        f"eval set: {len(items)} items "
        f"(over_rejected={sum(1 for x in items if x['group']=='over_rejected')}, "
        f"true_negative={sum(1 for x in items if x['group']=='true_negative')}, "
        f"control={sum(1 for x in items if x['group']=='control')})"
    )

    client = AsyncOpenAI(
        api_key=settings.upstage_api_key,
        base_url=settings.upstage_base_url,
    )

    out: dict = {"items": items, "variants": {}}
    for name, prompt in PROMPTS.items():
        print(f"running {name}...")
        result = await run_variant(client, name, prompt, items)
        out["variants"][name] = result
        print(f"  -> {result['correct']}/{result['total']} | by_group={result['by_group']}")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
