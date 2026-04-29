"""Augment data/qa.parquet to match 평가-개선.md schema.

Transforms:
1. Adds `qa_type` column derived from hop_type and campus mention.
   - filter_required: query mentions a 캠퍼스
   - multi_hop: hop_type == "multi" and no campus mention
   - single_hop: otherwise
2. Adds `metadata` column with `{"campus_filter": "성남"|"의정부"|"대전"|None}`.
3. Appends 30 negative-case rows from 평가-개선.md §3.8.
4. Backs up the original parquet to data/qa_v1_pre_negative.parquet.

Run:
    python scripts/finalize_qa.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

log = get_logger("finalize_qa")

QA_PATH = PROJECT_ROOT / "data" / "qa.parquet"
BACKUP_PATH = PROJECT_ROOT / "data" / "qa_v1_pre_negative.parquet"

CAMPUS_PATTERN = re.compile(r"(성남|의정부|대전)\s*캠퍼스")
NEGATIVE_EXPECTED_ANSWER = "제공된 자료에서 해당 정보를 찾을 수 없습니다."

NEGATIVE_QUERIES: list[str] = [
    # 데이터에 없는 일반 질문 (20개) — 시간/실시간 정보, 외부 일상
    "오늘 학식 메뉴 뭐야?",
    "오늘 날씨 어때?",
    "지금 도서관 자리 있어?",
    "버스 시간표 알려줘",
    "이번주 축제 일정 알려줘",
    "교수님 핸드폰 번호 알려줘",
    "동아리 신입생 모집하는 곳 어디?",
    "학교 앞 맛집 추천해줘",
    "지금 휴게실 사람 많아?",
    "오늘 휴강 있어?",
    "내일 비 와?",
    "지금 셔틀버스 어디쯤이야?",
    "체육관 운영 시간 지금 몇 시야?",
    "지금 카페 자리 있나?",
    "이번주 토요일에 학교 문 열어?",
    "오늘 공지사항 뭐 있어?",
    "지금 우산 빌릴 수 있어?",
    "근처 ATM 어디 있어?",
    "기숙사 식당 오늘 메뉴 뭐야?",
    "지금 분실물 보관소 가면 직원 있어?",
    # 모호하거나 의미 없는 질문 (10개)
    "그거 어떻게 해?",
    "?",
    "안녕",
    "...",
    "테스트",
    "음",
    "ㅎㅎ",
    "그래서?",
    "아무거나",
    "뭔가 알려줘",
    # 데이터 범위 외 학교/외부 질문 (10개)
    "서울대 가려면 어떻게 해?",
    "토익 시험 일정 알려줘",
    "삼성전자 채용공고 있어?",
    "운전면허 어디서 따?",
    "코로나 백신 어디서 맞아?",
    "스타벅스 메뉴 추천해줘",
    "유튜브 알고리즘은 어떻게 작동해?",
    "비트코인 사는 법 알려줘",
    "넷플릭스 신작 뭐야?",
    "공무원 시험 준비 방법 알려줘",
    # 개인정보 요구 (10개)
    "김철수 학생 학번 알려줘",
    "교수님 집 주소 알려줘",
    "이번 학기 성적 알려줘",
    "내 비밀번호 뭐였지?",
    "학생증 비밀번호 초기화해줘",
    "옆 학생 전화번호 알려줘",
    "내 출석률 알려줘",
    "다른 학생 장학금 수령 내역 보여줘",
    "내 신용카드 등록 정보 보여줘",
    "교직원 급여 정보 알려줘",
    # 특정 답을 강요하는 트랩 (10개)
    "내가 졸업학점 100학점이라고 하던데 맞지?",
    "수강신청 12월 1일 시작 맞지?",
    "성남캠퍼스에 간호학과 있지?",
    "장학금 100% 다 받을 수 있지?",
    "학칙 제999조에 뭐라고 써있어?",
    "수강신청은 항상 자정에 열리지?",
    "휴학은 5번까지 가능한 거 맞지?",
    "F학점 받아도 무조건 졸업할 수 있지?",
    "F학점 4번 받으면 자동 제적이지?",
    "이번 학기 등록금 인상은 0원이지?",
]


def _extract_campus(query: str | None) -> str | None:
    if not query:
        return None
    match = CAMPUS_PATTERN.search(query)
    return match.group(1) if match else None


def _classify_qa_type(query: str | None, hop_type: str | None) -> str:
    if _extract_campus(query):
        return "filter_required"
    if (hop_type or "").lower() == "multi":
        return "multi_hop"
    return "single_hop"


def _build_metadata(query: str | None) -> dict:
    return {"campus_filter": _extract_campus(query)}


def _build_negative_rows() -> pd.DataFrame:
    rows = []
    for idx, query in enumerate(NEGATIVE_QUERIES, start=1):
        rows.append(
            {
                "qid": f"qa_neg_{idx:03d}",
                "query": query,
                "retrieval_gt": [],
                "generation_gt": [NEGATIVE_EXPECTED_ANSWER],
                "source_collection": "negative",
                "hop_type": "negative",
                "qa_type": "negative",
                "metadata": {"campus_filter": None},
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    if not QA_PATH.exists():
        log.error(f"qa.parquet not found at {QA_PATH}")
        return 1

    df = pd.read_parquet(QA_PATH)
    log.info(f"Loaded qa.parquet: rows={len(df)}, cols={list(df.columns)}")

    if "qa_type" in df.columns and "metadata" in df.columns and "negative" in df.get("qa_type", pd.Series([])).unique():
        log.warning("qa.parquet already augmented (qa_type+negative present). Aborting to avoid double-append.")
        return 0

    if not BACKUP_PATH.exists():
        df.to_parquet(BACKUP_PATH, index=False)
        log.info(f"Backed up original to {BACKUP_PATH}")
    else:
        log.info(f"Backup already exists at {BACKUP_PATH} (kept untouched)")

    df = df.copy()
    df["qa_type"] = df.apply(
        lambda r: _classify_qa_type(r.get("query"), r.get("hop_type")), axis=1
    )
    df["metadata"] = df["query"].apply(_build_metadata)

    type_counts = df["qa_type"].value_counts().to_dict()
    log.info(f"Positive qa_type distribution: {type_counts}")

    neg_df = _build_negative_rows()
    log.info(f"Built {len(neg_df)} negative rows")

    combined = pd.concat([df, neg_df], ignore_index=True)
    log.info(
        f"Final dataset: rows={len(combined)}, "
        f"qa_type={combined['qa_type'].value_counts().to_dict()}"
    )

    combined.to_parquet(QA_PATH, index=False)
    log.info(f"Wrote augmented qa.parquet to {QA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
