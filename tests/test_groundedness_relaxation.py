"""Multi-hop / date-arithmetic detection + citation post-processor contract.

Targets the 19 fallback-cascade queries in `reports/eval_routing.json`
(got_top3=[]) where notSure verdicts on aggregating queries forced a
HyDE fallback even though retrieval was correct.
"""

from __future__ import annotations

from src.generation.citation import ensure_citation, has_citation
from src.pipeline.rag_pipeline import _is_relaxable


def test_is_relaxable_detects_multi_hop_keywords() -> None:
    assert _is_relaxable("성남캠퍼스 장학금 종류 알려줘") is True
    assert _is_relaxable("두 캠퍼스의 차이를 비교해줘") is True
    assert _is_relaxable("필수 과목 모두 알려줘") is True
    assert _is_relaxable("교과목 목록 보여줘") is True


def test_is_relaxable_detects_date_arithmetic() -> None:
    assert _is_relaxable("수강신청 며칠 동안이야?") is True
    assert _is_relaxable("휴학은 7일 전까지 신청해야 해?") is True
    assert _is_relaxable("등록 기간이 언제까지야?") is True
    assert _is_relaxable("D-7 알림이 와?") is True


def test_is_relaxable_returns_false_for_factoid() -> None:
    assert _is_relaxable("수강신청 언제 시작해?") is False
    assert _is_relaxable("의정부 도서관 전화번호 알려줘") is False
    assert _is_relaxable("총장 이름이 뭐야?") is False


def test_has_citation_detects_marker() -> None:
    assert has_citation("답변.\n[출처: x_1, 학칙, 전체]") is True
    assert has_citation("답변.\n[출처:x_1,학칙,전체]") is True
    assert has_citation("답변.") is False
    assert has_citation("") is False


def test_ensure_citation_appends_when_missing() -> None:
    answer = "졸업요건은 130학점입니다."
    candidates = [
        {"doc_id": "학칙_조항_43", "payload": {"category": "학칙", "campus": "전체"}}
    ]
    out = ensure_citation(answer, candidates)
    assert "[출처: 학칙_조항_43, 학칙, 전체]" in out


def test_ensure_citation_idempotent_when_present() -> None:
    answer = "답변입니다.\n[출처: x_1, 학칙, 전체]"
    candidates = [
        {"doc_id": "y_2", "payload": {"category": "FAQ", "campus": "성남"}}
    ]
    assert ensure_citation(answer, candidates) == answer


def test_ensure_citation_noop_when_no_candidates() -> None:
    answer = "정보를 찾을 수 없습니다."
    assert ensure_citation(answer, []) == answer


def test_ensure_citation_noop_on_empty_answer() -> None:
    candidates = [{"doc_id": "x", "payload": {"category": "y", "campus": "전체"}}]
    assert ensure_citation("", candidates) == ""


def test_ensure_citation_falls_back_to_source_collection() -> None:
    answer = "답변입니다."
    candidates = [
        {"doc_id": "a", "payload": {"source_collection": "장학금", "campus": "성남"}}
    ]
    out = ensure_citation(answer, candidates)
    assert "[출처: a, 장학금, 성남]" in out
