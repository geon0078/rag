"""Default-campus policy contract.

Targets the 6 campus_filter eval failures (`reports/eval_campus_filter.json`)
where queries with no explicit campus signal fell through to either
``got=null`` (sparse-only no-payload docs — fixed by `test_sparse_filter`) or
``got="no contexts"`` (router returned campus=None → no filter → reranked set
became empty for downstream filtering). The default-campus policy resolves
both: queries without a campus keyword now resolve to ``settings.default_campus``
(성남) so the BM25 + Qdrant filter is always applied.
"""

from __future__ import annotations

from src.config import settings
from src.generation.prompts import INFERRED_CAMPUS_NOTICE, annotate_inferred_campus
from src.retrieval.router import (
    build_metadata_filter,
    extract_campus,
    resolve_campus,
    route,
)


def test_extract_campus_returns_none_when_absent() -> None:
    assert extract_campus("졸업요건이 뭐야?") is None
    assert extract_campus("총장 이름 알려줘") is None


def test_extract_campus_detects_explicit_keyword() -> None:
    assert extract_campus("성남캠퍼스 도서관 위치 알려줘") == "성남"
    assert extract_campus("의정부 셔틀버스 시간표") == "의정부"


def test_resolve_campus_uses_default_when_no_signal() -> None:
    campus, was_inferred = resolve_campus("졸업요건이 뭐야?")
    assert campus == settings.default_campus
    assert was_inferred is True


def test_resolve_campus_keeps_explicit_signal() -> None:
    campus, was_inferred = resolve_campus("의정부캠퍼스 도서관 운영시간")
    assert campus == "의정부"
    assert was_inferred is False


def test_route_populates_inferred_flag_and_filter() -> None:
    decision = route("학칙 졸업요건 알려줘")
    assert decision.campus == settings.default_campus
    assert decision.campus_was_inferred is True
    assert decision.sparse_filter == {
        "campus": [settings.default_campus, "전체"]
    }
    assert decision.qdrant_filter is not None


def test_route_does_not_mark_explicit_campus_as_inferred() -> None:
    decision = route("성남캠퍼스 학식 메뉴 알려줘")
    assert decision.campus == "성남"
    assert decision.campus_was_inferred is False


def test_build_metadata_filter_returns_dual_value_filter() -> None:
    assert build_metadata_filter("성남") == {"campus": ["성남", "전체"]}
    assert build_metadata_filter("의정부") == {"campus": ["의정부", "전체"]}


def test_build_metadata_filter_passes_through_none() -> None:
    assert build_metadata_filter(None) is None


def test_annotate_inferred_campus_prepends_notice() -> None:
    out = annotate_inferred_campus("졸업요건은 130학점입니다.", "성남")
    assert out.startswith(INFERRED_CAMPUS_NOTICE.format(campus="성남"))
    assert "졸업요건은 130학점입니다." in out


def test_annotate_inferred_campus_idempotent() -> None:
    once = annotate_inferred_campus("답변.", "성남")
    twice = annotate_inferred_campus(once, "성남")
    assert once == twice


def test_annotate_inferred_campus_noop_on_empty() -> None:
    assert annotate_inferred_campus("", "성남") == ""
