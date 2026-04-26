"""Sparse-filter contract: BM25 metadata_filter must drop off-campus docs.

Targets the bug captured in `reports/eval_campus_filter.json` where sparse-only
candidates fused into the final result with payload={} bypassed the campus
filter and pulled in 의정부 docs for 성남 queries.
"""

from __future__ import annotations

import pytest

from src.retrieval.bm25_okt import OktBM25, _passes_filter


def test_passes_filter_no_filter_returns_true() -> None:
    assert _passes_filter({"campus": "성남"}, None) is True
    assert _passes_filter({}, None) is True


def test_passes_filter_scalar_match() -> None:
    assert _passes_filter({"campus": "성남"}, {"campus": "성남"}) is True
    assert _passes_filter({"campus": "의정부"}, {"campus": "성남"}) is False


def test_passes_filter_list_match() -> None:
    f = {"campus": ["성남", "전체"]}
    assert _passes_filter({"campus": "성남"}, f) is True
    assert _passes_filter({"campus": "전체"}, f) is True
    assert _passes_filter({"campus": "의정부"}, f) is False


def test_passes_filter_missing_field_fails() -> None:
    assert _passes_filter({}, {"campus": "성남"}) is False


def _toy_bm25() -> OktBM25:
    bm25 = OktBM25()
    bm25.build(
        doc_ids=["a", "b", "c"],
        contents=["성남캠퍼스 장학금 안내", "의정부 도서관 운영", "전체 학칙 조항"],
        payloads=[
            {"campus": "성남", "category": "장학금"},
            {"campus": "의정부", "category": "시설"},
            {"campus": "전체", "category": "학칙"},
        ],
    )
    return bm25


def test_bm25_filter_drops_other_campus() -> None:
    bm25 = _toy_bm25()
    hits = bm25.search("장학금", top_k=5, metadata_filter={"campus": ["성남", "전체"]})
    ids = [d for d, _ in hits]
    assert "b" not in ids
    assert "a" in ids


def test_bm25_unfiltered_returns_all_matches() -> None:
    bm25 = _toy_bm25()
    hits = bm25.search("도서관 운영", top_k=5)
    ids = [d for d, _ in hits]
    assert "b" in ids


def test_bm25_warns_when_filter_requested_without_payloads() -> None:
    bm25 = OktBM25()
    bm25.build(["a"], ["test content"])
    bm25.search("test", metadata_filter={"campus": "성남"})
    assert bm25.payload_lookup == {}


def test_bm25_pickle_roundtrip_preserves_payloads(tmp_path) -> None:
    bm25 = _toy_bm25()
    out = tmp_path / "bm25.pkl"
    bm25.save(out)

    loaded = OktBM25()
    loaded.load(out)

    assert loaded.doc_ids == bm25.doc_ids
    assert loaded.payload_lookup == bm25.payload_lookup
    assert loaded.get_payload("a")["campus"] == "성남"
