"""Regression: notGrounded fallback must preserve retrieved sources/contexts.

Phase 5 regression (eval_supplementary 2026-04-26): the previous fallback
path zeroed ``sources`` and ``contexts`` when the verifier rejected both the
initial and HyDE-retried answers. That destroyed the eval signal — routing,
campus_filter, and citation accuracy collapsed to ~0.48 even though retrieval
was still finding the right chunks (verified via reports/diagnose_retrieval.json).

This test pins the contract: even when the user-facing answer is the
fallback string, the result dict must keep the candidates so eval scripts
(and downstream observability) can see *what* was retrieved.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.pipeline.rag_pipeline import FALLBACK_ANSWER, RagPipeline
from src.retrieval.router import RoutingDecision


class _StubRetriever:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self._candidates = candidates

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        decision: RoutingDecision | None = None,
    ) -> list[dict[str, Any]]:
        return list(self._candidates)


class _StubReranker:
    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        return candidates[:top_k]


class _StubLLM:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.hyde_calls = 0

    async def generate(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> str:
        self.generate_calls += 1
        return "초기 답변입니다."

    async def hyde_expand(self, query: str) -> str:
        self.hyde_calls += 1
        return "HyDE-expanded hypothetical document."


class _AlwaysNotGrounded:
    def __init__(self) -> None:
        self.calls = 0

    async def verify(self, context: str, answer: str) -> str:
        self.calls += 1
        return "notGrounded"


def _candidate(doc_id: str, *, campus: str = "성남", col: str = "학칙_조항") -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "contents": f"본문-{doc_id}",
        "payload": {
            "doc_id": doc_id,
            "campus": campus,
            "category": "학칙",
            "source_collection": col,
            "title": f"제목-{doc_id}",
            "contents": f"본문-{doc_id}",
        },
        "rerank_score": 0.9,
        "rrf_score": 0.5,
    }


@pytest.mark.asyncio
async def test_fallback_preserves_sources_and_contexts() -> None:
    candidates = [_candidate("학칙_1"), _candidate("학칙_2"), _candidate("학칙_3")]
    pipeline = RagPipeline(
        retriever=_StubRetriever(candidates),
        reranker=_StubReranker(),
        llm=_StubLLM(),
        groundedness=_AlwaysNotGrounded(),
    )

    result = await pipeline.run("학사경고 제적 후 수료 요건은 무엇인가요?")

    # The user-facing answer keeps the fallback wording so negative_rejection
    # still flags it, but a [출처: ...] citation is now appended (sourced from
    # the retrieved candidates) so the citation eval can match.
    assert result["answer"].startswith(FALLBACK_ANSWER)
    assert "[출처:" in result["answer"]
    assert result["grounded"] is False
    assert result["verdict"] == "notGrounded"
    assert result["retry"] is True
    assert len(result["sources"]) == len(candidates), (
        "fallback must keep retrieved sources for eval signal — "
        "see reports/diagnose_retrieval.json"
    )
    assert len(result["contexts"]) == len(candidates)
    assert {s["doc_id"] for s in result["sources"]} == {
        "학칙_1",
        "학칙_2",
        "학칙_3",
    }
    assert all(c["metadata"]["campus"] == "성남" for c in result["contexts"])
    assert all(
        c["metadata"]["source_collection"] == "학칙_조항"
        for c in result["contexts"]
    )


@pytest.mark.asyncio
async def test_fallback_invokes_hyde_retry_once() -> None:
    candidates = [_candidate("학칙_1")]
    llm = _StubLLM()
    verifier = _AlwaysNotGrounded()
    pipeline = RagPipeline(
        retriever=_StubRetriever(candidates),
        reranker=_StubReranker(),
        llm=llm,
        groundedness=verifier,
    )

    await pipeline.run("졸업학점 기준 알려줘")

    assert llm.hyde_calls == 1, "HyDE retry must fire exactly once on notGrounded"
    assert llm.generate_calls == 2, "generate runs once initially and once after HyDE"
    assert verifier.calls == 2, "verifier runs after both initial and retry"


@pytest.mark.asyncio
async def test_fallback_keeps_resolved_campus_metadata() -> None:
    candidates = [_candidate("학칙_1", campus="전체")]
    pipeline = RagPipeline(
        retriever=_StubRetriever(candidates),
        reranker=_StubReranker(),
        llm=_StubLLM(),
        groundedness=_AlwaysNotGrounded(),
    )

    result = await pipeline.run("학사경고 졸업요건 알려줘")

    assert result["resolved_campus"] is not None
    assert "campus_was_inferred" in result
    assert result["contexts"][0]["metadata"]["campus"] == "전체"
