"""Self-Corrective RAG pipeline (hybrid -> rerank -> generate -> verify -> retry).

Groundedness relaxation:
  Multi-hop and date-arithmetic queries often combine facts that aren't all
  literally present in any single retrieved chunk; the judge then returns
  ``notSure`` even though the answer is correctly grounded in aggregate. For
  these query types we accept ``notSure`` instead of forcing a HyDE retry.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from src.config import settings
from src.generation.citation import ensure_citation
from src.generation.groundedness import GroundednessChecker
from src.generation.intent_classifier import IntentClassifier
from src.generation.prompts import annotate_inferred_campus, format_context
from src.generation.solar_llm import SolarLLM
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import KoReranker, PassthroughReranker
from src.retrieval.router import RoutingDecision, route
from src.utils.logger import get_logger

log = get_logger(__name__)


FALLBACK_ANSWER = "제공된 자료에서 해당 정보를 찾을 수 없습니다."

# Multi-hop indicators: queries that aggregate, compare, or list across docs.
_MULTI_HOP_PATTERNS = [
    r"비교",
    r"차이",
    r"각각",
    r"모두",
    r"전부",
    r"몇\s*가지",
    r"종류",
    r"리스트",
    r"목록",
    r"그리고\s*또",
    r"및\s*어떤",
]

# Date-arithmetic indicators: queries that need calendar reasoning.
_DATE_ARITHMETIC_PATTERNS = [
    r"\d+\s*(?:일|주|개월|달|년)\s*(?:전|후|이내|동안)",
    r"기간",
    r"며칠",
    r"몇\s*(?:일|주|달|개월|년)",
    r"언제까지",
    r"마감",
    r"D-\d+",
]

_MULTI_HOP_RE = re.compile("|".join(_MULTI_HOP_PATTERNS))
_DATE_ARITHMETIC_RE = re.compile("|".join(_DATE_ARITHMETIC_PATTERNS))


def _is_relaxable(query: str) -> bool:
    return bool(_MULTI_HOP_RE.search(query) or _DATE_ARITHMETIC_RE.search(query))


def _sources(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in candidates:
        payload = c.get("payload") or {}
        out.append(
            {
                "doc_id": c.get("doc_id"),
                "category": payload.get("category"),
                "subcategory": payload.get("subcategory"),
                "campus": payload.get("campus"),
                "source_collection": payload.get("source_collection"),
                "title": payload.get("title"),
                "rerank_score": c.get("rerank_score"),
                "rrf_score": c.get("rrf_score"),
            }
        )
    return out


def _contexts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Eval-friendly view: chunk text + metadata for RAGAS and validation scripts."""
    out: list[dict[str, Any]] = []
    for c in candidates:
        payload = c.get("payload") or {}
        contents = c.get("contents") or payload.get("contents") or ""
        out.append(
            {
                "doc_id": c.get("doc_id"),
                "contents": contents,
                "metadata": {
                    "campus": payload.get("campus"),
                    "category": payload.get("category"),
                    "source_collection": payload.get("source_collection"),
                    "title": payload.get("title"),
                },
            }
        )
    return out


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: Any = None,
        llm: SolarLLM | None = None,
        groundedness: GroundednessChecker | None = None,
        intent: IntentClassifier | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        if reranker is None:
            self.reranker = (
                KoReranker() if settings.reranker_enabled else PassthroughReranker()
            )
            log.info(
                f"reranker: {'KoReranker' if settings.reranker_enabled else 'Passthrough'}"
            )
        else:
            self.reranker = reranker
        self.llm = llm or SolarLLM()
        self.groundedness = groundedness or GroundednessChecker()
        self.intent = intent or IntentClassifier()

    async def _retrieve_then_rerank(
        self,
        query: str,
        hybrid_top_k: int,
        rerank_top_k: int,
        decision: RoutingDecision | None = None,
    ) -> tuple[list[dict[str, Any]], RoutingDecision]:
        # Caller may pass an existing decision (e.g., HyDE retry preserving the
        # original campus filter); only re-route when none is provided.
        decision = decision or route(query)
        candidates = await self.retriever.search(
            query, top_k=hybrid_top_k, decision=decision
        )
        candidates = await asyncio.to_thread(
            self.reranker.rerank, query, candidates, rerank_top_k
        )
        return candidates, decision

    async def run(self, query: str) -> dict[str, Any]:
        t0 = time.time()
        log.info(f"pipeline.run: {query!r}")

        # Intent gate: filter clearly-out-of-scope queries before retrieval.
        # Catches private info, ambiguous junk, external services, false-premise
        # questions — recovers negative_rejection which the relaxed groundedness
        # judge no longer guards against.
        if not await self.intent.is_answerable(query):
            log.info("intent=unanswerable -> skip retrieval, return fallback")
            return {
                "answer": FALLBACK_ANSWER,
                "grounded": False,
                "verdict": "unanswerable",
                "sources": [],
                "contexts": [],
                "retry": False,
                "resolved_campus": None,
                "campus_was_inferred": False,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

        relaxable = _is_relaxable(query)

        candidates, decision = await self._retrieve_then_rerank(
            query,
            hybrid_top_k=settings.top_k_dense,
            rerank_top_k=settings.top_k_rerank_final,
        )
        answer = await self.llm.generate(query, candidates)
        answer = ensure_citation(answer, candidates)
        verdict = await self.groundedness.verify(format_context(candidates), answer)

        retry = False
        # Multi-hop / date-arithmetic queries legitimately produce notSure
        # because the judge can't see the aggregation; only retry on hard fail.
        if verdict == "notGrounded" or (verdict == "notSure" and not relaxable):
            retry = True
            log.warning(
                f"verdict={verdict} relaxable={relaxable} -> HyDE retry"
            )
            hyde_doc = await self.llm.hyde_expand(query)
            expanded_query = f"{query}\n\n{hyde_doc}"
            # Reuse the original RoutingDecision so the campus filter survives
            # the HyDE expansion (the expanded text can dilute campus signals).
            candidates, decision = await self._retrieve_then_rerank(
                expanded_query,
                hybrid_top_k=settings.top_k_dense + 20,
                rerank_top_k=settings.top_k_rerank_retry,
                decision=decision,
            )
            answer = await self.llm.generate(query, candidates)
            answer = ensure_citation(answer, candidates)
            verdict = await self.groundedness.verify(format_context(candidates), answer)

            if verdict == "notGrounded":
                # Preserve retrieved contexts/sources so eval signals (routing,
                # citation, campus_filter) reflect *retrieval* quality, not the
                # fallback step. The user-facing answer stays as the safe
                # FALLBACK_ANSWER and grounded=False keeps negative-rejection
                # accounting intact.
                log.warning(
                    "retry still notGrounded -> fallback "
                    f"(contexts={len(candidates)} preserved for eval signal)"
                )
                return {
                    "answer": FALLBACK_ANSWER,
                    "grounded": False,
                    "verdict": verdict,
                    "sources": _sources(candidates),
                    "contexts": _contexts(candidates),
                    "retry": True,
                    "resolved_campus": decision.campus,
                    "campus_was_inferred": decision.campus_was_inferred,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                }

        # `grounded` for downstream eval: relaxable+notSure also passes.
        is_grounded = verdict == "grounded" or (verdict == "notSure" and relaxable)

        if decision.campus_was_inferred and decision.campus:
            answer = annotate_inferred_campus(answer, decision.campus)

        return {
            "answer": answer,
            "grounded": is_grounded,
            "verdict": verdict,
            "sources": _sources(candidates),
            "contexts": _contexts(candidates),
            "retry": retry,
            "resolved_campus": decision.campus,
            "campus_was_inferred": decision.campus_was_inferred,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
