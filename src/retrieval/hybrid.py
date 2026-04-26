"""Hybrid retrieval: dense (Qdrant) + sparse (BM25 Okt).

Fusion picks per `settings.hybrid_method`:
- "cc" (default, AutoRAG-tuned): min-max normalize both score lists, then
  final = w * dense + (1 - w) * sparse, with w = settings.hybrid_cc_weight.
- "rrf": score(d) = Sigma 1/(k + rank_i(d)), with k = settings.rrf_k.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.config import settings
from src.embeddings.solar_embedder import SolarEmbedder
from src.retrieval.bm25_okt import OktBM25
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.router import RoutingDecision, route
from src.utils.logger import get_logger

log = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        store: QdrantStore | None = None,
        bm25: OktBM25 | None = None,
        query_embedder: SolarEmbedder | None = None,
    ) -> None:
        self.store = store or QdrantStore()
        self.bm25 = bm25 or self._load_bm25()
        self.query_embedder = query_embedder or SolarEmbedder(mode="query")

    @staticmethod
    def _load_bm25() -> OktBM25:
        bm25 = OktBM25()
        bm25.load()
        return bm25

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        decision: RoutingDecision | None = None,
    ) -> list[dict[str, Any]]:
        decision = decision or route(query)
        k_dense = settings.top_k_dense
        k_sparse = settings.top_k_sparse
        k_final = top_k or settings.top_k_dense

        dense_task = asyncio.to_thread(self._dense, query, k_dense, decision)
        sparse_task = asyncio.to_thread(self._sparse, query, k_sparse, decision)
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

        return self._fuse(dense_hits, sparse_hits, decision, k_final)

    def _dense(
        self, query: str, k: int, decision: RoutingDecision
    ) -> list[dict[str, Any]]:
        vec = self.query_embedder.embed([query])[0]
        return self.store.search(vec, top_k=k, query_filter=decision.qdrant_filter)

    def _sparse(
        self, query: str, k: int, decision: RoutingDecision
    ) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k=k, metadata_filter=decision.sparse_filter)

    @staticmethod
    def _mm_normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        lo = min(values.values())
        hi = max(values.values())
        if hi == lo:
            return {k: 1.0 for k in values}
        span = hi - lo
        return {k: (v - lo) / span for k, v in values.items()}

    def _fuse(
        self,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[tuple[str, float]],
        decision: RoutingDecision,
        k_final: int,
    ) -> list[dict[str, Any]]:
        method = settings.hybrid_method
        payloads: dict[str, dict[str, Any]] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            payloads[hit["doc_id"]] = hit["payload"]
            dense_ranks[hit["doc_id"]] = rank
        for rank, (doc_id, _) in enumerate(sparse_hits, start=1):
            sparse_ranks[doc_id] = rank
            if doc_id not in payloads:
                payloads[doc_id] = self.bm25.get_payload(doc_id)

        if method == "cc":
            dense_raw = {h["doc_id"]: float(h.get("score", 0.0)) for h in dense_hits}
            sparse_raw = {doc_id: float(s) for doc_id, s in sparse_hits}
            dense_norm = self._mm_normalize(dense_raw)
            sparse_norm = self._mm_normalize(sparse_raw)
            w = settings.hybrid_cc_weight
            all_ids = set(dense_norm) | set(sparse_norm)
            scores = {
                doc_id: w * dense_norm.get(doc_id, 0.0)
                + (1.0 - w) * sparse_norm.get(doc_id, 0.0)
                for doc_id in all_ids
            }
            score_field = "cc_score"
        else:
            rrf_k = settings.rrf_k
            scores = {}
            for rank, hit in enumerate(dense_hits, start=1):
                doc_id = hit["doc_id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            for rank, (doc_id, _) in enumerate(sparse_hits, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            score_field = "rrf_score"

        if decision.boosts:
            for doc_id, payload in payloads.items():
                col = payload.get("source_collection")
                if col and col in decision.boosts:
                    scores[doc_id] = scores.get(doc_id, 0.0) * decision.boosts[col]

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k_final]

        results: list[dict[str, Any]] = []
        for doc_id, score in ranked:
            payload = payloads.get(doc_id, {})
            results.append(
                {
                    "doc_id": doc_id,
                    "contents": payload.get("contents", ""),
                    "payload": payload,
                    score_field: float(score),
                    "dense_rank": dense_ranks.get(doc_id),
                    "sparse_rank": sparse_ranks.get(doc_id),
                }
            )
        log.info(
            f"hybrid[{method}]: dense={len(dense_hits)} sparse={len(sparse_hits)} "
            f"fused={len(results)} campus={decision.campus} boosts={decision.boosts}"
        )
        return results

    def search_sync(
        self,
        query: str,
        top_k: int | None = None,
        decision: RoutingDecision | None = None,
    ) -> list[dict[str, Any]]:
        return asyncio.run(self.search(query, top_k=top_k, decision=decision))
