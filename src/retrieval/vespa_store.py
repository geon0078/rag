"""Vespa retriever — onyx+docmost 개발.md 시나리오 A 풀 교체.

Solar query embedding 으로 Vespa nearestNeighbor + BM25 hybrid 검색.
4개 ranking profile (bm25_only / vector_only / hybrid_cc / rrf_approx) 중
``rank_profile`` 인자로 선택.

Schema: vespa/schemas/eulji_chunk.sd 와 일치.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.embeddings.solar_embedder import SolarEmbedder
from src.utils.logger import get_logger

log = get_logger(__name__)


class VespaStore:
    def __init__(
        self,
        url: str | None = None,
        rank_profile: str = "hybrid_cc",
        cc_weight: float = 0.6,
        query_embedder: SolarEmbedder | None = None,
    ) -> None:
        self.url = (url or os.environ.get("VESPA_URL", "http://localhost:8080")).rstrip("/")
        self.rank_profile = rank_profile
        self.cc_weight = cc_weight
        self.query_embedder = query_embedder or SolarEmbedder(mode="query")
        self.client = httpx.Client(timeout=30.0)

    def search(
        self,
        query: str,
        top_k: int = 30,
        rank_profile: str | None = None,
        cc_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        profile = rank_profile or self.rank_profile
        w = cc_weight if cc_weight is not None else self.cc_weight

        body: dict[str, Any] = {
            "yql": f"select * from eulji_chunk where userQuery() or ({{targetHits:{top_k}}}nearestNeighbor(embedding,q_emb))",
            "query": query,
            "hits": top_k,
            "ranking.profile": profile,
        }

        if profile in ("vector_only", "hybrid_cc", "rrf_approx"):
            vec = self.query_embedder.embed([query])[0]
            body["input.query(q_emb)"] = vec
        if profile == "hybrid_cc":
            body["input.query(w)"] = w

        r = self.client.post(f"{self.url}/search/", json=body)
        if r.status_code != 200:
            log.warning(f"vespa query fail {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        children = (data.get("root") or {}).get("children", []) or []

        out: list[dict[str, Any]] = []
        for c in children:
            fields = c.get("fields") or {}
            out.append(
                {
                    "doc_id": fields.get("doc_id") or c.get("id", "").split("::")[-1],
                    "contents": fields.get("contents", ""),
                    "score": float(c.get("relevance", 0.0)),
                    "payload": {
                        "doc_id": fields.get("doc_id"),
                        "title": fields.get("title"),
                        "campus": fields.get("campus"),
                        "source_collection": fields.get("source_collection"),
                        "category": fields.get("source_collection"),
                        "path": fields.get("path"),
                        "contents": fields.get("contents", ""),
                    },
                }
            )
        log.info(f"vespa[{profile} w={w}]: {len(out)} hits for {query[:40]!r}")
        return out
