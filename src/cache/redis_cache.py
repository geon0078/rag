"""Redis-backed semantic answer cache.

Strategy:
- Cache key for one entry: ``cache:entry:{nanoid}`` -> JSON {answer, sources, ttl_class, embedding}.
- Recency index: ``cache:index`` is a Redis LIST of entry keys (LPUSH; trimmed to ``index_max``).
- Lookup: query embedded by Solar query model, then we LRANGE the index, fetch each entry's
  embedding via MGET, score by cosine similarity, and return the best entry if score >=
  ``settings.cache_similarity_threshold``.
- TTL: 24h default (``cache_ttl_default_sec``); 6h for entries marked as 학사일정-class
  (``cache_ttl_calendar_sec``). The class is decided by the caller via ``ttl_class``.

This sidesteps Redis Stack/RediSearch vector indexing dependency while still meeting the
0.95 similarity gate from the spec; for ~500-entry windows the scan cost is negligible
relative to the LLM call we're trying to skip.
"""

from __future__ import annotations

import asyncio
import json
import math
import secrets
from typing import Any

import redis.asyncio as aioredis

from src.config import settings
from src.embeddings.solar_embedder import SolarEmbedder
from src.utils.logger import get_logger

log = get_logger(__name__)

INDEX_KEY = "cache:index"
ENTRY_PREFIX = "cache:entry:"
TTL_CLASS_CALENDAR = "calendar"
TTL_CLASS_DEFAULT = "default"


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class SemanticCache:
    def __init__(
        self,
        embedder: SolarEmbedder | None = None,
        redis_client: aioredis.Redis | None = None,
        threshold: float | None = None,
        index_max: int = 500,
    ) -> None:
        self.embedder = embedder or SolarEmbedder(mode="query")
        self._redis = redis_client
        self.threshold = (
            threshold if threshold is not None else settings.cache_similarity_threshold
        )
        self.index_max = index_max

    async def _client(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def ping(self) -> bool:
        try:
            client = await self._client()
            return bool(await client.ping())
        except Exception as exc:
            log.warning(f"redis ping failed: {exc}")
            return False

    @staticmethod
    def _ttl_for_class(ttl_class: str) -> int:
        if ttl_class == TTL_CLASS_CALENDAR:
            return settings.cache_ttl_calendar_sec
        return settings.cache_ttl_default_sec

    async def _embed_query(self, query: str) -> list[float]:
        return (await asyncio.to_thread(self.embedder.embed, [query]))[0]

    async def lookup(self, query: str) -> dict[str, Any] | None:
        """Return cached payload if a near-duplicate query exists, else None."""
        try:
            client = await self._client()
            entry_keys = await client.lrange(INDEX_KEY, 0, self.index_max - 1)
            if not entry_keys:
                return None

            qvec = await self._embed_query(query)
            raw_entries = await client.mget(entry_keys)

            best_score = 0.0
            best_payload: dict[str, Any] | None = None
            stale_keys: list[str] = []
            for key, raw in zip(entry_keys, raw_entries):
                if raw is None:
                    stale_keys.append(key)
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    stale_keys.append(key)
                    continue
                emb = entry.get("embedding")
                if not emb:
                    continue
                score = _cosine(qvec, emb)
                if score > best_score:
                    best_score = score
                    best_payload = entry

            if stale_keys:
                pipe = client.pipeline()
                for key in stale_keys:
                    pipe.lrem(INDEX_KEY, 0, key)
                await pipe.execute()

            if best_payload is None or best_score < self.threshold:
                log.debug(
                    f"cache miss: best={best_score:.3f} threshold={self.threshold}"
                )
                return None

            log.info(f"cache hit: similarity={best_score:.3f}")
            return {
                "answer": best_payload.get("answer"),
                "sources": best_payload.get("sources", []),
                "grounded": best_payload.get("grounded", True),
                "verdict": best_payload.get("verdict", "grounded"),
                "similarity": best_score,
                "cached": True,
            }
        except Exception as exc:
            log.warning(f"cache lookup failed: {exc}")
            return None

    async def store(
        self,
        query: str,
        payload: dict[str, Any],
        ttl_class: str = TTL_CLASS_DEFAULT,
    ) -> None:
        """Store an answer keyed by its query embedding."""
        try:
            client = await self._client()
            qvec = await self._embed_query(query)
            entry = {
                "query": query,
                "answer": payload.get("answer"),
                "sources": payload.get("sources", []),
                "grounded": payload.get("grounded", True),
                "verdict": payload.get("verdict"),
                "ttl_class": ttl_class,
                "embedding": qvec,
            }
            entry_key = f"{ENTRY_PREFIX}{secrets.token_hex(8)}"
            ttl = self._ttl_for_class(ttl_class)

            pipe = client.pipeline()
            pipe.set(entry_key, json.dumps(entry, ensure_ascii=False), ex=ttl)
            pipe.lpush(INDEX_KEY, entry_key)
            pipe.ltrim(INDEX_KEY, 0, self.index_max - 1)
            await pipe.execute()

            log.debug(f"cache stored: ttl_class={ttl_class} ttl={ttl}s")
        except Exception as exc:
            log.warning(f"cache store failed: {exc}")

    async def stats(self) -> dict[str, Any]:
        try:
            client = await self._client()
            size = await client.llen(INDEX_KEY)
            return {"connected": True, "index_size": int(size)}
        except Exception as exc:
            return {"connected": False, "error": str(exc)}
