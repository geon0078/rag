"""Vespa 인덱서 — Qdrant 의 기존 Solar 임베딩 재사용 (시나리오 A).

Solar API 재호출 없이 Qdrant 컬렉션 `euljiu_knowledge` 에서 (payload, vector)
2,382개 scroll 로 가져와 Vespa `/document/v1/default/eulji_chunk/docid/{doc_id}`
로 feed.

Run:
    python scripts/index_vespa.py [--limit 0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("index_vespa")


VESPA_URL = os.environ.get("VESPA_URL", "http://localhost:8080")
DOC_API = f"{VESPA_URL}/document/v1/default/eulji_chunk/docid"


def _build_doc(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    vec = list(point.vector) if point.vector else []
    # 실제 청크 doc_id 는 payload.doc_id (예: 'lec_lecture_reviews_32_c26').
    # Qdrant 의 point.id 는 내부 UUID — eval expected_doc_ids 와 매칭 안 됨.
    real_doc_id = str(payload.get("doc_id") or point.id)
    return {
        "fields": {
            "doc_id": real_doc_id,
            "parent_doc_id": str(payload.get("parent_doc_id") or ""),
            "source_collection": str(payload.get("source_collection") or ""),
            "campus": str(payload.get("campus") or "전체"),
            "path": str(payload.get("path") or ""),
            "title": str(payload.get("title") or ""),
            "contents": str(payload.get("contents") or ""),
            "embedding": {"values": vec},
            "metadata_json": json.dumps(payload, ensure_ascii=False, default=str)[:8000],
            "status": str(payload.get("status") or "Indexed"),
        }
    }


async def _feed_one(client: httpx.AsyncClient, doc_id: str, doc: dict) -> bool:
    # Vespa docid 는 URL-safe 해야. 슬래시/콜론 등 escape.
    import urllib.parse
    safe_id = urllib.parse.quote(doc_id, safe="")
    url = f"{DOC_API}/{safe_id}"
    r = await client.post(url, json=doc)
    if r.status_code in (200, 201):
        return True
    if r.status_code == 400 and "already exists" in r.text:
        return True
    log.warning(f"feed fail {doc_id}: {r.status_code} {r.text[:200]}")
    return False


async def main_async(args: argparse.Namespace) -> int:
    from qdrant_client import QdrantClient  # noqa: E402

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    log.info(f"scrolling Qdrant collection {settings.qdrant_collection!r}")

    points: list[Any] = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=200,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if next_offset is None:
            break
        if args.limit and len(points) >= args.limit:
            points = points[: args.limit]
            break
    log.info(f"fetched {len(points)} points from Qdrant")

    fed = 0
    failed = 0
    async with httpx.AsyncClient(timeout=30.0) as ac:
        sem = asyncio.Semaphore(8)

        async def _go(point: Any) -> None:
            nonlocal fed, failed
            async with sem:
                doc = _build_doc(point)
                ok = await _feed_one(ac, str(point.id), doc)
                if ok:
                    fed += 1
                else:
                    failed += 1
                if (fed + failed) % 200 == 0:
                    log.info(f"  progress: fed={fed} failed={failed}")

        await asyncio.gather(*(_go(p) for p in points))

    log.info(f"done: fed={fed} failed={failed}")
    return 0 if failed == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
