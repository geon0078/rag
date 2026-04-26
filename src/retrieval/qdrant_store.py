"""Qdrant vector store for `euljiu_knowledge` collection."""

from __future__ import annotations

from typing import Any, Sequence
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


KEYWORD_INDEX_FIELDS = ("category", "subcategory", "campus", "source_collection")
DATETIME_INDEX_FIELDS = ("start_date", "end_date")


def _doc_id_to_point_id(doc_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, doc_id))


class QdrantStore:
    def __init__(self, collection: str | None = None) -> None:
        self.collection = collection or settings.qdrant_collection
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=False,
            timeout=60.0,
        )

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            log.warning(f"recreating collection {self.collection!r}")
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=settings.embedding_dim,
                    distance=qm.Distance.COSINE,
                ),
            )
            log.info(f"created collection {self.collection!r} dim={settings.embedding_dim}")

        for field in KEYWORD_INDEX_FIELDS:
            self._ensure_payload_index(field, qm.PayloadSchemaType.KEYWORD)
        for field in DATETIME_INDEX_FIELDS:
            self._ensure_payload_index(field, qm.PayloadSchemaType.DATETIME)

    def _ensure_payload_index(self, field: str, schema: qm.PayloadSchemaType) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
            )
            log.info(f"created payload index {field}={schema}")
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "exists" in msg:
                return
            log.warning(f"payload index {field} create skipped: {exc}")

    def upsert(
        self,
        doc_ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
        batch_size: int = 256,
    ) -> int:
        if not (len(doc_ids) == len(vectors) == len(payloads)):
            raise ValueError(
                f"length mismatch: doc_ids={len(doc_ids)} vectors={len(vectors)} payloads={len(payloads)}"
            )
        total = len(doc_ids)
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            points = [
                qm.PointStruct(
                    id=_doc_id_to_point_id(doc_ids[i]),
                    vector=list(vectors[i]),
                    payload={**payloads[i], "doc_id": doc_ids[i]},
                )
                for i in range(start, stop)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            log.info(f"upserted {stop}/{total}")
        return total

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int | None = None,
        query_filter: qm.Filter | None = None,
    ) -> list[dict[str, Any]]:
        k = top_k or settings.top_k_dense
        result = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            {
                "doc_id": h.payload.get("doc_id"),
                "score": float(h.score),
                "payload": dict(h.payload),
            }
            for h in result.points
        ]

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
            log.info(f"deleted collection {self.collection!r}")
