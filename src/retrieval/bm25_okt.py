"""BM25 sparse retrieval with Okt Korean morphological tokenizer.

Stores a per-doc payload lookup so callers can apply metadata filters
(e.g., campus) at search time, mirroring what Qdrant does for dense.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

from konlpy.tag import Okt
from rank_bm25 import BM25Okapi

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def _passes_filter(payload: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    for key, allowed in metadata_filter.items():
        actual = payload.get(key)
        if isinstance(allowed, (list, tuple, set)):
            if actual not in allowed:
                return False
        else:
            if actual != allowed:
                return False
    return True


class OktBM25:
    def __init__(self) -> None:
        self.okt = Okt()
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self.payload_lookup: dict[str, dict[str, Any]] = {}

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in self.okt.morphs(text, stem=True) if t.strip()]

    def build(
        self,
        doc_ids: Sequence[str],
        contents: Sequence[str],
        payloads: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if len(doc_ids) != len(contents):
            raise ValueError(
                f"length mismatch: doc_ids={len(doc_ids)} contents={len(contents)}"
            )
        if payloads is not None and len(payloads) != len(doc_ids):
            raise ValueError(
                f"payloads length {len(payloads)} != doc_ids {len(doc_ids)}"
            )
        log.info(f"tokenizing {len(contents)} documents with Okt (stem=True)")
        self.doc_ids = list(doc_ids)
        self.corpus_tokens = [self._tokenize(c) for c in contents]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        if payloads is None:
            self.payload_lookup = {}
        else:
            self.payload_lookup = {
                doc_id: dict(p) for doc_id, p in zip(self.doc_ids, payloads)
            }
        log.info(
            f"BM25 index built: {len(self.doc_ids)} docs "
            f"(payload_lookup={'on' if self.payload_lookup else 'off'})"
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built — call build() or load() first")
        k = top_k or settings.top_k_sparse
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)

        if metadata_filter and not self.payload_lookup:
            log.warning(
                "metadata_filter requested but BM25 has no payload_lookup; "
                "rebuild the index via scripts/index_corpus.py to enable filtering"
            )
            metadata_filter = None

        pairs: list[tuple[str, float]] = []
        for doc_id, score in zip(self.doc_ids, scores):
            if metadata_filter:
                payload = self.payload_lookup.get(doc_id, {})
                if not _passes_filter(payload, metadata_filter):
                    continue
            pairs.append((doc_id, float(score)))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]

    def get_payload(self, doc_id: str) -> dict[str, Any]:
        return self.payload_lookup.get(doc_id, {})

    def save(self, path: Path | None = None) -> Path:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built — nothing to save")
        out = path or settings.bm25_index_path
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_ids": self.doc_ids,
            "corpus_tokens": self.corpus_tokens,
            "bm25": self.bm25,
            "payload_lookup": self.payload_lookup,
            "schema_version": 2,
        }
        with open(out, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"BM25 index saved to {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
        return out

    def load(self, path: Path | None = None) -> None:
        src = path or settings.bm25_index_path
        with open(src, "rb") as f:
            payload = pickle.load(f)
        self.doc_ids = payload["doc_ids"]
        self.corpus_tokens = payload["corpus_tokens"]
        self.bm25 = payload["bm25"]
        self.payload_lookup = payload.get("payload_lookup", {})
        version = payload.get("schema_version", 1)
        log.info(
            f"BM25 index loaded from {src}: {len(self.doc_ids)} docs "
            f"(schema_v{version}, payloads={len(self.payload_lookup)})"
        )
