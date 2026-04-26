"""Index `data/corpus.parquet` into Qdrant + BM25.

Usage:
    python scripts/index_corpus.py            # incremental upsert
    python scripts/index_corpus.py --recreate # drop + rebuild collection
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.embeddings.solar_embedder import SolarEmbedder
from src.retrieval.bm25_okt import OktBM25
from src.retrieval.qdrant_store import QdrantStore
from src.utils.logger import get_logger

log = get_logger("index_corpus")


SAMPLE_QUERIES = [
    "수강신청 언제 시작해요?",
    "성남캠퍼스 장학금 종류 알려줘",
    "의정부캠퍼스 도서관 전화번호",
    "졸업요건 학점 몇 학점이야?",
    "휴학 신청 방법",
]


def _normalize_payload(metadata: dict) -> dict:
    payload = dict(metadata) if metadata else {}
    for key in ("start_date", "end_date"):
        v = payload.get(key)
        if v in ("", "nan", "NaT"):
            payload[key] = None
    return payload


def load_corpus() -> pd.DataFrame:
    path = settings.corpus_path
    if not path.exists():
        raise FileNotFoundError(f"corpus not found at {path}")
    df = pd.read_parquet(path)
    log.info(f"loaded corpus: {len(df)} rows from {path}")
    required = {"doc_id", "contents", "metadata"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"corpus missing columns: {missing}")
    return df


def index_dense(df: pd.DataFrame, recreate: bool) -> int:
    store = QdrantStore()
    store.ensure_collection(recreate=recreate)

    embedder = SolarEmbedder(mode="passage")
    contents = df["contents"].tolist()
    doc_ids = df["doc_id"].tolist()
    payloads = [_normalize_payload(m) for m in df["metadata"].tolist()]
    for p, c in zip(payloads, contents):
        p["contents"] = c

    log.info(f"embedding {len(contents)} chunks via Solar Passage (batch={settings.embed_batch_size})")
    t0 = time.time()
    vectors = embedder.embed_batched(contents, batch_size=settings.embed_batch_size, progress=True)
    log.info(f"embedding done in {time.time() - t0:.1f}s")

    n = store.upsert(doc_ids, vectors, payloads)
    count = store.count()
    log.info(f"Qdrant indexed: upserted={n} total_count={count}")
    return count


def index_sparse(df: pd.DataFrame) -> Path:
    bm25 = OktBM25()
    payloads = [_normalize_payload(m) for m in df["metadata"].tolist()]
    contents = df["contents"].tolist()
    for p, c in zip(payloads, contents):
        p["contents"] = c
    bm25.build(df["doc_id"].tolist(), contents, payloads=payloads)
    return bm25.save()


def verify(samples: list[str]) -> None:
    log.info("=" * 60)
    log.info("verification: dense search with sample queries")
    store = QdrantStore()
    embedder = SolarEmbedder(mode="query")
    for q in samples:
        vec = embedder.embed([q])[0]
        hits = store.search(vec, top_k=3)
        log.info(f"\nQ: {q}")
        for i, h in enumerate(hits, 1):
            payload = h["payload"]
            preview = (payload.get("contents") or "")[:80].replace("\n", " ")
            log.info(
                f"  [{i}] score={h['score']:.4f} "
                f"src={payload.get('source_collection')} "
                f"id={h['doc_id']} | {preview}..."
            )

    log.info("\nverification: sparse search with sample queries")
    bm25 = OktBM25()
    bm25.load()
    for q in samples:
        hits = bm25.search(q, top_k=3)
        log.info(f"\nQ: {q}")
        for i, (doc_id, score) in enumerate(hits, 1):
            log.info(f"  [{i}] bm25={score:.4f} id={doc_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Index corpus into Qdrant + BM25")
    parser.add_argument("--recreate", action="store_true", help="drop and recreate Qdrant collection")
    parser.add_argument("--skip-dense", action="store_true", help="skip Qdrant indexing")
    parser.add_argument("--skip-sparse", action="store_true", help="skip BM25 indexing")
    parser.add_argument("--skip-verify", action="store_true", help="skip sample query verification")
    args = parser.parse_args()

    df = load_corpus()

    if not args.skip_dense:
        count = index_dense(df, recreate=args.recreate)
        if count != len(df):
            log.warning(f"Qdrant count {count} != corpus rows {len(df)}")

    if not args.skip_sparse:
        index_sparse(df)

    if not args.skip_verify:
        verify(SAMPLE_QUERIES)

    log.info("indexing complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
