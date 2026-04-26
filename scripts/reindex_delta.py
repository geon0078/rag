"""Incremental reindex: re-embed only chunks whose contents changed.

Compares ``data/corpus.parquet`` against ``data/index_manifest.json`` (sha256 of
contents per doc_id), then:
  - new + changed doc_ids -> Solar Passage embed -> Qdrant upsert
  - deleted doc_ids -> Qdrant delete
  - BM25 always rebuilt (cheap, in-process)

Spec Phase 7 Task 7.3: weekly default; ``--collections 학사일정`` for daily runs.

Usage:
  python scripts/reindex_delta.py
  python scripts/reindex_delta.py --collections 학사일정
  python scripts/reindex_delta.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
from qdrant_client.http import models as qm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.embeddings.solar_embedder import SolarEmbedder
from src.retrieval.bm25_okt import OktBM25
from src.retrieval.qdrant_store import QdrantStore, _doc_id_to_point_id
from src.utils.logger import get_logger

log = get_logger("reindex_delta")

MANIFEST_PATH = settings.data_dir / "index_manifest.json"


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _load_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"manifest saved: {MANIFEST_PATH} ({len(manifest)} doc_ids)")


def _normalize_payload(metadata: dict) -> dict:
    payload = dict(metadata) if metadata else {}
    for key in ("start_date", "end_date"):
        v = payload.get(key)
        if v in ("", "nan", "NaT"):
            payload[key] = None
    return payload


def _filter_collections(df: pd.DataFrame, collections: list[str] | None) -> pd.DataFrame:
    if not collections:
        return df
    mask = df["metadata"].apply(
        lambda m: (m or {}).get("source_collection") in collections
    )
    filtered = df[mask].reset_index(drop=True)
    log.info(f"collection filter {collections}: {len(filtered)}/{len(df)} rows")
    return filtered


def _diff(
    current: dict[str, str],
    previous: dict[str, str],
    scoped_doc_ids: set[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (changed_doc_ids, deleted_doc_ids).

    When scoped_doc_ids is provided (collection filter), deletions are also
    restricted to that scope to avoid wiping out unrelated collections.
    """
    changed = [
        doc_id for doc_id, h in current.items() if previous.get(doc_id) != h
    ]
    if scoped_doc_ids is None:
        deleted = [doc_id for doc_id in previous if doc_id not in current]
    else:
        deleted = [
            doc_id
            for doc_id in previous
            if doc_id in scoped_doc_ids and doc_id not in current
        ]
    return changed, deleted


def _delete_from_qdrant(store: QdrantStore, doc_ids: list[str]) -> None:
    if not doc_ids:
        return
    point_ids = [_doc_id_to_point_id(d) for d in doc_ids]
    store.client.delete(
        collection_name=store.collection,
        points_selector=qm.PointIdsList(points=point_ids),
        wait=True,
    )
    log.info(f"qdrant deleted: {len(doc_ids)} points")


def _upsert_changed(
    store: QdrantStore,
    df: pd.DataFrame,
    changed: list[str],
) -> None:
    if not changed:
        return
    sub = df[df["doc_id"].isin(changed)]
    contents = sub["contents"].tolist()
    doc_ids = sub["doc_id"].tolist()
    payloads = [_normalize_payload(m) for m in sub["metadata"].tolist()]
    for p, c in zip(payloads, contents):
        p["contents"] = c

    embedder = SolarEmbedder(mode="passage")
    log.info(f"embedding {len(contents)} changed chunks")
    t0 = time.time()
    vectors = embedder.embed_batched(
        contents, batch_size=settings.embed_batch_size, progress=True
    )
    log.info(f"embedding done in {time.time() - t0:.1f}s")

    store.upsert(doc_ids, vectors, payloads)


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental reindex of corpus delta")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=None,
        help="restrict to these source_collection values (e.g. 학사일정)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report diff and exit"
    )
    parser.add_argument(
        "--skip-bm25", action="store_true", help="skip BM25 rebuild"
    )
    args = parser.parse_args()

    df = pd.read_parquet(settings.corpus_path)
    log.info(f"loaded corpus: {len(df)} rows")

    scope_df = _filter_collections(df, args.collections)
    scope_doc_ids = set(scope_df["doc_id"].tolist()) if args.collections else None

    current_hashes = {
        row.doc_id: _content_hash(row.contents) for row in scope_df.itertuples()
    }
    previous = _load_manifest()
    if args.collections:
        previous_in_scope = {
            doc_id: h for doc_id, h in previous.items() if doc_id in scope_doc_ids
        }
    else:
        previous_in_scope = previous

    changed, deleted = _diff(current_hashes, previous_in_scope, scope_doc_ids)

    log.info(
        f"diff: changed={len(changed)} deleted={len(deleted)} "
        f"total_scope={len(current_hashes)}"
    )
    if changed[:5]:
        log.info(f"changed sample: {changed[:5]}")
    if deleted[:5]:
        log.info(f"deleted sample: {deleted[:5]}")

    if args.dry_run:
        log.info("dry-run: no changes applied")
        return 0

    if not changed and not deleted:
        log.info("no changes; nothing to do")
        return 0

    store = QdrantStore()
    store.ensure_collection(recreate=False)

    _delete_from_qdrant(store, deleted)
    _upsert_changed(store, scope_df, changed)

    if not args.skip_bm25:
        bm25 = OktBM25()
        payloads = [_normalize_payload(m) for m in df["metadata"].tolist()]
        contents = df["contents"].tolist()
        for p, c in zip(payloads, contents):
            p["contents"] = c
        bm25.build(df["doc_id"].tolist(), contents, payloads=payloads)
        bm25.save()

    merged = dict(previous)
    for doc_id in deleted:
        merged.pop(doc_id, None)
    merged.update(current_hashes)
    _save_manifest(merged)

    log.info(
        f"reindex done: changed={len(changed)} deleted={len(deleted)} "
        f"qdrant_count={store.count()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
