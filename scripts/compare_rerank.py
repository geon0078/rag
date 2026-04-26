"""A/B test: hybrid retrieval with vs without reranker.

For each non-negative QA in data/qa.parquet, runs hybrid retrieval once
(top_30) and computes two top-5 views:

- no_rerank: take top-5 directly from fused hybrid scores
- with_rerank: bge-reranker-v2-m3-ko reorders top_30 -> top-5

Metrics:
- routing_top3: ground-truth source_collection appears in top-3 collections
- recall@5:     any ground-truth doc_id appears in top-5 doc_ids
- collection@1: top-1 collection matches ground-truth

Run:
    python scripts/compare_rerank.py [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.reranker import KoReranker  # noqa: E402
from src.retrieval.router import route  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("compare_rerank")
QA_PATH = PROJECT_ROOT / "data" / "qa.parquet"
OUT_PATH = PROJECT_ROOT / "reports" / "compare_rerank.json"


def _gt_doc_ids(retrieval_gt) -> set[str]:
    if retrieval_gt is None:
        return set()
    out: set[str] = set()
    try:
        for group in retrieval_gt:
            if group is None:
                continue
            for d in group:
                if d:
                    out.add(str(d))
    except TypeError:
        pass
    return out


def _collection_of(c: dict) -> str | None:
    payload = c.get("payload") or {}
    return payload.get("source_collection")


def _doc_id_of(c: dict) -> str | None:
    payload = c.get("payload") or {}
    return c.get("doc_id") or payload.get("doc_id")


def _eval_branch(samples: list[dict], branch: str) -> dict:
    rows = [s for s in samples if s["row"]["qa_type"] != "negative"]
    n = len(rows)
    routing_correct = 0
    recall_correct = 0
    coll1_correct = 0
    for s in rows:
        gt_coll = s["row"]["source_collection"]
        gt_docs = s["gt_docs"]
        top5 = s[branch]
        cols = [_collection_of(c) for c in top5]
        if gt_coll in cols[:3]:
            routing_correct += 1
        if cols and cols[0] == gt_coll:
            coll1_correct += 1
        if gt_docs and any(_doc_id_of(c) in gt_docs for c in top5):
            recall_correct += 1
    return {
        "n": n,
        "routing_top3": routing_correct / n if n else 0,
        "collection_top1": coll1_correct / n if n else 0,
        "recall_at_5_doc": recall_correct / n if n else 0,
    }


async def _run(limit: int | None) -> dict:
    qa_df = pd.read_parquet(QA_PATH)
    if limit:
        qa_df = qa_df.head(limit)
    log.info(f"loaded {len(qa_df)} QAs")

    retriever = HybridRetriever()
    reranker = KoReranker()

    samples: list[dict] = []
    t_hybrid_total = 0.0
    t_rerank_total = 0.0

    total = len(qa_df)
    for idx, (_, row) in enumerate(qa_df.iterrows(), start=1):
        query = str(row["query"])
        try:
            decision = route(query)
            t0 = time.time()
            cands30 = await retriever.search(
                query, top_k=settings.top_k_dense, decision=decision
            )
            t_hybrid_total += time.time() - t0

            no_rerank_top5 = list(cands30[:5])

            t1 = time.time()
            with_rerank_top5 = (
                reranker.rerank(query, cands30, top_k=5) if cands30 else []
            )
            t_rerank_total += time.time() - t1
        except Exception as exc:
            log.error(f"[{idx}/{total}] error on {query!r}: {exc}")
            continue

        samples.append(
            {
                "row": row.to_dict(),
                "gt_docs": _gt_doc_ids(row.get("retrieval_gt")),
                "no_rerank": no_rerank_top5,
                "with_rerank": with_rerank_top5,
            }
        )
        if idx % 20 == 0:
            log.info(f"  progress {idx}/{total}")

    log.info(f"collected {len(samples)} samples")

    no = _eval_branch(samples, "no_rerank")
    yes = _eval_branch(samples, "with_rerank")

    return {
        "n_total": len(samples),
        "no_rerank": no,
        "with_rerank": yes,
        "delta": {
            "routing_top3": yes["routing_top3"] - no["routing_top3"],
            "collection_top1": yes["collection_top1"] - no["collection_top1"],
            "recall_at_5_doc": yes["recall_at_5_doc"] - no["recall_at_5_doc"],
        },
        "timing_sec": {
            "hybrid_total": round(t_hybrid_total, 2),
            "rerank_total": round(t_rerank_total, 2),
            "rerank_per_query_ms": round(
                t_rerank_total * 1000 / max(len(samples), 1), 1
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = asyncio.run(_run(args.limit))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    no = result["no_rerank"]
    yes = result["with_rerank"]
    d = result["delta"]
    print()
    print(f"{'metric':22s} {'no_rerank':>10s} {'with_rerank':>12s} {'delta':>8s}")
    print("-" * 56)
    for m in ("routing_top3", "collection_top1", "recall_at_5_doc"):
        print(f"{m:22s} {no[m]:>10.3f} {yes[m]:>12.3f} {d[m]:>+8.3f}")
    print()
    print(f"timing: {result['timing_sec']}")
    print(f"report: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
