"""Diagnose why fallback_rate is ~64% despite high context_recall.

For every non-negative QA, the script:
  1. Calls retriever + reranker once to capture INITIAL top-5 doc_ids
     (before any HyDE retry) — i.e. what the first-pass retrieval saw.
  2. Calls RagPipeline.run() for the full execution (with possible HyDE
     retry, groundedness verdict, fallback).
  3. Cross-references both with retrieval_gt to bucket each query into:

     - D_no_fallback        : pipeline returned a real answer
     - A_legit_miss         : fallback AND gt never appeared (retrieval failure)
     - B_groundedness_strict: fallback AND gt is in final contexts
                              (retrieval was fine, judge rejected)
     - C_retry_lost_gt      : fallback AND gt was in INITIAL but NOT in final
                              (HyDE retry pushed gt out)
     - X_unclassified       : edge cases (e.g. retrieval_gt empty)

Run:
    python scripts/diagnose_fallback.py [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.retrieval.router import route  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("diagnose_fallback")
QA_PATH = PROJECT_ROOT / "data" / "qa.parquet"
OUT_PATH = PROJECT_ROOT / "reports" / "fallback_diagnosis.json"


def _gt_doc_ids(retrieval_gt: Any) -> set[str]:
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


def _doc_ids_of(items: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for it in items:
        d = it.get("doc_id")
        if d:
            ids.add(str(d))
    return ids


def _classify(
    is_fallback: bool,
    gt_in_initial: bool,
    gt_in_final: bool,
    has_gt: bool,
) -> str:
    if not has_gt:
        return "X_unclassified"
    if not is_fallback:
        return "D_no_fallback"
    if gt_in_final:
        return "B_groundedness_strict"
    if gt_in_initial and not gt_in_final:
        return "C_retry_lost_gt"
    return "A_legit_miss"


async def _diagnose_one(
    pipeline: RagPipeline, query: str, gt_docs: set[str]
) -> dict[str, Any]:
    decision = route(query)
    initial_cands = await pipeline.retriever.search(
        query, top_k=settings.top_k_dense, decision=decision
    )
    initial_top5 = await asyncio.to_thread(
        pipeline.reranker.rerank,
        query,
        initial_cands,
        settings.top_k_rerank_final,
    )
    initial_ids = _doc_ids_of(initial_top5)

    result = await pipeline.run(query)
    final_ids = _doc_ids_of(result.get("contexts", []) or [])
    is_fallback = (
        result.get("verdict") == "notGrounded"
        and result.get("grounded", True) is False
    )
    has_gt = len(gt_docs) > 0
    gt_in_initial = bool(gt_docs & initial_ids)
    gt_in_final = bool(gt_docs & final_ids)

    return {
        "initial_top5": sorted(initial_ids),
        "final_doc_ids": sorted(final_ids),
        "verdict": result.get("verdict"),
        "retry": bool(result.get("retry")),
        "is_fallback": is_fallback,
        "gt_in_initial": gt_in_initial,
        "gt_in_final": gt_in_final,
        "category": _classify(is_fallback, gt_in_initial, gt_in_final, has_gt),
    }


async def _main(limit: int | None) -> dict[str, Any]:
    qa_df = pd.read_parquet(QA_PATH)
    qa_df = qa_df[qa_df["qa_type"] != "negative"].reset_index(drop=True)
    if limit:
        qa_df = qa_df.head(limit)
    log.info(f"diagnosing {len(qa_df)} non-negative QAs")

    pipeline = RagPipeline()

    records: list[dict[str, Any]] = []
    t_start = time.time()
    total = len(qa_df)
    for idx, (_, row) in enumerate(qa_df.iterrows(), start=1):
        query = str(row["query"])
        gt_docs = _gt_doc_ids(row.get("retrieval_gt"))
        try:
            diag = await _diagnose_one(pipeline, query, gt_docs)
        except Exception as exc:
            log.error(f"[{idx}/{total}] error on {query!r}: {exc}")
            continue

        records.append(
            {
                "qid": str(row["qid"]),
                "qa_type": row.get("qa_type"),
                "query": query,
                "gt_doc_ids": sorted(gt_docs),
                "source_collection_gt": row.get("source_collection"),
                **diag,
            }
        )
        if idx % 10 == 0:
            elapsed = time.time() - t_start
            log.info(
                f"  progress {idx}/{total} elapsed={elapsed:.0f}s "
                f"({elapsed / max(idx, 1):.2f}s/q)"
            )

    counts = Counter(r["category"] for r in records)
    n = len(records)
    summary = {
        "n_total": n,
        "categories": {
            k: {"n": v, "rate": round(v / n, 3) if n else 0.0}
            for k, v in counts.most_common()
        },
        "fallback_n": sum(1 for r in records if r["is_fallback"]),
        "fallback_rate": round(
            sum(1 for r in records if r["is_fallback"]) / n if n else 0.0, 3
        ),
    }
    return {"summary": summary, "records": records}


def _print_summary(summary: dict[str, Any]) -> None:
    print()
    print(f"n_total = {summary['n_total']}")
    print(
        f"fallback_n = {summary['fallback_n']} "
        f"(rate = {summary['fallback_rate']})"
    )
    print()
    print(f"{'category':28s} {'n':>5s} {'rate':>7s}")
    print("-" * 44)
    legend = {
        "D_no_fallback": "OK (real answer)",
        "B_groundedness_strict": "judge rejected (gt present)",
        "C_retry_lost_gt": "HyDE retry lost gt",
        "A_legit_miss": "retrieval miss",
        "X_unclassified": "no gt available",
    }
    for cat, stat in summary["categories"].items():
        print(f"{cat:28s} {stat['n']:>5d} {stat['rate']:>7.3f}  {legend.get(cat, '')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = asyncio.run(_main(args.limit))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(result["summary"])
    print()
    print(f"report: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
