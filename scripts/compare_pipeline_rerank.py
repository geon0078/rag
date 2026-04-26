"""Full-pipeline A/B test: reranker ON vs OFF.

Runs the entire RagPipeline (retrieve -> rerank -> generate -> verify -> retry)
twice for each QA — once with the bge cross-encoder reranker, once with a
no-op reranker that just returns the top-K from hybrid scores. Computes the
4 supplementary metrics for each branch and writes a combined report.

This isolates the reranker's contribution to *answer quality*, not just
retrieval recall (which compare_rerank.py already covers).

Run:
    python scripts/compare_pipeline_rerank.py [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("compare_pipeline_rerank")

QA_PATH = PROJECT_ROOT / "data" / "qa.parquet"
OUT_PATH = PROJECT_ROOT / "reports" / "compare_pipeline_rerank.json"

REJECTION_KEYWORDS = ("찾을 수 없", "제공된 자료", "확인할 수 없", "정보가 없")
CITATION_PATTERN = re.compile(r"\[출처[:：]\s*[^\]]+\]")
CAMPUS_ALL = "전체"


class _NoOpReranker:
    """Bypass reranker: keep hybrid score order, take top_k as-is."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        k = top_k or settings.top_k_rerank_final
        out: list[dict[str, Any]] = []
        for c in list(candidates)[:k]:
            item = dict(c)
            item.setdefault("rerank_score", item.get("rrf_score", 0.0))
            out.append(item)
        return out


def _is_rejection(answer: str, grounded: bool) -> bool:
    if not grounded:
        return True
    return any(kw in answer for kw in REJECTION_KEYWORDS)


def _eval_branch(samples: list[dict], key: str) -> dict:
    metrics: dict[str, Any] = {}

    neg = [s for s in samples if s["row"].get("qa_type") == "negative"]
    rejected = sum(
        1
        for s in neg
        if _is_rejection(s[key]["answer"], s[key].get("grounded", False))
    )
    metrics["negative_rejection"] = {
        "n": len(neg),
        "rejected": rejected,
        "rate": (rejected / len(neg)) if neg else None,
        "target": 0.80,
    }

    target_rows = [s for s in samples if s["row"].get("qa_type") == "filter_required"]
    correct = 0
    for s in target_rows:
        meta = s["row"].get("metadata") or {}
        expected = meta.get("campus_filter") if isinstance(meta, dict) else None
        contexts = s[key].get("contexts", [])
        if not contexts:
            continue
        got = [(c.get("metadata") or {}).get("campus") for c in contexts]
        if all(c in (expected, CAMPUS_ALL) for c in got):
            correct += 1
    metrics["campus_filter"] = {
        "n": len(target_rows),
        "correct": correct,
        "accuracy": (correct / len(target_rows)) if target_rows else None,
        "target": 1.00,
    }

    rows = [s for s in samples if s["row"].get("qa_type") != "negative"]
    correct = 0
    for s in rows:
        gt = s["row"].get("source_collection")
        contexts = s[key].get("contexts", [])[:3]
        retrieved = [
            (c.get("metadata") or {}).get("source_collection") for c in contexts
        ]
        if gt in retrieved:
            correct += 1
    metrics["routing_top3"] = {
        "n": len(rows),
        "correct": correct,
        "accuracy": (correct / len(rows)) if rows else None,
        "target": 0.95,
    }

    correct = 0
    for s in rows:
        if CITATION_PATTERN.search(s[key]["answer"]):
            correct += 1
    metrics["citation"] = {
        "n": len(rows),
        "correct": correct,
        "accuracy": (correct / len(rows)) if rows else None,
        "target": 0.90,
    }

    fb_count = sum(1 for s in samples if s[key].get("verdict") == "notGrounded")
    metrics["fallback_rate"] = {
        "n": len(samples),
        "fallback": fb_count,
        "rate": (fb_count / len(samples)) if samples else None,
    }

    return metrics


async def _run_one(pipeline: RagPipeline, query: str) -> dict[str, Any]:
    try:
        return await pipeline.run(query)
    except Exception as exc:
        log.error(f"pipeline error on {query!r}: {exc}")
        return {
            "answer": "",
            "grounded": False,
            "verdict": "error",
            "sources": [],
            "contexts": [],
        }


async def _main_async(limit: int | None) -> dict:
    qa_df = pd.read_parquet(QA_PATH)
    if limit:
        qa_df = qa_df.head(limit)
    log.info(f"loaded {len(qa_df)} QAs")

    pipe_on = RagPipeline()
    pipe_off = RagPipeline(reranker=_NoOpReranker())

    samples: list[dict] = []
    t_on_total = 0.0
    t_off_total = 0.0

    total = len(qa_df)
    for idx, (_, row) in enumerate(qa_df.iterrows(), start=1):
        query = str(row["query"])

        t0 = time.time()
        res_on = await _run_one(pipe_on, query)
        t_on_total += time.time() - t0

        t1 = time.time()
        res_off = await _run_one(pipe_off, query)
        t_off_total += time.time() - t1

        samples.append(
            {"row": row.to_dict(), "with_rerank": res_on, "no_rerank": res_off}
        )
        if idx % 10 == 0:
            log.info(f"  progress {idx}/{total}")

    log.info(f"collected {len(samples)} samples")

    metrics_on = _eval_branch(samples, "with_rerank")
    metrics_off = _eval_branch(samples, "no_rerank")

    return {
        "n_total": len(samples),
        "with_rerank": metrics_on,
        "no_rerank": metrics_off,
        "timing_sec": {
            "with_rerank_total": round(t_on_total, 2),
            "no_rerank_total": round(t_off_total, 2),
            "with_rerank_per_query": round(t_on_total / max(len(samples), 1), 2),
            "no_rerank_per_query": round(t_off_total / max(len(samples), 1), 2),
        },
    }


def _print_summary(result: dict) -> None:
    on = result["with_rerank"]
    off = result["no_rerank"]
    metrics = [
        ("negative_rejection", "rate"),
        ("campus_filter", "accuracy"),
        ("routing_top3", "accuracy"),
        ("citation", "accuracy"),
        ("fallback_rate", "rate"),
    ]
    print()
    print(f"{'metric':22s} {'no_rerank':>12s} {'with_rerank':>14s} {'delta':>10s}")
    print("-" * 62)
    for name, key in metrics:
        v_off = off[name].get(key)
        v_on = on[name].get(key)
        if v_off is None or v_on is None:
            print(f"{name:22s} {'n/a':>12s} {'n/a':>14s} {'-':>10s}")
            continue
        d = v_on - v_off
        print(f"{name:22s} {v_off:>12.3f} {v_on:>14.3f} {d:>+10.3f}")
    print()
    print(f"timing: {result['timing_sec']}")
    print(f"report: {OUT_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = asyncio.run(_main_async(args.limit))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
