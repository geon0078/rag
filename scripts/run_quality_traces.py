"""Run adversarial QAs through the production pipeline and save full traces.

Output (logs/quality_traces.jsonl) is the input to scripts/judge_traces.py.
Each line:

    {
        "qid": "...",
        "query": "...",
        "challenge_type": "T1_conversational",
        "expected_gt": "ground-truth answer string",
        "retrieval_gt": ["doc_id_a", "doc_id_b"],
        "trace": {
            "answer": "...",
            "grounded": bool,
            "verdict": "grounded|notSure|notGrounded",
            "retry": bool,
            "resolved_campus": "성남|...|null",
            "campus_was_inferred": bool,
            "elapsed_ms": int,
            "sources": [
                {"doc_id": "...", "score": 0.x, "category": "...", "campus": "..."}
            ],
            "retrieval_hit": bool         // any expected doc_id present in sources
        }
    }

Run:
    python scripts/run_quality_traces.py
        [--input data/qa_adversarial.parquet]
        [--output logs/quality_traces.jsonl]
        [--limit 0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("run_quality_traces")


def _flatten_gt(gt: Any) -> list[str]:
    if hasattr(gt, "tolist"):
        gt = gt.tolist()
    out: list[str] = []
    for inner in gt:
        if hasattr(inner, "tolist"):
            inner = inner.tolist()
        if isinstance(inner, (list, tuple, np.ndarray)):
            out.extend(str(x) for x in inner)
        else:
            out.append(str(inner))
    return out


async def main_async(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    if not inp.exists():
        log.error(f"adversarial QA missing: {inp}")
        return 1
    df = pd.read_parquet(inp)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    log.info(f"running {len(df)} traces")

    pipeline = RagPipeline()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_err = 0, 0
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            query = str(row.query)
            t0 = time.time()
            try:
                result = await pipeline.run(query)
            except Exception as exc:  # noqa: BLE001
                log.error(f"[{idx}/{len(df)}] pipeline error qid={row.qid}: {exc}")
                fh.write(json.dumps({
                    "qid": row.qid,
                    "query": query,
                    "challenge_type": row.challenge_type,
                    "error": f"{type(exc).__name__}: {exc}",
                }, ensure_ascii=False) + "\n")
                n_err += 1
                continue

            gt_ids = _flatten_gt(row.retrieval_gt)
            sources = result.get("sources", [])
            retrieval_hit = any(
                s.get("doc_id") in gt_ids for s in sources if s.get("doc_id")
            )

            gen_gt = row.generation_gt
            if hasattr(gen_gt, "tolist"):
                gen_gt = gen_gt.tolist()
            expected = gen_gt[0] if gen_gt else ""

            trace = {
                "qid": row.qid,
                "query": query,
                "challenge_type": row.challenge_type,
                "source_collection": row.source_collection,
                "expected_gt": expected,
                "retrieval_gt": gt_ids,
                "trace": {
                    "answer": result.get("answer", ""),
                    "grounded": bool(result.get("grounded", False)),
                    "verdict": result.get("verdict", ""),
                    "retry": bool(result.get("retry", False)),
                    "resolved_campus": result.get("resolved_campus"),
                    "campus_was_inferred": bool(result.get("campus_was_inferred", False)),
                    "elapsed_ms": int(result.get("elapsed_ms", int((time.time() - t0) * 1000))),
                    "sources": [
                        {
                            "doc_id": s.get("doc_id"),
                            "score": s.get("score"),
                            "category": s.get("category") or s.get("source_collection"),
                            "campus": s.get("campus"),
                        }
                        for s in sources[:5]
                    ],
                    "retrieval_hit": retrieval_hit,
                },
            }
            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
            n_ok += 1
            if idx % 25 == 0:
                log.info(f"  progress {idx}/{len(df)} ok={n_ok} err={n_err}")

    log.info(f"done: ok={n_ok} err={n_err} -> {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(settings.data_dir / "qa_adversarial.parquet"))
    p.add_argument("--output", default=str(PROJECT_ROOT / "logs" / "quality_traces.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
