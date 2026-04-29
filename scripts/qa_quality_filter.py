"""Quality-filter the generated QA dataset.

Two checks, each independently configurable:

1. Round-trip retrieval validation
   Run the production hybrid retriever on every generated query at top_k=N
   (default 10). Drop the row if NONE of the expected ``retrieval_gt`` doc_ids
   appears in the candidate list — those are queries the corpus simply cannot
   answer with this index, so they're not useful as eval signal for the
   pipeline (they punish the generator for retrieval failures it can't fix).

2. Near-duplicate removal
   Greedy O(n^2) cosine similarity over a Solar query embedding of each
   ``query`` string. If two surviving queries have similarity ≥ threshold
   (default 0.95), drop the later one. Keeps generated quotas balanced across
   collections.

Run:
    python scripts/qa_quality_filter.py
        [--input data/qa.parquet]
        [--output data/qa.parquet]
        [--top-k 10]
        [--dup-threshold 0.95]
        [--no-roundtrip]
        [--no-dedup]

Writes a quality report to ``reports/qa_quality_report.json`` regardless.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.embeddings.solar_embedder import SolarEmbedder  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.router import route  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("qa_quality_filter")


def _flatten_gt(gt) -> list[str]:
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


async def _round_trip_check(df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, list[dict]]:
    retriever = HybridRetriever()
    keep_idx: list[int] = []
    drops: list[dict] = []
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False), start=1):
        query = str(row.query)
        gt = _flatten_gt(row.retrieval_gt)
        if not gt:
            keep_idx.append(i - 1)
            continue
        try:
            decision = route(query)
            candidates = await retriever.search(query, top_k=top_k, decision=decision)
            cand_ids = {c.get("doc_id") for c in candidates}
        except Exception as exc:
            log.error(f"[{i}/{total}] retrieval error for qid={row.qid}: {exc}")
            keep_idx.append(i - 1)
            continue

        if any(g in cand_ids for g in gt):
            keep_idx.append(i - 1)
        else:
            drops.append({"qid": row.qid, "query": query[:80], "retrieval_gt": gt})
        if i % 50 == 0:
            log.info(f"  round-trip {i}/{total} (kept {len(keep_idx)}, dropped {len(drops)})")

    return df.iloc[keep_idx].reset_index(drop=True), drops


async def _dedup_by_embedding(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[dict]]:
    embedder = SolarEmbedder(mode="query")
    queries = df["query"].astype(str).tolist()
    log.info(f"embedding {len(queries)} queries for dedup ...")
    vecs = await asyncio.to_thread(embedder.embed_batched, queries, None, False)
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    keep = [True] * len(df)
    drops: list[dict] = []
    for i in range(len(df)):
        if not keep[i]:
            continue
        sims = mat[i + 1 :] @ mat[i]
        for j_offset, sim in enumerate(sims):
            j = i + 1 + j_offset
            if keep[j] and sim >= threshold:
                keep[j] = False
                drops.append({
                    "qid_kept": df.iloc[i]["qid"],
                    "qid_dropped": df.iloc[j]["qid"],
                    "similarity": float(sim),
                    "kept_query": queries[i][:80],
                    "dropped_query": queries[j][:80],
                })
    out = df[pd.Series(keep)].reset_index(drop=True)
    return out, drops


async def main_async(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    if not inp.exists():
        log.error(f"input missing: {inp}")
        return 1
    df = pd.read_parquet(inp)
    log.info(f"loaded {len(df)} rows from {inp}")

    started = datetime.now(timezone.utc).astimezone()
    n_before = len(df)
    rt_drops: list[dict] = []
    dup_drops: list[dict] = []

    if not args.no_roundtrip:
        df, rt_drops = await _round_trip_check(df, args.top_k)
        log.info(f"round-trip: kept {len(df)}, dropped {len(rt_drops)}")

    if not args.no_dedup and len(df) > 1:
        df, dup_drops = await _dedup_by_embedding(df, args.dup_threshold)
        log.info(f"dedup: kept {len(df)}, dropped {len(dup_drops)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(f"wrote {len(df)} rows to {out}")

    report = {
        "started_at": started.isoformat(timespec="seconds"),
        "input": str(inp),
        "output": str(out),
        "rows_in": n_before,
        "rows_out": len(df),
        "round_trip": {
            "enabled": not args.no_roundtrip,
            "top_k": args.top_k,
            "dropped": len(rt_drops),
            "samples": rt_drops[:25],
        },
        "dedup": {
            "enabled": not args.no_dedup,
            "threshold": args.dup_threshold,
            "dropped": len(dup_drops),
            "samples": dup_drops[:25],
        },
    }
    report_path = PROJECT_ROOT / "reports" / "qa_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"report: {report_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(settings.data_dir / "qa.parquet"))
    p.add_argument("--output", default=str(settings.data_dir / "qa.parquet"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--dup-threshold", type=float, default=0.95)
    p.add_argument("--no-roundtrip", action="store_true")
    p.add_argument("--no-dedup", action="store_true")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
