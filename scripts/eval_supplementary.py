"""Phase 5 supplementary validators (single pipeline pass for efficiency).

Runs the RAG pipeline once over data/qa.parquet and computes 4 metrics from
평가-개선.md §6:

    1. Negative rejection rate            (reports/eval_negative.json)        ≥ 0.80
    2. Campus filter accuracy             (reports/eval_campus_filter.json)   = 1.00
    3. Category routing accuracy (top-3)  (reports/eval_routing.json)         ≥ 0.95
    4. Citation format accuracy           (reports/eval_citation.json)        ≥ 0.90

Writes a combined snapshot to reports/eval_supplementary.json.

Run:
    python scripts/eval_supplementary.py [--limit N] [--out-dir reports]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("eval_supplementary")

QA_PATH = PROJECT_ROOT / "data" / "qa.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports"

REJECTION_KEYWORDS = ("찾을 수 없", "제공된 자료", "확인할 수 없", "정보가 없")
CITATION_PATTERN = re.compile(r"\[출처[:：]\s*[^\]]+\]")
CAMPUS_ALL = "전체"


async def _run_all(qa_df: pd.DataFrame, limit: int | None) -> list[dict]:
    pipeline = RagPipeline()
    rows = qa_df.head(limit) if limit else qa_df
    total = len(rows)
    out: list[dict] = []
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        query = str(row["query"])
        try:
            result = await pipeline.run(query)
        except Exception as exc:
            log.error(f"[{idx}/{total}] error for {query!r}: {exc}")
            continue
        out.append({"row": row.to_dict(), "result": result})
        if idx % 10 == 0:
            log.info(f"  progress {idx}/{total}")
    return out


def _is_rejection(answer: str, grounded: bool) -> bool:
    if not grounded:
        return True
    return any(kw in answer for kw in REJECTION_KEYWORDS)


def _eval_negative(samples: list[dict]) -> dict:
    neg = [s for s in samples if s["row"].get("qa_type") == "negative"]
    if not neg:
        return {
            "n": 0,
            "rejected": 0,
            "rejection_rate": None,
            "target": 0.80,
            "passed": False,
            "failures": [],
        }
    rejected = sum(
        1
        for s in neg
        if _is_rejection(s["result"]["answer"], s["result"].get("grounded", False))
    )
    rate = rejected / len(neg)
    failures = [
        {
            "qid": s["row"]["qid"],
            "query": s["row"]["query"],
            "answer": s["result"]["answer"][:200],
        }
        for s in neg
        if not _is_rejection(
            s["result"]["answer"], s["result"].get("grounded", False)
        )
    ]
    return {
        "n": len(neg),
        "rejected": rejected,
        "rejection_rate": rate,
        "target": 0.80,
        "passed": rate >= 0.80,
        "failures": failures,
    }


def _eval_campus_filter(samples: list[dict]) -> dict:
    target_rows = [s for s in samples if s["row"].get("qa_type") == "filter_required"]
    if not target_rows:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": None,
            "target": 1.00,
            "passed": False,
            "failures": [],
        }
    correct = 0
    failures: list[dict] = []
    for s in target_rows:
        meta = s["row"].get("metadata") or {}
        expected = meta.get("campus_filter") if isinstance(meta, dict) else None
        contexts = s["result"].get("contexts", [])
        if not contexts:
            failures.append(
                {"qid": s["row"]["qid"], "expected": expected, "got": "no contexts"}
            )
            continue
        got = [(c.get("metadata") or {}).get("campus") for c in contexts]
        all_match = all(c in (expected, CAMPUS_ALL) for c in got)
        if all_match:
            correct += 1
        else:
            failures.append(
                {"qid": s["row"]["qid"], "expected": expected, "got": got}
            )
    acc = correct / len(target_rows)
    return {
        "n": len(target_rows),
        "correct": correct,
        "accuracy": acc,
        "target": 1.00,
        "passed": acc >= 1.00,
        "failures": failures[:20],
    }


def _eval_routing(samples: list[dict]) -> dict:
    rows = [s for s in samples if s["row"].get("qa_type") != "negative"]
    if not rows:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": None,
            "target": 0.95,
            "passed": False,
            "failures": [],
        }
    correct = 0
    failures: list[dict] = []
    for s in rows:
        gt_collection = s["row"].get("source_collection")
        contexts = s["result"].get("contexts", [])[:3]
        retrieved = [
            (c.get("metadata") or {}).get("source_collection") for c in contexts
        ]
        if gt_collection in retrieved:
            correct += 1
        else:
            failures.append(
                {
                    "qid": s["row"]["qid"],
                    "expected": gt_collection,
                    "got_top3": retrieved,
                }
            )
    acc = correct / len(rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": acc,
        "target": 0.95,
        "passed": acc >= 0.95,
        "failures": failures[:20],
    }


def _eval_citation(samples: list[dict]) -> dict:
    rows = [s for s in samples if s["row"].get("qa_type") != "negative"]
    if not rows:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": None,
            "target": 0.90,
            "passed": False,
            "failures": [],
        }
    correct = 0
    failures: list[dict] = []
    for s in rows:
        ans = s["result"]["answer"]
        if CITATION_PATTERN.search(ans):
            correct += 1
        else:
            failures.append({"qid": s["row"]["qid"], "answer": ans[:200]})
    acc = correct / len(rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": acc,
        "target": 0.90,
        "passed": acc >= 0.90,
        "failures": failures[:20],
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not QA_PATH.exists():
        log.error(f"qa.parquet missing: {QA_PATH}")
        return 1

    qa_df = pd.read_parquet(QA_PATH)
    if "qa_type" not in qa_df.columns:
        log.error("qa.parquet has no qa_type. Run scripts/finalize_qa.py first.")
        return 2

    log.info(f"Running pipeline over {len(qa_df)} QAs (limit={args.limit})")
    samples = asyncio.run(_run_all(qa_df, args.limit))
    log.info(f"Collected {len(samples)} pipeline runs")

    neg = _eval_negative(samples)
    campus = _eval_campus_filter(samples)
    routing = _eval_routing(samples)
    citation = _eval_citation(samples)

    out = args.out_dir
    _write_json(out / "eval_negative.json", neg)
    _write_json(out / "eval_campus_filter.json", campus)
    _write_json(out / "eval_routing.json", routing)
    _write_json(out / "eval_citation.json", citation)
    _write_json(
        out / "eval_supplementary.json",
        {
            "negative": neg,
            "campus_filter": campus,
            "routing": routing,
            "citation": citation,
        },
    )

    log.info("=" * 60)
    log.info(
        f"Negative rejection: {neg['rejection_rate']} (target≥{neg['target']}) -> "
        f"{'PASS' if neg['passed'] else 'FAIL'}"
    )
    log.info(
        f"Campus filter:      {campus['accuracy']} (target={campus['target']}) -> "
        f"{'PASS' if campus['passed'] else 'FAIL'}"
    )
    log.info(
        f"Routing top-3:      {routing['accuracy']} (target≥{routing['target']}) -> "
        f"{'PASS' if routing['passed'] else 'FAIL'}"
    )
    log.info(
        f"Citation format:    {citation['accuracy']} (target≥{citation['target']}) -> "
        f"{'PASS' if citation['passed'] else 'FAIL'}"
    )
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
