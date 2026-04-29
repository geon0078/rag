"""Adversarial Set evaluator (평가명세서 §8 + §13).

Adversarial QA (250건) 를 사용해 현재 파이프라인을 측정한다. Golden Set 평가
와 같은 통과 기준을 쓰되, **challenge_type 별 breakdown** 으로 어떤 적대
패턴이 약한지 진단한다.

Challenge types (50건씩 균등):
  T1_conversational  — 짧은 구어체
  T2_vague           — 의도 모호
  T3_paraphrase      — 표현 변형
  T4_multi_intent    — 두 질문 결합
  T5_inference       — 다단계 추론 필요

Run:
    python scripts/eval_adversarial.py [--limit 0] [--no-claim-faithfulness]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.eval.retrieval_metrics import RetrievalSample, aggregate  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

from scripts.eval_golden import _process_one  # noqa: E402

log = get_logger("eval_adversarial")


def _flatten_gt(value: Any) -> list[str]:
    """retrieval_gt 는 AutoRAG 스타일 nested array — 평탄화."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    out: list[str] = []
    for item in value or []:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(str(x) for x in item)
        else:
            out.append(str(item))
    return out


def _aligned_row(row: pd.Series) -> pd.Series:
    """eval_golden._process_one 이 기대하는 shape 으로 변환.

    수작업 dataset 은 ``expected_doc_ids`` 가 list (또는 'a|b' 문자열) 로,
    AutoRAG-생성 dataset 은 ``retrieval_gt`` 가 nested array 로 들어옴.
    """
    if "expected_doc_ids" in row and row.get("expected_doc_ids") is not None:
        raw = row["expected_doc_ids"]
        if isinstance(raw, str):
            expected = [x.strip() for x in raw.split("|") if x.strip()]
        else:
            expected = _flatten_gt(raw)
    elif "retrieval_gt" in row:
        expected = _flatten_gt(row["retrieval_gt"])
    else:
        expected = []
    return pd.Series(
        {
            "qid": row["qid"],
            "query": row["query"],
            "expected_doc_ids": expected,
            "source_collection": row.get("source_collection"),
            "challenge_type": row.get("challenge_type", ""),
            "hop_type": row.get("hop_type", "single"),
        }
    )


def _bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        RetrievalSample(
            qid=r["qid"],
            expected_doc_ids=tuple(r.get("expected_doc_ids") or []),
            retrieved_doc_ids=tuple(r.get("retrieved_doc_ids") or []),
            source_collection=r.get("source_collection"),
        )
        for r in rows
        if "error" not in r
    ]
    rt = aggregate(samples, ks=(5, 10))
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "retrieval": rt,
        "generation": {
            "citation": sum(1 for r in rows if r.get("has_citation")) / n,
            "grounded": sum(1 for r in rows if r.get("grounded")) / n,
            "retry": sum(1 for r in rows if r.get("retry")) / n,
        },
    }


def _score_pass(block: dict[str, Any]) -> tuple[int, list[str]]:
    rt = block["retrieval"].get("overall", {})
    gen = block["generation"]
    targets = [
        ("recall@5", rt.get("recall@5", 0), 0.85),
        ("recall@10", rt.get("recall@10", 0), 0.95),
        ("mrr", rt.get("mrr", 0), 0.65),
        ("ndcg@5", rt.get("ndcg@5", 0), 0.75),
        ("citation", gen.get("citation", 0), 0.90),
    ]
    if "claim_faithfulness" in gen:
        targets.append(("claim_faithfulness", gen["claim_faithfulness"], 0.85))
    passes = sum(1 for _, v, t in targets if v >= t)
    failed = [k for k, v, t in targets if v < t]
    return passes, failed


def _build_md(summary: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# Adversarial Evaluation\n")
    L.append(f"> 측정일: {summary['meta']['started_at']}  ")
    L.append(f"> Adversarial Set: `{summary['meta']['adversarial']}` · n={summary['meta']['n']}\n")
    L.append("평가명세서 §8.1 기준: recall@5≥0.85, recall@10≥0.95, MRR≥0.65, nDCG@5≥0.75, citation≥0.90, faithfulness≥0.85\n")
    L.append("---\n")

    overall = summary["overall"]
    rt = overall["retrieval"].get("overall", {})
    gen = overall["generation"]

    L.append("## 1. 전체 메트릭\n")
    L.append("| 메트릭 | 값 | 목표 | 판정 |")
    L.append("|--------|-----|------|------|")
    rows = [
        ("recall@5", rt.get("recall@5", 0), 0.85),
        ("recall@10", rt.get("recall@10", 0), 0.95),
        ("MRR", rt.get("mrr", 0), 0.65),
        ("nDCG@5", rt.get("ndcg@5", 0), 0.75),
        ("citation", gen.get("citation", 0), 0.90),
    ]
    if "claim_faithfulness" in gen:
        rows.append(("faithfulness", gen["claim_faithfulness"], 0.85))
    for k, v, t in rows:
        verdict = "✅ PASS" if v >= t else "❌ FAIL"
        L.append(f"| {k} | {v:.3f} | ≥{t} | {verdict} |")
    L.append(f"| grounded rate | {gen.get('grounded', 0):.3f} | — | — |")
    L.append(f"| HyDE retry rate | {gen.get('retry', 0):.3f} | — | — |")
    passes, failed = _score_pass(overall)
    L.append("")
    L.append(f"**판정: {passes}/5 PASS** {'· 미달: ' + ', '.join(failed) if failed else '· 모든 기준 통과 ✅'}\n")

    L.append("## 2. Challenge type 별 breakdown\n")
    L.append("| type | n | recall@5 | recall@10 | MRR | nDCG@5 | citation | grounded | retry |")
    L.append("|------|---|----------|-----------|-----|--------|----------|----------|-------|")
    for ct, block in summary["by_challenge_type"].items():
        rt = block["retrieval"].get("overall", {})
        gen = block["generation"]
        L.append(
            f"| **{ct}** | {block['n']} | {rt.get('recall@5', 0):.3f} | {rt.get('recall@10', 0):.3f} | "
            f"{rt.get('mrr', 0):.3f} | {rt.get('ndcg@5', 0):.3f} | "
            f"{gen.get('citation', 0):.3f} | {gen.get('grounded', 0):.3f} | "
            f"{gen.get('retry', 0):.3f} |"
        )
    L.append("")

    L.append("## 3. Hop type 별\n")
    L.append("| type | n | recall@5 | MRR | nDCG@5 | grounded |")
    L.append("|------|---|----------|-----|--------|----------|")
    for ht, block in summary["by_hop_type"].items():
        rt = block["retrieval"].get("overall", {})
        gen = block["generation"]
        L.append(
            f"| **{ht}** | {block['n']} | {rt.get('recall@5', 0):.3f} | {rt.get('mrr', 0):.3f} | "
            f"{rt.get('ndcg@5', 0):.3f} | {gen.get('grounded', 0):.3f} |"
        )
    L.append("")

    L.append("## 4. Source collection 별 retrieval\n")
    by_col = overall["retrieval"].get("by_collection", {})
    L.append("| 컬렉션 | n | recall@5 | MRR |")
    L.append("|--------|---|----------|-----|")
    for sc, st in sorted(by_col.items()):
        L.append(f"| {sc} | {int(st['n'])} | {st['recall@5']:.3f} | {st['mrr']:.3f} |")
    L.append("")

    L.append("## 5. 약점 진단\n")
    weak: list[str] = []
    for ct, block in summary["by_challenge_type"].items():
        rt = block["retrieval"].get("overall", {})
        if rt.get("recall@5", 0) < 0.85:
            weak.append(f"- **{ct}**: recall@5={rt.get('recall@5', 0):.3f} (목표 0.85 미달)")
    if not weak:
        L.append("- 모든 challenge type 이 recall@5 ≥ 0.85 ✅")
    else:
        L.extend(weak)
    L.append("")

    L.append("## 6. 산출물\n")
    L.append("- `reports/eval_adversarial.json` — 모든 메트릭 + per-row")
    L.append("- `reports/eval_adversarial.md` — 이 보고서")
    L.append("")
    return "\n".join(L)


async def main_async(args: argparse.Namespace) -> int:
    p = Path(args.adversarial)
    if not p.exists():
        log.error(f"adversarial missing: {p}")
        return 1
    df = pd.read_parquet(p)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    aligned = df.apply(_aligned_row, axis=1)
    log.info(f"evaluating {len(aligned)} adversarial samples")

    pipeline = RagPipeline()
    cf_checker = None
    if not args.no_claim_faithfulness:
        from src.eval.claim_faithfulness import ClaimFaithfulnessChecker
        cf_checker = ClaimFaithfulnessChecker()

    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(aligned.iterrows(), start=1):
        rec = await _process_one(pipeline, cf_checker, row)
        rec["challenge_type"] = row["challenge_type"]
        rec["hop_type"] = row["hop_type"]
        rows.append(rec)
        if idx % 25 == 0:
            log.info(f"  progress {idx}/{len(aligned)}")

    overall = _bucket_metrics(rows)
    by_challenge: dict[str, Any] = {}
    for ct in sorted({r["challenge_type"] for r in rows if "challenge_type" in r}):
        sub = [r for r in rows if r.get("challenge_type") == ct]
        by_challenge[ct] = _bucket_metrics(sub)
    by_hop: dict[str, Any] = {}
    for ht in sorted({r["hop_type"] for r in rows if "hop_type" in r}):
        sub = [r for r in rows if r.get("hop_type") == ht]
        by_hop[ht] = _bucket_metrics(sub)

    started = datetime.now(timezone.utc).astimezone()
    summary = {
        "meta": {
            "started_at": started.isoformat(timespec="seconds"),
            "adversarial": str(p),
            "n": len(rows),
            "n_errors": sum(1 for r in rows if "error" in r),
            "config": {
                "hybrid_method": settings.hybrid_method,
                "hybrid_cc_weight": settings.hybrid_cc_weight,
                "top_k_dense": settings.top_k_dense,
                "top_k_rerank_final": settings.top_k_rerank_final,
                "reranker_enabled": settings.reranker_enabled,
            },
        },
        "overall": overall,
        "by_challenge_type": by_challenge,
        "by_hop_type": by_hop,
        "rows": rows,
    }

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_adversarial.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "eval_adversarial.md").write_text(_build_md(summary), encoding="utf-8")
    passes, failed = _score_pass(overall)
    log.info(f"wrote reports/eval_adversarial.json + .md (pass={passes}/5, failed={failed})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adversarial",
        default=str(PROJECT_ROOT / "data" / "eval_dataset_250_manual.parquet"),
        help="평가 데이터셋 — 기본은 수작업 250건 (정확도 검증됨).",
    )
    p.add_argument("--limit", type=int, default=0, help="0=전체 250건")
    p.add_argument("--no-claim-faithfulness", action="store_true")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
