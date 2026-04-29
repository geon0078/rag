"""Pipeline variant sweep on Golden Set (평가명세서 §8·§13).

Goal: 평가명세서 §8.1 통과 기준에 비춰 더 나은 파이프라인 구성을 찾는다.
  retrieval_recall@5 ≥ 0.85, recall@10 ≥ 0.95, MRR ≥ 0.65, nDCG@5 ≥ 0.75
  citation ≥ 0.90

Tests 7 variants — each runs the full pipeline over the Golden Set and reports
retrieval/generation metrics. Settings overrides are applied by mutating
``settings`` between runs (config.py uses plain BaseModel so attributes are
runtime-mutable). Re-creates RagPipeline per variant to pick up new state.

Variants:
  V1 baseline      hyde_on,  cc w=0.4, top_k=30, method=cc          (current default)
  V2 hyde_off      hyde_off, cc w=0.4, top_k=30, method=cc
  V3 cc_w_high     hyde_on,  cc w=0.6, top_k=30, method=cc          (more semantic)
  V4 cc_w_low      hyde_on,  cc w=0.2, top_k=30, method=cc          (more BM25)
  V5 rrf           hyde_on,            top_k=30, method=rrf         (rank fusion)
  V6 wider         hyde_on,  cc w=0.4, top_k=50, method=cc          (more retrieval candidates)
  V7 final_top10   hyde_on,  cc w=0.4, top_k=30, method=cc, final=10

Run:
    python scripts/pipeline_sweep.py [--limit 50] [--no-claim-faithfulness]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.eval.retrieval_metrics import RetrievalSample, aggregate  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

from scripts.eval_golden import _process_one  # noqa: E402

log = get_logger("pipeline_sweep")


@dataclass
class Variant:
    name: str
    description: str
    hyde_enabled: bool = True
    hybrid_method: str = "cc"
    hybrid_cc_weight: float = 0.4
    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_rerank_final: int = 5
    reranker_enabled: bool = False
    rewrite_enabled: bool = False


VARIANTS: list[Variant] = [
    # 베이스라인 + 단일 knob
    Variant("V1_baseline", "default — hyde_on, cc w=0.4, top_k=30, final=5"),
    Variant("V2_hyde_off", "hyde_off", hyde_enabled=False),
    Variant("V3_final_top10", "final=10 (이전 sweep 5/5 PASS 후보)", top_k_rerank_final=10),
    Variant("V4_cc_w_high", "cc w=0.6 (semantic-heavy)", hybrid_cc_weight=0.6),
    Variant("V5_cc_w_low", "cc w=0.2 (BM25-heavy)", hybrid_cc_weight=0.2),
    Variant("V6_rrf", "RRF rank fusion", hybrid_method="rrf"),

    # bge reranker on (GPU)
    Variant("V7_bge_rerank", "bge-reranker-v2-m3-ko ON", reranker_enabled=True),
    Variant(
        "V8_bge_rerank_top10",
        "bge-reranker ON + final=10",
        reranker_enabled=True,
        top_k_rerank_final=10,
    ),
    Variant(
        "V9_bge_rerank_wider",
        "bge-reranker ON + top_k=50",
        reranker_enabled=True,
        top_k_dense=50,
        top_k_sparse=50,
    ),
    Variant(
        "V10_bge_rerank_w06",
        "bge-reranker ON + cc w=0.6",
        reranker_enabled=True,
        hybrid_cc_weight=0.6,
    ),

    # rewriter (옵션 A) — 비교용
    Variant("V11_rewriter_only", "rewriter ON (no rerank)", rewrite_enabled=True),
    Variant(
        "V12_rewriter_plus_bge",
        "rewriter + bge-reranker (combo)",
        rewrite_enabled=True,
        reranker_enabled=True,
    ),
]


def _apply(v: Variant) -> None:
    settings.hybrid_method = v.hybrid_method
    settings.hybrid_cc_weight = v.hybrid_cc_weight
    settings.top_k_dense = v.top_k_dense
    settings.top_k_sparse = v.top_k_sparse
    settings.top_k_rerank_final = v.top_k_rerank_final
    settings.reranker_enabled = v.reranker_enabled


async def _run_variant(
    v: Variant,
    df: pd.DataFrame,
    use_claim: bool,
) -> dict[str, Any]:
    log.info(f"=== {v.name} | {v.description} ===")
    _apply(v)
    pipeline = RagPipeline(
        hyde_enabled=v.hyde_enabled,
        rewrite_enabled=v.rewrite_enabled,
    )
    cf_checker = None
    if use_claim:
        from src.eval.claim_faithfulness import ClaimFaithfulnessChecker
        cf_checker = ClaimFaithfulnessChecker()

    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        rec = await _process_one(pipeline, cf_checker, row)
        rec["_qid"] = str(row["qid"])
        rows.append(rec)
        if idx % 25 == 0:
            log.info(f"  [{v.name}] {idx}/{len(df)}")

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
    retrieval = aggregate(samples, ks=(5, 10))

    n = max(1, len(rows))
    gen: dict[str, float] = {
        "citation": sum(1 for r in rows if r.get("has_citation")) / n,
        "grounded": sum(1 for r in rows if r.get("grounded")) / n,
        "retry": sum(1 for r in rows if r.get("retry")) / n,
    }
    cf_scores = [r["claim_faithfulness"] for r in rows
                 if r.get("claim_faithfulness") is not None]
    if cf_scores:
        gen["claim_faithfulness"] = sum(cf_scores) / len(cf_scores)

    return {
        "name": v.name,
        "description": v.description,
        "config": {
            "hyde_enabled": v.hyde_enabled,
            "hybrid_method": v.hybrid_method,
            "hybrid_cc_weight": v.hybrid_cc_weight,
            "top_k_dense": v.top_k_dense,
            "top_k_rerank_final": v.top_k_rerank_final,
        },
        "n": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "retrieval": retrieval,
        "generation": gen,
    }


def _score_pass(row: dict[str, Any]) -> tuple[int, list[str]]:
    """평가명세서 §8.1 통과 여부."""
    rt = row["retrieval"].get("overall", {})
    gen = row["generation"]
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
    L.append("# Pipeline Variant Sweep\n")
    L.append(f"> 측정일: {summary['meta']['started_at']}  ")
    L.append(f"> Golden Set: `{summary['meta']['golden']}` · n={summary['meta']['n']}\n")
    L.append("평가명세서 §8.1 기준: recall@5≥0.85, recall@10≥0.95, MRR≥0.65, nDCG@5≥0.75, citation≥0.90, faithfulness≥0.85\n")
    L.append("---\n")

    L.append("## 1. 통합 비교\n")
    L.append("| Variant | recall@5 | recall@10 | MRR | nDCG@5 | citation | grounded | retry | PASS |")
    L.append("|---------|----------|-----------|-----|--------|----------|----------|-------|------|")
    for v in summary["variants"]:
        rt = v["retrieval"].get("overall", {})
        gen = v["generation"]
        passes, _ = _score_pass(v)
        L.append(
            f"| **{v['name']}** | {rt.get('recall@5', 0):.3f} | {rt.get('recall@10', 0):.3f} | "
            f"{rt.get('mrr', 0):.3f} | {rt.get('ndcg@5', 0):.3f} | "
            f"{gen.get('citation', 0):.3f} | {gen.get('grounded', 0):.3f} | "
            f"{gen.get('retry', 0):.3f} | {passes}/5 |"
        )
    L.append("")

    winner = max(summary["variants"], key=lambda v: (
        _score_pass(v)[0],
        v["retrieval"].get("overall", {}).get("recall@5", 0)
        + v["retrieval"].get("overall", {}).get("mrr", 0)
        + v["generation"].get("citation", 0)
        + v["generation"].get("grounded", 0),
    ))
    summary["winner"] = winner["name"]

    L.append("## 2. 베스트 시나리오\n")
    L.append(f"**🏆 {winner['name']}** — {winner['description']}\n")
    rt = winner["retrieval"].get("overall", {})
    gen = winner["generation"]
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
    L.append("")

    L.append("## 3. 변형별 설정\n")
    L.append("| Variant | description |")
    L.append("|---------|-------------|")
    for v in summary["variants"]:
        L.append(f"| {v['name']} | {v['description']} |")
    L.append("")

    L.append("## 4. 통과 기준 미달 항목\n")
    for v in summary["variants"]:
        passes, failed = _score_pass(v)
        if failed:
            L.append(f"- **{v['name']}** ({passes}/5 PASS): {', '.join(failed)}")
        else:
            L.append(f"- **{v['name']}** ({passes}/5 PASS): all criteria met ✅")
    L.append("")

    L.append("## 5. 산출물\n")
    L.append("- `reports/pipeline_sweep.json` — 모든 변형 raw 메트릭")
    L.append("- `reports/pipeline_sweep.md` — 이 보고서")
    L.append("")
    return "\n".join(L)


async def main_async(args: argparse.Namespace) -> int:
    p = Path(args.golden)
    if not p.exists():
        fallback = PROJECT_ROOT / "data" / "golden_candidates_v1.parquet"
        if fallback.exists():
            p = fallback
            log.warning(f"using fallback {p}")
        else:
            log.error(f"golden missing: {p}")
            return 1

    df = pd.read_parquet(p)
    # Adversarial parquet uses `retrieval_gt` (nested np array). eval_golden._process_one
    # expects `expected_doc_ids`. Auto-convert if needed.
    if "retrieval_gt" in df.columns and "expected_doc_ids" not in df.columns:
        from scripts.eval_adversarial import _flatten_gt
        df["expected_doc_ids"] = df["retrieval_gt"].apply(_flatten_gt)
        log.info("converted adversarial retrieval_gt → expected_doc_ids")
    if "curated" in df.columns and df["curated"].any():
        df = df[df["curated"]].reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    log.info(f"sweeping {len(VARIANTS)} variants × {len(df)} samples")

    started = datetime.now(timezone.utc).astimezone()
    variants_out: list[dict[str, Any]] = []
    for v in VARIANTS:
        out = await _run_variant(v, df, use_claim=not args.no_claim_faithfulness)
        variants_out.append(out)

    summary = {
        "meta": {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "golden": str(p),
            "n": len(df),
            "variants_count": len(VARIANTS),
        },
        "variants": variants_out,
    }

    md = _build_md(summary)

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline_sweep.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "pipeline_sweep.md").write_text(md, encoding="utf-8")
    log.info(f"wrote reports/pipeline_sweep.json + .md (winner={summary.get('winner')})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=str(PROJECT_ROOT / "data" / "golden_curated_v1.parquet"))
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Golden Set 행 제한 (default 50 — 7 variants × 50 ≈ 30분 이내)",
    )
    p.add_argument(
        "--no-claim-faithfulness",
        action="store_true",
        help="Solar 호출 절약 (claim faithfulness 스킵)",
    )
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
