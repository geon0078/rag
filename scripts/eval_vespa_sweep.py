"""Vespa retriever 변형 비교 (시나리오 A 풀 교체 측정).

수작업 250 (data/eval_dataset_250_manual.parquet) 으로 Vespa 의 4 ranking
profile + cc_weight 변형을 retrieval-only 비교한다.

답변 / Groundedness 단계는 제외 — 순수 retriever 의 recall/MRR/nDCG 만 측정.
우리 V4 baseline (Qdrant+Okt, recall@5 0.852) 와 정량 비교.

Run:
    python scripts/eval_vespa_sweep.py [--limit 0]
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

from src.eval.retrieval_metrics import RetrievalSample, aggregate  # noqa: E402
from src.retrieval.vespa_store import VespaStore  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("eval_vespa_sweep")


@dataclass
class Variant:
    name: str
    description: str
    rank_profile: str
    cc_weight: float = 0.6


VARIANTS: list[Variant] = [
    Variant("Vespa_BM25", "Vespa BM25 only (한국어 char-gram=2)", "bm25_only"),
    Variant("Vespa_Vector", "Solar embedding only (cosine)", "vector_only"),
    Variant("Vespa_Hybrid_w0.6", "hybrid_cc, vector weight=0.6 (V4 동등)", "hybrid_cc", 0.6),
    Variant("Vespa_Hybrid_w0.4", "hybrid_cc, vector weight=0.4 (BM25 강조)", "hybrid_cc", 0.4),
    Variant("Vespa_Hybrid_w0.8", "hybrid_cc, vector weight=0.8 (semantic 강조)", "hybrid_cc", 0.8),
    Variant("Vespa_RRF", "RRF approx (vector + 1/(60+bm25))", "rrf_approx"),
]


def _norm_expected(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split("|") if x.strip()]
    if hasattr(v, "tolist"):
        v = v.tolist()
    out: list[str] = []
    for x in v or []:
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, (list, tuple)):
            out.extend(str(y) for y in x)
        else:
            out.append(str(x))
    return out


def _run_variant(v: Variant, df: pd.DataFrame) -> dict[str, Any]:
    log.info(f"=== {v.name} | {v.description} ===")
    store = VespaStore(rank_profile=v.rank_profile, cc_weight=v.cc_weight)
    samples: list[RetrievalSample] = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        expected = _norm_expected(row.get("expected_doc_ids"))
        hits = store.search(str(row["query"]), top_k=10)
        retrieved = [h["doc_id"] for h in hits]
        samples.append(
            RetrievalSample(
                qid=str(row["qid"]),
                expected_doc_ids=tuple(expected),
                retrieved_doc_ids=tuple(retrieved),
                source_collection=row.get("source_collection"),
            )
        )
        if idx % 50 == 0:
            log.info(f"  [{v.name}] {idx}/{len(df)}")
    rt = aggregate(samples, ks=(5, 10))
    return {
        "name": v.name,
        "description": v.description,
        "rank_profile": v.rank_profile,
        "cc_weight": v.cc_weight,
        "n": len(samples),
        "retrieval": rt,
    }


def _build_md(summary: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# Vespa Retriever Sweep — 시나리오 A 풀 교체 측정\n")
    L.append(f"> 측정일: {summary['meta']['started_at']}  ")
    L.append(f"> Eval set: `{summary['meta']['eval_set']}` · n={summary['meta']['n']}\n")
    L.append("우리 Qdrant+Okt 베이스라인 (V4_cc_w_high): recall@5 **0.852** · MRR 0.678 · nDCG@5 0.716\n")
    L.append("---\n")

    L.append("## 1. Vespa 변형 비교\n")
    L.append("| Variant | rank_profile | cc_w | recall@5 | recall@10 | MRR | nDCG@5 | vs V4 (recall@5) |")
    L.append("|---------|--------------|------|----------|-----------|-----|--------|------------------|")
    baseline_r5 = 0.852
    for v in summary["variants"]:
        rt = v["retrieval"].get("overall", {})
        r5 = rt.get("recall@5", 0)
        delta = r5 - baseline_r5
        L.append(
            f"| **{v['name']}** | {v['rank_profile']} | {v['cc_weight']:.1f} | "
            f"{r5:.3f} | {rt.get('recall@10', 0):.3f} | {rt.get('mrr', 0):.3f} | "
            f"{rt.get('ndcg@5', 0):.3f} | {delta:+.3f} |"
        )
    L.append("")

    L.append("## 2. 컬렉션별 recall@5 (Vespa best variant)\n")
    best = max(summary["variants"], key=lambda x: x["retrieval"].get("overall", {}).get("recall@5", 0))
    by_col = best["retrieval"].get("by_collection", {})
    L.append(f"Best variant: **{best['name']}** (recall@5 {best['retrieval']['overall']['recall@5']:.3f})\n")
    L.append("| 컬렉션 | n | recall@5 | MRR |")
    L.append("|--------|---|----------|-----|")
    for sc, st in sorted(by_col.items()):
        L.append(f"| {sc} | {int(st['n'])} | {st['recall@5']:.3f} | {st['mrr']:.3f} |")
    L.append("")

    L.append("## 3. 결론\n")
    if best["retrieval"]["overall"]["recall@5"] >= baseline_r5:
        L.append(f"- ✅ Vespa best ({best['name']}) 가 V4 baseline ≥ recall@5 0.852 도달")
        L.append("- 풀 교체 가능성 있음. 추가 검증 (groundedness e2e) 권장.")
    else:
        L.append(f"- ⚠ Vespa best ({best['name']}) recall@5 {best['retrieval']['overall']['recall@5']:.3f} < V4 0.852")
        L.append(f"- 격차: {best['retrieval']['overall']['recall@5'] - baseline_r5:+.3f}")
        L.append("- 한국어 형태소 분석 (Okt) 부재 영향 추정 — Vespa 의 char-gram 이 Okt 에 못 미침.")
    L.append("")

    L.append("## 4. 산출물\n")
    L.append("- `reports/vespa_sweep.json`")
    L.append("- `reports/vespa_sweep.md` — 본 보고서")
    L.append("")
    return "\n".join(L)


async def main_async(args: argparse.Namespace) -> int:
    eval_path = Path(args.eval)
    if not eval_path.exists():
        log.error(f"eval set missing: {eval_path}")
        return 1
    df = pd.read_parquet(eval_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    log.info(f"sweeping {len(VARIANTS)} variants × {len(df)} samples")

    started = datetime.now(timezone.utc).astimezone()
    variants_out: list[dict[str, Any]] = []
    for v in VARIANTS:
        out = _run_variant(v, df)
        variants_out.append(out)

    summary = {
        "meta": {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "eval_set": str(eval_path),
            "n": len(df),
            "baseline_v4_recall@5": 0.852,
        },
        "variants": variants_out,
    }

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vespa_sweep.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "vespa_sweep.md").write_text(_build_md(summary), encoding="utf-8")
    best = max(variants_out, key=lambda x: x["retrieval"].get("overall", {}).get("recall@5", 0))
    log.info(f"wrote reports/vespa_sweep.json + .md (best={best['name']})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval",
        default=str(PROJECT_ROOT / "data" / "eval_dataset_250_manual.parquet"),
    )
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
