"""Golden Set 통합 평가 (평가명세서 §4·§8·§12).

Golden Set (curated 100건) 을 입력받아 다음을 한 번에 측정:

  Retrieval 분리 메트릭 (E2):
    - recall@5, recall@10, hit@5, hit@10, MRR, nDCG@5
    - 카테고리별 분리 측정

  Generation 메트릭:
    - groundedness verdict 비율
    - claim-level faithfulness (E3, RAGChecker 패턴)
    - citation accuracy (`[출처: ...]` 패턴)

  Synthetic vs Golden 갭 분석:
    - reports/eval_supplementary.json (Trial F, paraphrase) 와 비교
    - 갭 ≥ 15pt 면 "평가 편향 의심" 경고

출력:
  reports/eval_golden.json    — 모든 메트릭 + 메타데이터
  reports/eval_golden.md      — 사람용 요약 보고서

Run:
    python scripts/eval_golden.py
        [--golden data/golden_curated_v1.parquet]
        [--limit 0]
        [--no-claim-faithfulness]    # Solar 호출 절약 (~30분 단축)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.eval.claim_faithfulness import ClaimFaithfulnessChecker, FaithfulnessResult  # noqa: E402
from src.eval.retrieval_metrics import RetrievalSample, aggregate  # noqa: E402
from src.generation.prompts import format_context  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("eval_golden")

CITATION_PATTERN = re.compile(r"\[출처[:：]\s*[^\]]+\]")
GAP_WARNING_THRESHOLD = 0.15


def _flatten(seq: Any) -> list[str]:
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    out: list[str] = []
    for item in seq or []:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(str(x) for x in item)
        else:
            out.append(str(item))
    return out


async def _process_one(
    pipeline: RagPipeline,
    cf_checker: ClaimFaithfulnessChecker | None,
    row: pd.Series,
) -> dict[str, Any]:
    query = str(row["query"])
    expected_doc_ids = _flatten(row["expected_doc_ids"])
    sc = str(row.get("source_collection") or "")

    try:
        result = await pipeline.run(query)
    except Exception as exc:  # noqa: BLE001
        log.error(f"pipeline error qid={row['qid']}: {exc}")
        return {
            "qid": str(row["qid"]),
            "query": query,
            "source_collection": sc,
            "error": f"{type(exc).__name__}: {exc}",
        }

    sources = result.get("sources") or []
    retrieved_ids = [s.get("doc_id") for s in sources if s.get("doc_id")]

    has_citation = bool(CITATION_PATTERN.search(result.get("answer", "")))

    out: dict[str, Any] = {
        "qid": str(row["qid"]),
        "query": query,
        "source_collection": sc,
        "expected_doc_ids": expected_doc_ids,
        "retrieved_doc_ids": retrieved_ids[:10],
        "verdict": result.get("verdict"),
        "grounded": bool(result.get("grounded", False)),
        "retry": bool(result.get("retry", False)),
        "answer": result.get("answer", "")[:500],
        "has_citation": has_citation,
    }

    if cf_checker is not None:
        try:
            cf: FaithfulnessResult = await cf_checker.score_answer(
                result.get("answer", ""),
                format_context(result.get("contexts") or sources),
            )
            out["claim_faithfulness"] = cf.score
            out["claim_n"] = cf.n_claims
        except Exception as exc:  # noqa: BLE001
            log.warning(f"claim_faithfulness error qid={row['qid']}: {exc}")
            out["claim_faithfulness"] = None
            out["claim_n"] = 0

    return out


def _load_synth_baseline() -> dict[str, float] | None:
    p = PROJECT_ROOT / "reports" / "eval_supplementary.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return {
            "negative_rejection": d["negative"]["rejection_rate"],
            "campus_filter": d["campus_filter"]["accuracy"],
            "routing_top3": d["routing"]["accuracy"],
            "citation": d["citation"]["accuracy"],
        }
    except Exception:  # noqa: BLE001
        return None


def _build_md(meta: dict[str, Any], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Golden Set Evaluation\n")
    lines.append(f"> 측정일: {meta.get('started_at','—')}  ")
    lines.append(f"> n: **{meta.get('n', 0)}**  ")
    lines.append(f"> Pipeline config: hybrid_cc weight={meta.get('config',{}).get('hybrid_cc_weight','?')}\n")
    lines.append("---\n")

    lines.append("## 1. Retrieval 메트릭 (E2)\n")
    rt = summary.get("retrieval", {}).get("overall", {})
    lines.append("| 메트릭 | 값 | 목표 | 판정 |")
    lines.append("|--------|-----|------|------|")
    targets = {"recall@5": 0.85, "recall@10": 0.95, "mrr": 0.65, "ndcg@5": 0.75}
    for k, target in targets.items():
        v = rt.get(k, 0.0)
        verdict = "✅ PASS" if v >= target else "❌ FAIL"
        lines.append(f"| {k} | {v:.3f} | ≥{target} | {verdict} |")
    lines.append("")

    lines.append("### 카테고리별 recall@5\n")
    by_col = summary.get("retrieval", {}).get("by_collection", {})
    lines.append("| 카테고리 | n | recall@5 | mrr |")
    lines.append("|----------|---|----------|-----|")
    for sc, st in sorted(by_col.items()):
        lines.append(f"| {sc} | {int(st['n'])} | {st['recall@5']:.3f} | {st['mrr']:.3f} |")
    lines.append("")

    lines.append("## 2. Generation 메트릭\n")
    gen = summary.get("generation", {})
    lines.append("| 메트릭 | 값 | 목표 | 판정 |")
    lines.append("|--------|-----|------|------|")
    if "claim_faithfulness" in gen:
        cf = gen["claim_faithfulness"]
        lines.append(f"| claim-level faithfulness | {cf:.3f} | ≥0.85 | {'✅ PASS' if cf >= 0.85 else '❌ FAIL'} |")
    cit = gen.get("citation", 0.0)
    lines.append(f"| citation accuracy | {cit:.3f} | ≥0.90 | {'✅ PASS' if cit >= 0.90 else '❌ FAIL'} |")
    lines.append(f"| grounded rate | {gen.get('grounded_rate', 0.0):.3f} | — | — |")
    lines.append(f"| HyDE retry rate | {gen.get('retry_rate', 0.0):.3f} | — | — |")
    lines.append("")

    lines.append("## 3. 갭 분석 (Synthetic vs Golden)\n")
    if summary.get("gap"):
        lines.append("| 메트릭 | Synthetic (Trial F) | Golden | Gap | 신호 |")
        lines.append("|--------|---------------------|--------|-----|------|")
        for m, g in summary["gap"].items():
            warn = "⚠️ paraphrase bias 의심" if abs(g["gap"]) >= GAP_WARNING_THRESHOLD else "정상"
            lines.append(f"| {m} | {g['synthetic']:.3f} | {g['golden']:.3f} | {g['gap']:+.3f} | {warn} |")
    else:
        lines.append("_synthetic baseline 없음 (`reports/eval_supplementary.json` 미발견)_")
    lines.append("")

    lines.append("## 4. 산출물\n")
    lines.append("- `reports/eval_golden.json` — 모든 메트릭 + row-level")
    lines.append("- `reports/eval_golden.md` — 이 보고서")
    lines.append("")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    p = Path(args.golden)
    if not p.exists():
        log.error(f"golden set missing: {p}")
        return 1
    df = pd.read_parquet(p)
    if "curated" in df.columns and df["curated"].any():
        df = df[df["curated"]].reset_index(drop=True)
        log.info(f"using {len(df)} curated rows")
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    log.info(f"evaluating {len(df)} golden samples")

    pipeline = RagPipeline()
    cf_checker = ClaimFaithfulnessChecker() if not args.no_claim_faithfulness else None

    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        rec = await _process_one(pipeline, cf_checker, row)
        rows.append(rec)
        if idx % 10 == 0:
            log.info(f"  progress {idx}/{len(df)}")

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
    retrieval_summary = aggregate(samples, ks=(5, 10))

    cit = sum(1 for r in rows if r.get("has_citation")) / max(1, len(rows))
    grounded = sum(1 for r in rows if r.get("grounded")) / max(1, len(rows))
    retry = sum(1 for r in rows if r.get("retry")) / max(1, len(rows))
    gen_block: dict[str, float] = {
        "citation": cit,
        "grounded_rate": grounded,
        "retry_rate": retry,
    }
    cf_scores = [r["claim_faithfulness"] for r in rows
                 if r.get("claim_faithfulness") is not None]
    if cf_scores:
        gen_block["claim_faithfulness"] = sum(cf_scores) / len(cf_scores)

    synth = _load_synth_baseline()
    gap_block: dict[str, dict[str, float]] | None = None
    if synth is not None:
        gap_block = {}
        if "citation" in synth:
            gap_block["citation"] = {
                "synthetic": synth["citation"],
                "golden": cit,
                "gap": synth["citation"] - cit,
            }

    started = datetime.now(timezone.utc).astimezone()
    summary = {
        "meta": {
            "started_at": started.isoformat(timespec="seconds"),
            "n": len(rows),
            "n_errors": sum(1 for r in rows if "error" in r),
            "judge_model_claim": "solar-pro3" if cf_checker else None,
            "config": {
                "hybrid_method": settings.hybrid_method,
                "hybrid_cc_weight": settings.hybrid_cc_weight,
                "hybrid_cc_normalize": settings.hybrid_cc_normalize,
                "reranker_enabled": settings.reranker_enabled,
            },
        },
        "retrieval": retrieval_summary,
        "generation": gen_block,
        "gap": gap_block,
        "rows": rows,
    }

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_golden.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "eval_golden.md").write_text(
        _build_md(summary["meta"], summary), encoding="utf-8"
    )
    log.info(f"wrote reports/eval_golden.json + .md")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=str(settings.data_dir / "golden_curated_v1.parquet"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-claim-faithfulness", action="store_true")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
