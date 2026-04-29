"""HyDE on/off A/B test on Golden Set (평가명세서 §13.3).

Runs RagPipeline twice over the same Golden Set:
  - Arm A (control):   hyde_enabled=True   — current production behavior
  - Arm B (treatment): hyde_enabled=False  — retry uses raw query

Compares retrieval/generation metrics + paired bootstrap CI on the
binary outcomes (grounded, has_citation). Outputs:

  reports/ab_test_hyde.json   — both arms + delta + bootstrap CIs
  reports/ab_test_hyde.md     — operator-readable summary

Run:
    python scripts/ab_test_hyde.py
        [--golden data/golden_curated_v1.parquet]
        [--limit 0]
        [--no-claim-faithfulness]
        [--bootstrap 1000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.eval.claim_faithfulness import ClaimFaithfulnessChecker  # noqa: E402
from src.eval.retrieval_metrics import RetrievalSample, aggregate  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

from scripts.eval_golden import _process_one  # noqa: E402

log = get_logger("ab_test_hyde")


def _resolve_golden(arg_path: str) -> Path:
    p = Path(arg_path)
    if p.exists():
        return p
    fallback = PROJECT_ROOT / "data" / "golden_candidates_v1.parquet"
    if fallback.exists():
        log.warning(f"golden missing at {p}; using fallback {fallback}")
        return fallback
    raise FileNotFoundError(f"No golden parquet found (tried {p} and {fallback})")


async def _run_arm(
    label: str,
    df: pd.DataFrame,
    hyde_enabled: bool,
    use_claim: bool,
) -> dict[str, Any]:
    log.info(f"=== Arm {label} (hyde_enabled={hyde_enabled}) ===")
    pipeline = RagPipeline(hyde_enabled=hyde_enabled)
    cf_checker = ClaimFaithfulnessChecker() if use_claim else None

    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        rec = await _process_one(pipeline, cf_checker, row)
        rec["_qid"] = str(row["qid"])
        rows.append(rec)
        if idx % 10 == 0:
            log.info(f"  [{label}] progress {idx}/{len(df)}")

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

    n = max(1, len(rows))
    gen: dict[str, float] = {
        "citation": sum(1 for r in rows if r.get("has_citation")) / n,
        "grounded_rate": sum(1 for r in rows if r.get("grounded")) / n,
        "retry_rate": sum(1 for r in rows if r.get("retry")) / n,
    }
    cf_scores = [r["claim_faithfulness"] for r in rows
                 if r.get("claim_faithfulness") is not None]
    if cf_scores:
        gen["claim_faithfulness"] = sum(cf_scores) / len(cf_scores)

    return {
        "label": label,
        "hyde_enabled": hyde_enabled,
        "n": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "retrieval": retrieval_summary,
        "generation": gen,
        "rows": rows,
    }


def _paired_bootstrap_ci(
    arm_a: list[int],
    arm_b: list[int],
    iters: int,
    seed: int = 7,
) -> dict[str, float]:
    """Paired bootstrap on the per-sample binary outcome.

    Returns mean delta (B - A) and 95% CI bounds. Same indices are
    sampled together so the pairing structure is preserved.
    """
    if len(arm_a) != len(arm_b) or not arm_a:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_le_0": 1.0}
    rng = random.Random(seed)
    n = len(arm_a)
    deltas: list[float] = []
    for _ in range(iters):
        idxs = [rng.randrange(n) for _ in range(n)]
        sa = sum(arm_a[i] for i in idxs) / n
        sb = sum(arm_b[i] for i in idxs) / n
        deltas.append(sb - sa)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[max(0, int(0.975 * iters) - 1)]
    p_le = sum(1 for d in deltas if d <= 0) / iters
    return {
        "mean_delta": sum(deltas) / iters,
        "ci_low": lo,
        "ci_high": hi,
        "p_le_0": p_le,
    }


def _align_rows(arm_a: dict, arm_b: dict) -> tuple[list[dict], list[dict]]:
    by_qid_a = {r["_qid"]: r for r in arm_a["rows"]}
    by_qid_b = {r["_qid"]: r for r in arm_b["rows"]}
    common = sorted(set(by_qid_a) & set(by_qid_b))
    return [by_qid_a[q] for q in common], [by_qid_b[q] for q in common]


def _build_delta(arm_a: dict, arm_b: dict, iters: int) -> dict[str, Any]:
    rows_a, rows_b = _align_rows(arm_a, arm_b)
    paired_n = len(rows_a)
    if paired_n == 0:
        return {"paired_n": 0}

    metric_fns = {
        "grounded": lambda r: 1 if r.get("grounded") else 0,
        "citation": lambda r: 1 if r.get("has_citation") else 0,
        "retry": lambda r: 1 if r.get("retry") else 0,
    }
    out: dict[str, Any] = {"paired_n": paired_n}
    for key, fn in metric_fns.items():
        a = [fn(r) for r in rows_a]
        b = [fn(r) for r in rows_b]
        out[key] = {
            "arm_a_rate": sum(a) / paired_n,
            "arm_b_rate": sum(b) / paired_n,
            "delta": sum(b) / paired_n - sum(a) / paired_n,
            "bootstrap": _paired_bootstrap_ci(a, b, iters),
        }

    def _recall5(r: dict) -> int:
        exp = set(r.get("expected_doc_ids") or [])
        ret = (r.get("retrieved_doc_ids") or [])[:5]
        return 1 if (exp and any(d in exp for d in ret)) else 0

    a_r = [_recall5(r) for r in rows_a]
    b_r = [_recall5(r) for r in rows_b]
    out["recall@5"] = {
        "arm_a_rate": sum(a_r) / paired_n,
        "arm_b_rate": sum(b_r) / paired_n,
        "delta": sum(b_r) / paired_n - sum(a_r) / paired_n,
        "bootstrap": _paired_bootstrap_ci(a_r, b_r, iters),
    }
    return out


def _build_md(summary: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# HyDE on/off A/B Test\n")
    L.append(f"> 측정일: {summary['meta']['started_at']}  ")
    L.append(f"> Golden Set: `{summary['meta']['golden']}`  ")
    L.append(f"> Paired n: **{summary['delta'].get('paired_n', 0)}**  ")
    L.append(f"> Bootstrap iters: {summary['meta']['bootstrap']}\n")
    L.append("---\n")

    L.append("## 1. Arm 요약 (paired bootstrap)\n")
    L.append("| 메트릭 | A (HyDE on) | B (HyDE off) | Δ (B−A) | 95% CI | p(Δ≤0) |")
    L.append("|--------|-------------|---------------|---------|--------|--------|")
    delta = summary.get("delta", {})
    for key, label in [
        ("recall@5", "retrieval recall@5"),
        ("grounded", "grounded rate"),
        ("citation", "citation accuracy"),
        ("retry", "retry rate"),
    ]:
        d = delta.get(key)
        if not d:
            continue
        bs = d["bootstrap"]
        L.append(
            f"| {label} | {d['arm_a_rate']:.3f} | {d['arm_b_rate']:.3f} | "
            f"{d['delta']:+.3f} | [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}] | "
            f"{bs['p_le_0']:.3f} |"
        )
    L.append("")

    L.append("## 2. Retrieval 분리 메트릭 (Arm 별)\n")
    L.append("| 메트릭 | A (on) | B (off) | Δ |")
    L.append("|--------|--------|---------|---|")
    ra = summary["arm_a"]["retrieval"].get("overall", {})
    rb = summary["arm_b"]["retrieval"].get("overall", {})
    for k in ("recall@5", "recall@10", "mrr", "ndcg@5"):
        a = ra.get(k, 0.0)
        b = rb.get(k, 0.0)
        L.append(f"| {k} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    L.append("")

    L.append("## 3. Generation 메트릭 (Arm 별)\n")
    L.append("| 메트릭 | A (on) | B (off) | Δ |")
    L.append("|--------|--------|---------|---|")
    ga = summary["arm_a"]["generation"]
    gb = summary["arm_b"]["generation"]
    for k in ("grounded_rate", "citation", "retry_rate", "claim_faithfulness"):
        a = ga.get(k)
        b = gb.get(k)
        if a is None or b is None:
            continue
        L.append(f"| {k} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    L.append("")

    L.append("## 4. 판정 가이드\n")
    L.append("- `p(Δ≤0)` 가 작을수록 (≤ 0.05) HyDE off 가 통계적으로 더 나음.")
    L.append("- `p(Δ≤0)` 가 클수록 (≥ 0.95) HyDE on 이 더 나음.")
    L.append("- 그 사이는 차이 미미 — 비용·지연 측면에서 HyDE 비활성화 검토 가능.")
    L.append("")

    L.append("## 5. 산출물\n")
    L.append("- `reports/ab_test_hyde.json` — 모든 메트릭 + per-row")
    L.append("- `reports/ab_test_hyde.md` — 이 보고서")
    L.append("")
    return "\n".join(L)


async def main_async(args: argparse.Namespace) -> int:
    p = _resolve_golden(args.golden)
    df = pd.read_parquet(p)
    if "curated" in df.columns and df["curated"].any():
        df = df[df["curated"]].reset_index(drop=True)
        log.info(f"using {len(df)} curated rows")
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    log.info(f"A/B testing over {len(df)} samples")

    started = datetime.now(timezone.utc).astimezone()

    arm_a = await _run_arm(
        "A_hyde_on", df, hyde_enabled=True, use_claim=not args.no_claim_faithfulness
    )
    arm_b = await _run_arm(
        "B_hyde_off", df, hyde_enabled=False, use_claim=not args.no_claim_faithfulness
    )

    delta = _build_delta(arm_a, arm_b, iters=args.bootstrap)

    summary = {
        "meta": {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "golden": str(p),
            "n": len(df),
            "bootstrap": args.bootstrap,
            "config": {
                "hybrid_method": settings.hybrid_method,
                "hybrid_cc_weight": settings.hybrid_cc_weight,
                "reranker_enabled": settings.reranker_enabled,
            },
        },
        "arm_a": arm_a,
        "arm_b": arm_b,
        "delta": delta,
    }

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_test_hyde.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "ab_test_hyde.md").write_text(_build_md(summary), encoding="utf-8")
    log.info("wrote reports/ab_test_hyde.json + .md")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--golden",
        default=str(PROJECT_ROOT / "data" / "golden_curated_v1.parquet"),
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-claim-faithfulness", action="store_true")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
