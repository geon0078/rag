"""Aggregate Solar judgments into a human-readable diagnosis.

Inputs:
    reports/quality_judgments.jsonl   (from judge_traces.py)
    logs/quality_traces.jsonl         (full trace, used for context in samples)
    data/qa_adversarial.parquet       (to keep bad-case rows for regression)

Outputs:
    reports/quality_diagnosis.md      (the report a human reads)
    reports/quality_diagnosis.json    (machine-readable mirror)
    data/qa_bad_cases.parquet         (rows with quality_score < 3, for regression)

Run:
    python scripts/aggregate_diagnosis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

log = get_logger("aggregate_diagnosis")


FAILURE_MODE_DESC = {
    "OK": "만족 (quality_score ≥ 4)",
    "A": "Retrieval 실패 — 정답 doc 미검색",
    "B": "Groundedness 과거부 — 검색 OK인데 judge가 거부",
    "C": "Context noise — 잡음 doc이 답변을 흐림",
    "D": "Generation 부실 — 질문 핵심 비켜감 (relevance ↓)",
    "E": "Fallback 과반응 — 부적절한 거부",
    "ERR": "Judge 호출 실패",
}

FIX_GUIDE = {
    "A": "router CAMPUS_PATTERN/synonym 보강, hybrid_cc_weight 재튜닝, query expansion 도입 검토",
    "B": "groundedness judge prompt 완화, multi-hop relax 패턴 추가",
    "C": "passage_reranker 도입(GPU 가용 시), top_k_final 축소, 컨텍스트 정렬 prompt 추가",
    "D": "answer prompt에 '질문 키워드를 첫 문장에 직접 응답' 강제, 답변 후 질문↔답변 cosine 유사도 검증",
    "E": "fallback 트리거 임계값 재튜닝, two-stage retry (top-10 확장 후 재시도) 도입",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in obj:
                continue
            out.append(obj)
    return out


def _meta_of(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in obj:
                return obj["_meta"]
            return {}
    return {}


def _trace_index(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {t["qid"]: t for t in traces if "qid" in t}


def _quality_histogram(judgments: list[dict[str, Any]]) -> dict[int, int]:
    counts = Counter()
    for j in judgments:
        s = j.get("quality_score") or 0
        counts[s] += 1
    return dict(sorted(counts.items()))


def _format_histogram(hist: dict[int, int], total: int) -> str:
    lines = ["| Score | Count | % | Bar |", "|-------|-------|---|-----|"]
    for score in (5, 4, 3, 2, 1, 0):
        c = hist.get(score, 0)
        pct = (c / total * 100) if total else 0.0
        bar = "█" * int(round(pct / 2))
        label = "ERR" if score == 0 else str(score)
        lines.append(f"| {label} | {c} | {pct:.1f}% | `{bar}` |")
    return "\n".join(lines)


def _samples_per_mode(judgments: list[dict[str, Any]],
                      trace_idx: dict[str, dict[str, Any]],
                      n: int = 5) -> dict[str, list[dict[str, Any]]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for j in judgments:
        mode = j.get("failure_mode") or "ERR"
        if mode == "OK":
            continue
        if len(by_mode[mode]) >= n:
            continue
        qid = j.get("qid")
        trace = trace_idx.get(qid, {}).get("trace", {})
        by_mode[mode].append({
            "qid": qid,
            "query": j.get("query"),
            "challenge_type": j.get("challenge_type"),
            "quality_score": j.get("quality_score"),
            "diagnosis": j.get("diagnosis"),
            "suggested_fix": j.get("suggested_fix"),
            "evidence": j.get("evidence"),
            "verdict": trace.get("verdict"),
            "retry": trace.get("retry"),
            "retrieval_hit": trace.get("retrieval_hit"),
            "answer_preview": (trace.get("answer") or "")[:200],
            "expected_preview": (trace_idx.get(qid, {}).get("expected_gt") or "")[:200],
            "top_sources": [s.get("doc_id") for s in (trace.get("sources") or [])[:3]],
            "retrieval_gt": trace_idx.get(qid, {}).get("retrieval_gt"),
        })
    return by_mode


def _by_challenge_type(judgments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_ct = defaultdict(list)
    for j in judgments:
        by_ct[j.get("challenge_type") or "?"].append(j)
    for ct, group in sorted(by_ct.items()):
        n = len(group)
        n_ok = sum(1 for j in group if j.get("failure_mode") == "OK")
        avg = sum((j.get("quality_score") or 0) for j in group) / n if n else 0.0
        out[ct] = {
            "n": n,
            "ok_rate": n_ok / n if n else 0.0,
            "avg_score": avg,
            "mode_counts": dict(Counter(j.get("failure_mode") or "ERR" for j in group)),
        }
    return out


def _aggregate_fixes(judgments: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Frequency-weighted suggested_fix list (top 10)."""
    counts: Counter[str] = Counter()
    for j in judgments:
        fix = (j.get("suggested_fix") or "").strip()
        if fix:
            counts[fix] += 1
    return list(counts.most_common(10))


def _build_md(meta: dict[str, Any],
              total: int,
              hist: dict[int, int],
              mode_counts: dict[str, int],
              by_ct: dict[str, dict[str, Any]],
              samples: dict[str, list[dict[str, Any]]],
              top_fixes: list[tuple[str, int]]) -> str:
    lines: list[str] = []
    lines.append("# RAG Quality Diagnosis (Solar LLM-as-judge)\n")
    lines.append(f"> 측정일: {meta.get('started_at', '—')}  ")
    lines.append(f"> Judge model: `{meta.get('judge_model', 'solar-pro3')}`  ")
    lines.append(f"> 평가 traces: **{total}** (OK={meta.get('n_ok', 0)}, ERR={meta.get('n_err', 0)})\n")
    lines.append("---\n")

    lines.append("## 1. Quality Score 분포\n")
    lines.append(_format_histogram(hist, total))
    avg = sum(s * c for s, c in hist.items() if s) / max(1, total - hist.get(0, 0))
    lines.append(f"\n**평균 quality_score**: {avg:.2f} / 5.00 (ERR 제외)\n")
    lines.append("---\n")

    lines.append("## 2. 실패 유형별 분포\n")
    lines.append("| Mode | n | % | 설명 |")
    lines.append("|------|---|---|------|")
    for mode in ("OK", "A", "B", "C", "D", "E", "ERR"):
        c = mode_counts.get(mode, 0)
        pct = (c / total * 100) if total else 0.0
        lines.append(f"| **{mode}** | {c} | {pct:.1f}% | {FAILURE_MODE_DESC[mode]} |")
    lines.append("")
    lines.append("---\n")

    lines.append("## 3. Challenge Type 별 성능\n")
    lines.append("| Challenge | n | OK rate | avg score | mode 분포 |")
    lines.append("|-----------|---|---------|-----------|------------|")
    for ct, stats in by_ct.items():
        modes = ", ".join(f"{k}:{v}" for k, v in sorted(stats["mode_counts"].items()))
        lines.append(f"| {ct} | {stats['n']} | {stats['ok_rate']*100:.1f}% | {stats['avg_score']:.2f} | {modes} |")
    lines.append("")
    lines.append("---\n")

    lines.append("## 4. 실패 유형별 대표 사례 (각 5건)\n")
    for mode in ("A", "B", "C", "D", "E"):
        lst = samples.get(mode, [])
        if not lst:
            continue
        lines.append(f"### Mode {mode} — {FAILURE_MODE_DESC[mode]}\n")
        lines.append(f"**권장 fix 방향**: {FIX_GUIDE.get(mode, '—')}\n")
        for s in lst:
            lines.append(f"- **Q** ({s['challenge_type']}, score={s['quality_score']}): {s['query']}")
            lines.append(f"  - verdict={s['verdict']} retry={s['retry']} retrieval_hit={s['retrieval_hit']}")
            lines.append(f"  - 정답 doc: {s.get('retrieval_gt')} / 검색 top3: {s.get('top_sources')}")
            lines.append(f"  - expected: `{s['expected_preview']}`")
            lines.append(f"  - answer:   `{s['answer_preview']}`")
            lines.append(f"  - diagnosis: {s['diagnosis']}")
            lines.append(f"  - fix: {s['suggested_fix']}")
            lines.append("")
    lines.append("---\n")

    lines.append("## 5. 빈도 기반 Top Fix 제안 (Solar 자동 추출)\n")
    if top_fixes:
        lines.append("| 빈도 | 제안 |")
        lines.append("|------|------|")
        for fix, c in top_fixes:
            lines.append(f"| {c} | {fix} |")
    else:
        lines.append("_(suggested_fix 없음)_")
    lines.append("")
    lines.append("---\n")

    lines.append("## 6. 산출물\n")
    lines.append("- `reports/quality_diagnosis.md` — 이 보고서")
    lines.append("- `reports/quality_diagnosis.json` — 기계 판독용 통계")
    lines.append("- `reports/quality_judgments.jsonl` — row-level 판정")
    lines.append("- `logs/quality_traces.jsonl` — row-level trace")
    lines.append("- `data/qa_adversarial.parquet` — adversarial 평가셋")
    lines.append("- `data/qa_bad_cases.parquet` — quality_score < 3 행 (회귀 테스트용)")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", default=str(PROJECT_ROOT / "reports" / "quality_judgments.jsonl"))
    p.add_argument("--traces", default=str(PROJECT_ROOT / "logs" / "quality_traces.jsonl"))
    p.add_argument("--adversarial", default=str(PROJECT_ROOT / "data" / "qa_adversarial.parquet"))
    p.add_argument("--out-md", default=str(PROJECT_ROOT / "reports" / "quality_diagnosis.md"))
    p.add_argument("--out-json", default=str(PROJECT_ROOT / "reports" / "quality_diagnosis.json"))
    p.add_argument("--bad-cases", default=str(PROJECT_ROOT / "data" / "qa_bad_cases.parquet"))
    args = p.parse_args()

    judgments = _load_jsonl(Path(args.judgments))
    traces = _load_jsonl(Path(args.traces))
    meta = _meta_of(Path(args.judgments))
    if not judgments:
        log.error("no judgments found")
        return 1

    log.info(f"loaded judgments={len(judgments)} traces={len(traces)}")
    trace_idx = _trace_index(traces)

    total = len(judgments)
    hist = _quality_histogram(judgments)
    mode_counts = dict(Counter(j.get("failure_mode") or "ERR" for j in judgments))
    by_ct = _by_challenge_type(judgments)
    samples = _samples_per_mode(judgments, trace_idx, n=5)
    top_fixes = _aggregate_fixes(judgments)

    md = _build_md(meta, total, hist, mode_counts, by_ct, samples, top_fixes)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md, encoding="utf-8")
    log.info(f"wrote {args.out_md}")

    machine = {
        "meta": meta,
        "total": total,
        "histogram": hist,
        "mode_counts": mode_counts,
        "by_challenge_type": by_ct,
        "samples": samples,
        "top_fixes": [{"fix": f, "count": c} for f, c in top_fixes],
    }
    Path(args.out_json).write_text(
        json.dumps(machine, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    log.info(f"wrote {args.out_json}")

    bad_qids = {
        j["qid"]
        for j in judgments
        if (j.get("quality_score") or 0)
        and j["quality_score"] < 3
        and j.get("failure_mode") != "ERR"
    }
    if Path(args.adversarial).exists() and bad_qids:
        adv = pd.read_parquet(args.adversarial)
        bad = adv[adv["qid"].isin(bad_qids)].reset_index(drop=True)
        Path(args.bad_cases).parent.mkdir(parents=True, exist_ok=True)
        bad.to_parquet(args.bad_cases, index=False)
        log.info(f"wrote {len(bad)} bad cases -> {args.bad_cases}")
    else:
        log.info("no bad cases to persist (or adversarial parquet missing)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
