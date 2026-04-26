"""Combine AutoRAG + RAGAS + supplementary evals into reports/final_report.md.

Reads (all optional, missing inputs render as "missing"):
    benchmark/0/summary.csv         - AutoRAG champion modules per node
    reports/ragas_summary.json      - RAGAS aggregate scores
    reports/eval_supplementary.json - 4 supplementary metric blocks

Writes:
    reports/final_report.md

Run:
    python scripts/generate_eval_report.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_API_KEY_RE = re.compile(r"(['\"]?(?:api_key|api-key|apiKey|UPSTAGE_API_KEY)['\"]?\s*[:=]\s*['\"])([^'\"]+)(['\"])")
_BARE_UPSTAGE_KEY_RE = re.compile(r"\bup_[A-Za-z0-9]{10,}\b")


def _redact(text: str) -> str:
    """Strip API keys before writing to the markdown report."""
    text = _API_KEY_RE.sub(r"\1***REDACTED***\3", text)
    text = _BARE_UPSTAGE_KEY_RE.sub("***REDACTED***", text)
    return text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

log = get_logger("generate_eval_report")

AUTORAG_SUMMARY = PROJECT_ROOT / "benchmark" / "0" / "summary.csv"
RAGAS_SUMMARY = PROJECT_ROOT / "reports" / "ragas_summary.json"
SUPPLEMENTARY = PROJECT_ROOT / "reports" / "eval_supplementary.json"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "final_report.md"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        log.warning(f"missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_autorag(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning(f"missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.error(f"failed to read {path}: {exc}")
        return None


def _autorag_section(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "## 1. AutoRAG 자동 평가\n\n_summary.csv 없음 (`autorag evaluate` 미실행)_\n"
    lines = [
        "## 1. AutoRAG 자동 평가\n",
        "| node_line | node_type | best_module | filename | params |",
        "|---|---|---|---|---|",
    ]

    def _pick(row, *candidates: str) -> str:
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return str(row[c])
        return "?"

    for _, row in df.iterrows():
        node_line = _pick(row, "node_line_name", "node_line")
        node = _pick(row, "node_type")
        module = _pick(row, "best_module_name", "best_module")
        filename = _pick(row, "best_module_filename", "filename")
        params = _redact(_pick(row, "best_module_params", "module_params"))
        # collapse newlines so the row stays on one markdown line
        params = params.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {node_line} | {node} | {module} | {filename} | {params} |")
    return "\n".join(lines) + "\n"


def _ragas_section(summary: dict | None) -> str:
    if not summary:
        return "## 2. RAGAS 평가\n\n_ragas_summary.json 없음 (`evaluate_ragas.py` 미실행)_\n"
    lines = ["## 2. RAGAS 평가\n", "| 메트릭 | 점수 |", "|---|---|"]
    for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if k in summary:
            lines.append(f"| {k} | {summary[k]:.3f} |")
    if "n" in summary:
        lines.append(f"| n (샘플 수) | {summary['n']} |")
    return "\n".join(lines) + "\n"


def _supp_section(supp: dict | None) -> str:
    if not supp:
        return "## 3. 추가 검증 4종\n\n_eval_supplementary.json 없음 (`eval_supplementary.py` 미실행)_\n"
    rows = [
        (
            "Negative 거절률",
            supp.get("negative", {}).get("rejection_rate"),
            supp.get("negative", {}).get("target"),
            supp.get("negative", {}).get("passed"),
        ),
        (
            "캠퍼스 필터 정확도",
            supp.get("campus_filter", {}).get("accuracy"),
            supp.get("campus_filter", {}).get("target"),
            supp.get("campus_filter", {}).get("passed"),
        ),
        (
            "라우팅 top-3 정확도",
            supp.get("routing", {}).get("accuracy"),
            supp.get("routing", {}).get("target"),
            supp.get("routing", {}).get("passed"),
        ),
        (
            "출처 인용 형식",
            supp.get("citation", {}).get("accuracy"),
            supp.get("citation", {}).get("target"),
            supp.get("citation", {}).get("passed"),
        ),
    ]
    lines = ["## 3. 추가 검증 4종\n", "| 메트릭 | 실제 | 목표 | 통과 |", "|---|---|---|---|"]
    for label, actual, target, passed in rows:
        actual_s = f"{actual:.3f}" if isinstance(actual, (int, float)) else "—"
        target_s = f"{target:.2f}" if isinstance(target, (int, float)) else "—"
        flag = "✅" if passed else ("❌" if passed is False else "—")
        lines.append(f"| {label} | {actual_s} | {target_s} | {flag} |")
    return "\n".join(lines) + "\n"


def _verdict_section(ragas: dict | None, supp: dict | None) -> str:
    rows = []
    if ragas:
        f = ragas.get("faithfulness")
        rows.append(("RAGAS faithfulness", f, 0.85, f is not None and f >= 0.85))
    else:
        rows.append(("RAGAS faithfulness", None, 0.85, None))

    if supp:
        neg = supp.get("negative", {}).get("rejection_rate")
        rows.append(("Negative 거절률", neg, 0.80, neg is not None and neg >= 0.80))
        camp = supp.get("campus_filter", {}).get("accuracy")
        rows.append(("캠퍼스 필터 정확도", camp, 1.00, camp is not None and camp >= 1.00))
        rt = supp.get("routing", {}).get("accuracy")
        rows.append(("카테고리 라우팅 정확도", rt, 0.95, rt is not None and rt >= 0.95))

    if not rows:
        return "## 4. 통과 기준 검증\n\n_평가 결과 없음_\n"

    lines = ["## 4. 통과 기준 검증\n", "| 지표 | 실제 | 목표 | 통과 |", "|---|---|---|---|"]
    all_passed = True
    for label, actual, target, passed in rows:
        actual_s = f"{actual:.3f}" if isinstance(actual, (int, float)) else "—"
        flag = "✅" if passed else ("❌" if passed is False else "⚠️")
        if passed is not True:
            all_passed = False
        lines.append(f"| {label} | {actual_s} | ≥ {target} | {flag} |")
    lines.append("")
    lines.append(
        f"**전체 결과: {'✅ PASS' if all_passed else '❌ FAIL — 미달 항목 확인 필요'}**\n"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    autorag_df = _load_autorag(AUTORAG_SUMMARY)
    ragas = _load_json(RAGAS_SUMMARY)
    supp = _load_json(SUPPLEMENTARY)

    parts: list[str] = []
    parts.append("# RAG 시스템 평가 리포트")
    parts.append("")
    parts.append(f"평가 일시: {datetime.now().isoformat(timespec='seconds')}")
    parts.append("")
    parts.append(_autorag_section(autorag_df))
    parts.append(_ragas_section(ragas))
    parts.append(_supp_section(supp))
    parts.append(_verdict_section(ragas, supp))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    log.info(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
