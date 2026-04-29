"""Solar LLM-as-judge over quality traces.

Reads logs/quality_traces.jsonl (from run_quality_traces.py) and asks Solar
to score each (query, retrieval, answer) triple on a 5-point rubric, classify
the failure mode if any, and propose a concrete fix. Output:
reports/quality_judgments.jsonl, one judgment per trace.

Failure mode taxonomy (must match aggregate_diagnosis):
    OK  satisfactory answer
    A   retrieval missed the gt doc
    B   retrieval hit gt but groundedness judge over-rejected
    C   context noise: relevant + irrelevant docs together confused the LLM
    D   generation answered the wrong question / weak relevance
    E   fallback over-triggered: corpus could answer but pipeline returned
        the safe "정보 없음" response

Run:
    python scripts/judge_traces.py
        [--input logs/quality_traces.jsonl]
        [--output reports/quality_judgments.jsonl]
        [--concurrency 2]
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

from openai import APIError, AsyncOpenAI, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("judge_traces")

JUDGE_MODEL = "solar-pro3"
TEMPERATURE = 0.0
MAX_TOKENS = 500
RETRY_MAX = 5
RETRY_BASE = 2.0


SYSTEM = """당신은 한국어 RAG 시스템 평가자입니다. 사용자 질문, 검색된 출처, 시스템 답변, 그리고 정답을 보고 다음 JSON 한 줄만 출력하세요.

평가 기준 (quality_score 1~5):
5 = 정답과 매우 유사하고 핵심을 정확히 답함
4 = 정답에 가까우나 일부 누락/표현 차이
3 = 부분적으로 맞으나 중요한 정보가 빠지거나 부정확
2 = 답변이 질문에서 벗어나거나 대부분 틀림
1 = 완전히 틀리거나 무의미한 응답

failure_mode (정확히 하나 선택):
"OK"  — quality_score 4 이상이며 만족스러움
"A"   — 검색이 정답 doc을 놓침 (retrieval_hit=false 또는 sources에 정답 doc_id 없음)
"B"   — 검색은 정답 포함했으나 groundedness가 거부해 fallback 됨 (verdict=notGrounded + retrieval_hit=true)
"C"   — sources에 무관/잡음 doc이 섞여 답변이 흐려짐 (정답 doc은 있지만 답에 다른 정보 인용)
"D"   — 검색은 OK이고 답변도 nominally grounded이지만 질문이 묻는 핵심을 비켜감 (relevance 낮음)
"E"   — 답변이 "정보 없음/찾을 수 없습니다" fallback인데 corpus 정답이 있어 부적절 거부

규칙:
- 반드시 JSON 한 줄만 출력. 다른 텍스트 금지.
- diagnosis는 1~2 문장.
- suggested_fix는 구체적 행동 (예: "router CAMPUS_PATTERN에 단축형 추가", "answer prompt에 질문 키워드 직접 응답 강조" 등).

출력 형식:
{"quality_score": <1-5>, "failure_mode": "OK|A|B|C|D|E", "diagnosis": "...", "suggested_fix": "...", "evidence": "<한 줄, 구체적 인용/수치>"}"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "quality_score" not in obj or "failure_mode" not in obj:
        return None
    try:
        obj["quality_score"] = int(obj["quality_score"])
    except (TypeError, ValueError):
        return None
    if obj["quality_score"] not in (1, 2, 3, 4, 5):
        return None
    if obj["failure_mode"] not in {"OK", "A", "B", "C", "D", "E"}:
        return None
    obj.setdefault("diagnosis", "")
    obj.setdefault("suggested_fix", "")
    obj.setdefault("evidence", "")
    return obj


def _build_user_prompt(rec: dict[str, Any]) -> str:
    trace = rec.get("trace", {}) or {}
    sources = trace.get("sources", []) or []
    src_lines = []
    for i, s in enumerate(sources[:5], 1):
        sid = s.get("doc_id") or "?"
        score = s.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        src_lines.append(f"  {i}. {sid} (cat={s.get('category', '—')}, campus={s.get('campus', '—')}, score={score_str})")
    src_block = "\n".join(src_lines) if src_lines else "  (없음)"

    answer = trace.get("answer") or "(빈 답변)"
    if len(answer) > 800:
        answer = answer[:800] + "..."
    expected = rec.get("expected_gt") or "(없음)"
    if len(expected) > 600:
        expected = expected[:600] + "..."

    return (
        f"[질문]\n{rec.get('query','')}\n\n"
        f"[정답 (expected_gt)]\n{expected}\n\n"
        f"[정답 doc_id]\n{', '.join(rec.get('retrieval_gt', []) or ['(없음)'])}\n\n"
        f"[검색된 sources (top-5)]\n{src_block}\n\n"
        f"[시스템 verdict / retry / retrieval_hit]\n"
        f"  verdict={trace.get('verdict','—')}, retry={trace.get('retry')}, "
        f"retrieval_hit={trace.get('retrieval_hit')}, grounded={trace.get('grounded')}\n\n"
        f"[시스템 답변]\n{answer}\n\n"
        "위 정보를 종합해 평가 JSON을 출력하세요."
    )


class Judge:
    def __init__(self, concurrency: int) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.upstage_api_key, base_url=settings.upstage_base_url
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def _chat(self, user: str) -> str:
        async with self._sem:
            attempt = 0
            delay = RETRY_BASE
            while True:
                attempt += 1
                try:
                    resp = await self.client.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    return resp.choices[0].message.content or ""
                except RateLimitError:
                    if attempt >= RETRY_MAX:
                        raise
                    log.warning(f"judge: 429 retry {attempt}/{RETRY_MAX} sleep {delay:.1f}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, 30.0)
                except APIError:
                    if attempt >= 2:
                        raise
                    await asyncio.sleep(delay)

    async def judge(self, rec: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = await self._chat(_build_user_prompt(rec))
        except Exception as exc:  # noqa: BLE001
            return {**_skeleton(rec), "quality_score": 0, "failure_mode": "ERR",
                    "diagnosis": f"judge call failed: {type(exc).__name__}: {exc}",
                    "suggested_fix": "", "evidence": ""}
        parsed = _extract_json(raw)
        if not parsed:
            return {**_skeleton(rec), "quality_score": 0, "failure_mode": "ERR",
                    "diagnosis": f"unparseable judge output: {raw[:120]!r}",
                    "suggested_fix": "", "evidence": ""}
        return {**_skeleton(rec), **parsed}


def _skeleton(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": rec.get("qid"),
        "query": rec.get("query"),
        "challenge_type": rec.get("challenge_type"),
        "source_collection": rec.get("source_collection"),
        "retrieval_hit": (rec.get("trace") or {}).get("retrieval_hit"),
        "verdict": (rec.get("trace") or {}).get("verdict"),
        "retry": (rec.get("trace") or {}).get("retry"),
    }


async def main_async(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    if not inp.exists():
        log.error(f"traces missing: {inp}")
        return 1

    records: list[dict[str, Any]] = []
    with inp.open(encoding="utf-8") as fh:
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
            records.append(obj)
    log.info(f"loaded {len(records)} traces from {inp}")

    judge = Judge(args.concurrency)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).astimezone()

    sem_local = asyncio.Semaphore(args.concurrency)

    async def _bounded(rec: dict[str, Any]) -> dict[str, Any]:
        async with sem_local:
            return await judge.judge(rec)

    results = await asyncio.gather(*(_bounded(r) for r in records))
    n_ok = sum(1 for r in results if r.get("failure_mode") == "OK")
    n_err = sum(1 for r in results if r.get("failure_mode") == "ERR")
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "_meta": {
                "started_at": started.isoformat(timespec="seconds"),
                "judge_model": JUDGE_MODEL,
                "n_records": len(records),
                "n_ok": n_ok,
                "n_err": n_err,
            }
        }, ensure_ascii=False) + "\n")
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"done: total={len(results)} OK={n_ok} ERR={n_err} -> {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(PROJECT_ROOT / "logs" / "quality_traces.jsonl"))
    p.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "quality_judgments.jsonl"))
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
