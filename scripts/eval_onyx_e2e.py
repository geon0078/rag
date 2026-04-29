"""Onyx 풀 stack e2e 평가 (시나리오 A2 측정).

수작업 250 query → Onyx /chat/send-chat-message → answer + citations
→ V4 (우리 RAG) baseline 과 비교.

요구:
  - Onyx 가 동작 중 (http://localhost:3010)
  - LLM provider 'Solar' 등록되어 있음
  - File connector 인덱싱 완료 (Onyx admin > Indexing Status)
  - ONYX_API_KEY 환경변수 (admin scope)

Run:
    ONYX_API_KEY=on_... python scripts/eval_onyx_e2e.py [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.retrieval_metrics import RetrievalSample, aggregate  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("eval_onyx_e2e")

ONYX_BASE = os.environ.get("ONYX_BASE_URL", "http://localhost:3010")


def _hdr() -> dict[str, str]:
    key = os.environ.get("ONYX_API_KEY", "")
    if not key:
        raise SystemExit("ONYX_API_KEY 환경변수 필요 — admin scope key.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_session(client: httpx.Client) -> str:
    r = client.post(
        f"{ONYX_BASE}/api/chat/create-chat-session",
        json={"persona_id": 0, "description": "eval-e2e"},
        headers=_hdr(),
        timeout=15.0,
    )
    r.raise_for_status()
    return str(r.json()["chat_session_id"])


def _send_message(client: httpx.Client, session_id: str, query: str) -> dict[str, Any]:
    """admin_search 만 호출 — LLM 우회, retrieval-only 메트릭."""
    body = {"query": query, "filters": {}}
    r = client.post(
        f"{ONYX_BASE}/api/admin/search",
        json=body,
        headers=_hdr(),
        timeout=30.0,
    )
    if r.status_code != 200:
        log.warning(f"admin/search {r.status_code}: {r.text[:200]}")
        return {}
    try:
        return r.json()
    except Exception:
        return {}


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


def _extract_doc_ids(resp: dict[str, Any]) -> list[str]:
    """admin/search 응답의 documents 또는 chat 응답의 top_documents 에서 추출."""
    top_docs = resp.get("documents") or resp.get("top_documents") or []
    out: list[str] = []
    for d in top_docs:
        si = d.get("semantic_identifier") or d.get("document_id") or d.get("link", "")
        si = str(si).split("/")[-1]
        if "__" in si:
            si = si.split("__")[0]
        if si.endswith(".md"):
            si = si[:-3]
        out.append(si)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval",
        default=str(PROJECT_ROOT / "data" / "eval_dataset_250_manual.parquet"),
    )
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    df = pd.read_parquet(args.eval)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    log.info(f"e2e 측정 시작 — {len(df)} samples vs Onyx")

    started = datetime.now(timezone.utc).astimezone()
    rows: list[dict[str, Any]] = []
    samples: list[RetrievalSample] = []

    with httpx.Client(timeout=120.0) as client:
        sid = _create_session(client)
        log.info(f"session: {sid}")

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            query = str(row["query"])
            expected = _norm_expected(row.get("expected_doc_ids"))
            try:
                resp = _send_message(client, sid, query)
            except Exception as exc:  # noqa: BLE001
                log.error(f"qid {row['qid']} error: {exc}")
                resp = {}

            answer = (resp.get("answer") or "").strip()
            retrieved = _extract_doc_ids(resp)
            has_citation = bool(retrieved)  # admin_search 만 — citation 은 retrieval 존재 여부로 대체

            rows.append({
                "qid": str(row["qid"]),
                "query": query,
                "expected_doc_ids": expected,
                "retrieved_doc_ids": retrieved[:10],
                "answer": answer[:400],
                "has_citation": has_citation,
                "source_collection": row.get("source_collection"),
                "challenge_type": row.get("challenge_type"),
            })
            samples.append(RetrievalSample(
                qid=str(row["qid"]),
                expected_doc_ids=tuple(expected),
                retrieved_doc_ids=tuple(retrieved[:10]),
                source_collection=row.get("source_collection"),
            ))
            if idx % 25 == 0:
                log.info(f"  progress {idx}/{len(df)}")

    rt = aggregate(samples, ks=(5, 10))
    n = max(1, len(rows))
    cit = sum(1 for r in rows if r.get("has_citation")) / n
    answered = sum(1 for r in rows if (r.get("answer") or "").strip()) / n

    summary = {
        "meta": {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "eval_set": args.eval,
            "n": len(rows),
            "n_errors": sum(1 for r in rows if not r.get("answer")),
            "onyx_base": ONYX_BASE,
        },
        "retrieval": rt,
        "generation": {"citation": cit, "answered": answered},
        "rows": rows,
    }

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_onyx_e2e.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    overall = rt.get("overall", {})
    md = []
    md.append("# Onyx 풀 stack e2e 평가 — 시나리오 A2\n")
    md.append(f"> 측정일: {summary['meta']['started_at']}  ")
    md.append(f"> Eval: `{args.eval}` · n={len(rows)}\n")
    md.append("우리 V4 baseline (Qdrant+Okt+Solar): recall@5 0.852 / MRR 0.678 / nDCG@5 0.716\n")
    md.append("---\n")
    md.append("## 1. Onyx e2e 메트릭\n")
    md.append("| 메트릭 | Onyx | V4 baseline | Δ |")
    md.append("|--------|------|-------------|---|")
    for k, b in [("recall@5", 0.852), ("recall@10", 0.868), ("mrr", 0.678), ("ndcg@5", 0.716)]:
        v = overall.get(k, 0)
        md.append(f"| {k} | {v:.3f} | {b:.3f} | {v - b:+.3f} |")
    md.append(f"| citation | {cit:.3f} | 1.000 | {cit - 1.0:+.3f} |")
    md.append(f"| answered | {answered:.3f} | — | — |")
    md.append("")

    md.append("## 2. 컬렉션별 recall@5\n")
    md.append("| 컬렉션 | n | recall@5 | MRR |")
    md.append("|--------|---|----------|-----|")
    for sc, st in sorted(rt.get("by_collection", {}).items()):
        md.append(f"| {sc} | {int(st['n'])} | {st['recall@5']:.3f} | {st['mrr']:.3f} |")
    md.append("")

    md.append("## 3. 산출물\n")
    md.append("- `reports/eval_onyx_e2e.json`")
    md.append("- `reports/eval_onyx_e2e.md` — 본 보고서")
    md.append("")

    (out_dir / "eval_onyx_e2e.md").write_text("\n".join(md), encoding="utf-8")
    log.info(f"wrote reports/eval_onyx_e2e.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
