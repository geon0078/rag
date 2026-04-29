"""Adversarial 250 평가 — V4 + Outline 인덱스 (옵션 b).

흐름:
  1. eval_dataset_250_manual.parquet 로드 (250 queries)
  2. 각 query → V4 /v1/chat/completions (Onyx LLM provider 와 동일 경로)
  3. 응답의 비표준 citations 필드에서 outline doc UUID 추출
  4. expected_doc_ids (corpus 시대 ID) → outline UUID 매핑 변환
  5. recall@5/10, MRR, nDCG@5/10, citation rate, answered rate 계산
  6. reports/eval_outline_250.{json,md}

Run:
    python scripts/eval_outline_250.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.retrieval_metrics import RetrievalSample, aggregate  # noqa: E402

V4_BASE = "http://localhost:8000"
EVAL_DATASET = PROJECT_ROOT / "data" / "eval_dataset_250_manual.parquet"
URL_MAP = PROJECT_ROOT / "data" / "outline_url_map.json"
OUT_JSON = PROJECT_ROOT / "reports" / "eval_outline_250.json"
OUT_MD = PROJECT_ROOT / "reports" / "eval_outline_250.md"

V4_BASELINE = {
    "recall@5": 0.852, "recall@10": 0.868, "mrr": 0.678, "ndcg@5": 0.716,
}
ONYX_NOMIC_BASELINE = {
    "recall@5": 0.454, "recall@10": 0.518, "mrr": 0.368, "ndcg@5": 0.380,
}


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


def build_corpus_to_uuid(url_map: dict[str, Any]) -> dict[str, str]:
    """corpus_doc_id → outline_uuid (via url 매칭)."""
    by_corpus = url_map.get("by_corpus_doc_id") or {}
    by_uuid = url_map.get("by_outline_doc_uuid") or {}
    url_to_uuid = {url: uuid for uuid, url in by_uuid.items()}
    return {
        corpus_id: url_to_uuid[url]
        for corpus_id, url in by_corpus.items()
        if url in url_to_uuid
    }


def extract_uuid_from_citation(citation_doc_id: str) -> str | None:
    """outline_<uuid>_c<idx> → uuid."""
    m = re.match(r"outline_([a-f0-9-]{36})", citation_doc_id or "")
    return m.group(1) if m else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    df = pd.read_parquet(EVAL_DATASET)
    if args.limit:
        df = df.head(args.limit)
    print(f"[eval] {len(df)} queries")

    url_map = json.loads(URL_MAP.read_text(encoding="utf-8"))
    corpus_to_uuid = build_corpus_to_uuid(url_map)
    print(f"[eval] corpus→outline_uuid 매핑: {len(corpus_to_uuid)}")

    started = datetime.now(timezone.utc).astimezone()
    rows: list[dict[str, Any]] = []
    samples: list[RetrievalSample] = []

    with httpx.Client(timeout=180.0) as client:
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            qid = str(row["qid"])
            query = str(row["query"])
            expected_corpus = _norm_expected(row.get("expected_doc_ids"))
            expected_uuid = sorted({
                corpus_to_uuid[c] for c in expected_corpus if c in corpus_to_uuid
            })

            try:
                r = client.post(
                    f"{V4_BASE}/v1/chat/completions",
                    json={
                        "model": "solar-pro",
                        "messages": [{"role": "user", "content": query}],
                        "stream": False,
                    },
                    timeout=180.0,
                )
                resp = r.json() if r.status_code == 200 else {}
            except Exception as exc:  # noqa: BLE001
                print(f"  qid {qid} 오류: {exc}")
                resp = {}

            answer = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            citations = resp.get("citations") or []
            retrieved_uuids: list[str] = []
            seen = set()
            for c in citations:
                uid = extract_uuid_from_citation(c.get("doc_id", ""))
                if uid and uid not in seen:
                    retrieved_uuids.append(uid)
                    seen.add(uid)

            has_citation = bool(answer and ("출처" in answer or retrieved_uuids))
            rows.append({
                "qid": qid,
                "query": query,
                "expected_corpus": expected_corpus,
                "expected_uuid": expected_uuid,
                "retrieved_uuid": retrieved_uuids[:10],
                "answer": answer[:300],
                "has_citation": has_citation,
                "source_collection": row.get("source_collection"),
                "challenge_type": row.get("challenge_type"),
            })
            samples.append(RetrievalSample(
                qid=qid,
                expected_doc_ids=tuple(expected_uuid),
                retrieved_doc_ids=tuple(retrieved_uuids[:10]),
                source_collection=row.get("source_collection"),
            ))
            if idx % 25 == 0:
                print(f"  progress {idx}/{len(df)}")

    rt = aggregate(samples, ks=(5, 10))
    n = max(1, len(rows))
    cit = sum(1 for r in rows if r.get("has_citation")) / n
    answered = sum(1 for r in rows if (r.get("answer") or "").strip()) / n

    overall = rt.get("overall", {})
    summary = {
        "meta": {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "eval_set": str(EVAL_DATASET),
            "n": len(rows),
            "v4_endpoint": f"{V4_BASE}/v1/chat/completions",
            "config": {
                "qdrant_collection": "euljiu_outline",
                "bm25_path": "data/bm25_outline.pkl",
                "chunks": 2387,
            },
        },
        "retrieval": rt,
        "generation": {"citation": cit, "answered": answered},
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    md = []
    md.append("# Adversarial 250 — V4 + Outline 인덱스\n")
    md.append(f"> 측정일: {summary['meta']['started_at']} · n={len(rows)}\n")
    md.append("V4 baseline (corpus.parquet): recall@5 0.852 / MRR 0.678 / nDCG@5 0.716\n")
    md.append("Onyx-nomic baseline: recall@5 0.454 / MRR 0.368\n")
    md.append("---\n\n## 1. 메트릭 비교\n")
    md.append("| 메트릭 | V4+Outline | V4 (legacy) | Δ vs V4 | Onyx nomic | Δ vs nomic |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for k in ("recall@5", "recall@10", "mrr", "ndcg@5"):
        v = overall.get(k, 0)
        v4 = V4_BASELINE.get(k, 0)
        ny = ONYX_NOMIC_BASELINE.get(k, 0)
        md.append(f"| {k} | {v:.3f} | {v4:.3f} | {v-v4:+.3f} | {ny:.3f} | {v-ny:+.3f} |")
    md.append(f"| citation | {cit:.3f} | 1.000 | {cit-1.0:+.3f} | 1.000 | {cit-1.0:+.3f} |")
    md.append(f"| answered | {answered:.3f} | — | — | — | — |")
    md.append("")
    md.append("## 2. 컬렉션별 recall@5\n")
    md.append("| 컬렉션 | n | recall@5 | MRR |")
    md.append("|---|---:|---:|---:|")
    for sc, st in sorted(rt.get("by_collection", {}).items()):
        md.append(f"| {sc} | {int(st['n'])} | {st['recall@5']:.3f} | {st['mrr']:.3f} |")
    md.append("")
    md.append("## 3. 산출물\n")
    md.append("- `reports/eval_outline_250.json`")
    md.append("- `reports/eval_outline_250.md`")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[eval] saved {OUT_JSON}, {OUT_MD}")
    print(f"  recall@5={overall.get('recall@5',0):.3f} | mrr={overall.get('mrr',0):.3f} | citation={cit:.3f} | answered={answered:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
