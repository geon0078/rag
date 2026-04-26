"""D1-D5 retrieval diagnostics for Phase 5 regression debugging.

Maps directly to the user's 5-point Option-2 debug plan:
- D1: payload_lookup structure dump (flattened? campus key direct access?)
- D2: BM25 search() with metadata_filter for 학칙 chunks
- D3: 20 empty-context queries re-searched without campus filter
- D4: 6 campus_filter failures — leaked chunks' campus values
- D5: dense/sparse independently — campus distribution per query

Run: python scripts/diagnose_retrieval.py
Output: stdout (UTF-8) + reports/diagnose_retrieval.json
"""

from __future__ import annotations

import asyncio
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.retrieval.bm25_okt import OktBM25  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.router import _build_filter, _build_sparse_filter, route  # noqa: E402

REPORT_PATH = ROOT / "reports" / "diagnose_retrieval.json"
EVAL_JSON = ROOT / "reports" / "eval_supplementary.json"
QA_PARQUET = ROOT / "data" / "qa.parquet"
BM25_PKL = ROOT / "data" / "bm25_okt.pkl"


def section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def d1_payload_structure(report: dict[str, Any]) -> OktBM25:
    section("D1: payload_lookup structure")
    with open(BM25_PKL, "rb") as f:
        raw = pickle.load(f)
    payload_lookup: dict[str, dict[str, Any]] = raw.get("payload_lookup", {})
    print(f"schema_version={raw.get('schema_version')} doc_count={len(raw['doc_ids'])}")
    print(f"payload_lookup entries: {len(payload_lookup)}")
    sample_ids = list(payload_lookup.keys())[:3]
    for did in sample_ids:
        p = payload_lookup[did]
        print(f"\n  doc_id={did}")
        print(f"    keys={list(p.keys())}")
        print(f"    campus={p.get('campus')!r} (type={type(p.get('campus')).__name__})")
        print(f"    source_collection={p.get('source_collection')!r}")
    hakchik_ids = [
        did
        for did, p in payload_lookup.items()
        if p.get("source_collection") == "학칙_조항"
    ]
    print(f"\n학칙_조항 doc_count: {len(hakchik_ids)}")
    if hakchik_ids:
        sample_p = payload_lookup[hakchik_ids[0]]
        print(f"  first 학칙 doc_id={hakchik_ids[0]}")
        print(f"  campus={sample_p.get('campus')!r}")
    campus_counter = Counter(p.get("campus") for p in payload_lookup.values())
    print(f"\nCampus distribution: {dict(campus_counter)}")

    bm25 = OktBM25()
    bm25.load()

    report["D1"] = {
        "schema_version": raw.get("schema_version"),
        "doc_count": len(raw["doc_ids"]),
        "payload_count": len(payload_lookup),
        "campus_distribution": {str(k): v for k, v in campus_counter.items()},
        "hakchik_count": len(hakchik_ids),
        "hakchik_first_campus": payload_lookup[hakchik_ids[0]].get("campus")
        if hakchik_ids
        else None,
        "sample_keys": list(payload_lookup[sample_ids[0]].keys()) if sample_ids else [],
    }
    return bm25


def d2_bm25_filter_test(bm25: OktBM25, qa: pd.DataFrame, report: dict[str, Any]) -> None:
    section("D2: BM25 search() with campus filter on 학칙 queries")
    hakchik_qa = qa[qa["source_collection"] == "학칙_조항"].head(3)
    results: list[dict[str, Any]] = []
    for _, row in hakchik_qa.iterrows():
        q = row["query"]
        qid = row["qid"]
        print(f"\n  qid={qid[:8]} query={q[:60]}")
        variants = {
            "no_filter": None,
            "campus_성남_only": {"campus": "성남"},
            "campus_성남_or_전체": {"campus": ["성남", "전체"]},
        }
        per_query: dict[str, Any] = {"qid": qid, "query": q, "variants": {}}
        for label, mf in variants.items():
            hits = bm25.search(q, top_k=5, metadata_filter=mf)
            cols = []
            campuses = []
            for did, _ in hits:
                p = bm25.get_payload(did)
                cols.append(p.get("source_collection"))
                campuses.append(p.get("campus"))
            print(f"    {label}: {len(hits)} hits collections={cols} campuses={campuses}")
            per_query["variants"][label] = {
                "n_hits": len(hits),
                "collections": cols,
                "campuses": campuses,
            }
        results.append(per_query)
    report["D2"] = results


def d3_empty_contexts_research(
    bm25: OktBM25, qa: pd.DataFrame, eval_data: dict[str, Any], report: dict[str, Any]
) -> None:
    section("D3: 20 empty-contexts queries — refetch without campus filter")
    routing_failures = eval_data["routing"]["failures"]
    empty_qids = [f["qid"] for f in routing_failures if f.get("got_top3") == []][:20]
    qa_lookup = qa.set_index("qid").to_dict("index")
    results: list[dict[str, Any]] = []
    for qid in empty_qids:
        if qid not in qa_lookup:
            continue
        row = qa_lookup[qid]
        q = row["query"]
        expected_col = row["source_collection"]
        decision = route(q)
        sparse_with = bm25.search(q, top_k=5, metadata_filter=decision.sparse_filter)
        sparse_no = bm25.search(q, top_k=5, metadata_filter=None)
        with_cols = [bm25.get_payload(d).get("source_collection") for d, _ in sparse_with]
        no_cols = [bm25.get_payload(d).get("source_collection") for d, _ in sparse_no]
        with_match = expected_col in with_cols
        no_match = expected_col in no_cols
        print(
            f"  qid={qid[:8]} expected={expected_col} "
            f"campus={decision.campus} "
            f"with_filter_match={with_match} no_filter_match={no_match}"
        )
        results.append(
            {
                "qid": qid,
                "expected": expected_col,
                "campus": decision.campus,
                "sparse_filter": decision.sparse_filter,
                "with_filter_match": with_match,
                "no_filter_match": no_match,
                "with_filter_cols": with_cols,
                "no_filter_cols": no_cols,
            }
        )
    n_with = sum(1 for r in results if r["with_filter_match"])
    n_no = sum(1 for r in results if r["no_filter_match"])
    print(f"\nSummary: with_filter={n_with}/{len(results)} no_filter={n_no}/{len(results)}")
    report["D3"] = {
        "n_total": len(results),
        "n_with_filter_match": n_with,
        "n_no_filter_match": n_no,
        "details": results,
    }


def d4_campus_filter_leaks(
    bm25: OktBM25, qa: pd.DataFrame, eval_data: dict[str, Any], report: dict[str, Any]
) -> None:
    section("D4: campus_filter 6 failures — leaked chunks' campus values")
    failures = eval_data["campus_filter"]["failures"]
    qa_lookup = qa.set_index("qid").to_dict("index")
    results: list[dict[str, Any]] = []
    for f in failures:
        qid = f["qid"]
        expected = f["expected"]
        if qid not in qa_lookup:
            continue
        q = qa_lookup[qid]["query"]
        decision = route(q)
        hits = bm25.search(q, top_k=5, metadata_filter=decision.sparse_filter)
        chunk_data = []
        for did, score in hits:
            p = bm25.get_payload(did)
            chunk_data.append(
                {
                    "doc_id": did,
                    "campus": p.get("campus"),
                    "source_collection": p.get("source_collection"),
                    "score": score,
                }
            )
        print(
            f"\n  qid={qid[:8]} expected_campus={expected} "
            f"router_campus={decision.campus} sparse_filter={decision.sparse_filter}"
        )
        for c in chunk_data:
            print(
                f"    campus={c['campus']!r} "
                f"col={c['source_collection']} score={c['score']:.3f}"
            )
        results.append(
            {
                "qid": qid,
                "expected": expected,
                "router_campus": decision.campus,
                "sparse_filter": decision.sparse_filter,
                "chunks": chunk_data,
            }
        )
    report["D4"] = results


def d5_dense_sparse_separately(
    qa: pd.DataFrame, eval_data: dict[str, Any], report: dict[str, Any]
) -> None:
    section("D5: dense/sparse independently — campus distribution")
    retriever = HybridRetriever()
    routing_failures = eval_data["routing"]["failures"]
    sample_qids = [f["qid"] for f in routing_failures[:5]]
    qa_lookup = qa.set_index("qid").to_dict("index")
    results: list[dict[str, Any]] = []
    for qid in sample_qids:
        if qid not in qa_lookup:
            continue
        q = qa_lookup[qid]["query"]
        expected = qa_lookup[qid]["source_collection"]
        decision = route(q)
        dense_hits = retriever._dense(q, k=10, decision=decision)
        sparse_hits = retriever._sparse(q, k=10, decision=decision)
        dense_campuses = Counter(h["payload"].get("campus") for h in dense_hits)
        dense_cols = Counter(h["payload"].get("source_collection") for h in dense_hits)
        sparse_campuses = Counter(
            retriever.bm25.get_payload(d).get("campus") for d, _ in sparse_hits
        )
        sparse_cols = Counter(
            retriever.bm25.get_payload(d).get("source_collection") for d, _ in sparse_hits
        )
        print(f"\n  qid={qid[:8]} expected={expected} campus={decision.campus}")
        print(f"    dense({len(dense_hits)}) campuses={dict(dense_campuses)} cols={dict(dense_cols)}")
        print(
            f"    sparse({len(sparse_hits)}) "
            f"campuses={dict(sparse_campuses)} cols={dict(sparse_cols)}"
        )
        results.append(
            {
                "qid": qid,
                "expected": expected,
                "campus": decision.campus,
                "dense_n": len(dense_hits),
                "dense_campuses": {str(k): v for k, v in dense_campuses.items()},
                "dense_cols": {str(k): v for k, v in dense_cols.items()},
                "sparse_n": len(sparse_hits),
                "sparse_campuses": {str(k): v for k, v in sparse_campuses.items()},
                "sparse_cols": {str(k): v for k, v in sparse_cols.items()},
            }
        )
    report["D5"] = results


async def main() -> None:
    with open(EVAL_JSON, encoding="utf-8") as f:
        eval_data = json.load(f)
    qa = pd.read_parquet(QA_PARQUET)
    report: dict[str, Any] = {}

    bm25 = d1_payload_structure(report)
    d2_bm25_filter_test(bm25, qa, report)
    d3_empty_contexts_research(bm25, qa, eval_data, report)
    d4_campus_filter_leaks(bm25, qa, eval_data, report)
    d5_dense_sparse_separately(qa, eval_data, report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
