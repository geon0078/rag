"""Retrieval-side evaluation metrics (평가명세서 §4.2).

End-to-end OK rate hides whether failures are retrieval or generation. These
helpers consume Golden Set rows (each with `expected_doc_ids`) and a list of
ranked candidate doc_ids returned by the retriever, and produce the standard
information-retrieval metrics:

  recall@k   — gt 중 top-k 안에 포함된 비율 (k=5, 10)
  hit@k      — top-k에 gt 하나라도 있으면 1, 없으면 0 (binary recall)
  mrr        — 첫 번째 gt 의 역순위 평균 (1/rank)
  ndcg@k     — 순위 가중 정확도 (이상적 순위 대비 비율)

각 메트릭은 카테고리별 분리 측정이 가능하도록 (overall + by_collection)
``aggregate`` 형태로 반환한다.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RetrievalSample:
    """단일 query 의 retrieval 결과 한 행."""

    qid: str
    expected_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    source_collection: str | None = None


def hit_at_k(sample: RetrievalSample, k: int) -> float:
    if not sample.expected_doc_ids:
        return 0.0
    top_k = sample.retrieved_doc_ids[:k]
    return 1.0 if any(g in top_k for g in sample.expected_doc_ids) else 0.0


def recall_at_k(sample: RetrievalSample, k: int) -> float:
    """gt 중 top-k 에 포함된 비율 (multi-hop 도 안전)."""
    gt = set(sample.expected_doc_ids)
    if not gt:
        return 0.0
    top_k = sample.retrieved_doc_ids[:k]
    found = sum(1 for g in gt if g in top_k)
    return found / len(gt)


def mrr(sample: RetrievalSample) -> float:
    """첫 번째 gt 의 역순위 (1-based). 못 찾으면 0."""
    gt = set(sample.expected_doc_ids)
    for idx, doc_id in enumerate(sample.retrieved_doc_ids, start=1):
        if doc_id in gt:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(sample: RetrievalSample, k: int) -> float:
    """nDCG@k. 모든 gt 를 동일 relevance(=1) 로 처리.

    DCG = sum_{i in top-k} rel_i / log2(i+1) (1-based)
    IDCG = sum_{i=1..min(|gt|,k)} 1 / log2(i+1)
    """
    gt = set(sample.expected_doc_ids)
    if not gt:
        return 0.0
    top_k = sample.retrieved_doc_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in gt:
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(gt), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(
    samples: Sequence[RetrievalSample],
    ks: Sequence[int] = (5, 10),
) -> dict[str, dict[str, float]]:
    """Overall + by_collection 메트릭 집계.

    Returns:
        {
            "overall": {"hit@5": .., "recall@5": .., "mrr": .., "ndcg@5": .., "n": int, ...},
            "by_collection": {
                "FAQ": {"hit@5": .., ..., "n": int},
                "학칙_조항": {...},
                ...
            }
        }
    """
    by_col: dict[str | None, list[RetrievalSample]] = defaultdict(list)
    for s in samples:
        by_col[s.source_collection].append(s)

    def _stats(group: Sequence[RetrievalSample]) -> dict[str, float]:
        out: dict[str, float] = {"n": float(len(group))}
        for k in ks:
            out[f"hit@{k}"] = _safe_mean(hit_at_k(s, k) for s in group)
            out[f"recall@{k}"] = _safe_mean(recall_at_k(s, k) for s in group)
            out[f"ndcg@{k}"] = _safe_mean(ndcg_at_k(s, k) for s in group)
        out["mrr"] = _safe_mean(mrr(s) for s in group)
        return out

    overall = _stats(samples)
    by_collection = {sc or "_none": _stats(group) for sc, group in by_col.items()}
    return {"overall": overall, "by_collection": by_collection}
