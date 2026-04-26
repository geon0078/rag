"""Citation post-processor.

The system prompt instructs the LLM to end answers with
``[출처: doc_id, 카테고리, 캠퍼스]``. Solar occasionally drops or reformats this
line (~1/163 queries in the AutoRAG eval). ``ensure_citation()`` enforces the
contract deterministically: if the answer already has a recognizable citation
marker, leave it alone; otherwise append one synthesized from the top retrieved
doc.

Functions are pure — no LLM call — so they're cheap to run on every reply.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

CITATION_PATTERN = re.compile(r"\[출처\s*[::]")


def _format_one(payload: dict[str, Any], doc_id: str | None) -> str:
    cat = payload.get("category") or payload.get("source_collection") or "출처"
    campus = payload.get("campus") or "전체"
    doc = doc_id or payload.get("doc_id") or "?"
    return f"[출처: {doc}, {cat}, {campus}]"


def has_citation(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(answer or ""))


def ensure_citation(answer: str, candidates: Sequence[dict[str, Any]]) -> str:
    """Return the answer with a guaranteed citation suffix.

    - If the answer already carries a ``[출처: ...]`` marker, return as-is.
    - If no candidates were retrieved, return as-is (caller decided not to
      cite — usually a fallback path).
    - Otherwise append a citation synthesized from the top candidate.
    """
    text = (answer or "").rstrip()
    if not text:
        return answer
    if has_citation(text):
        return text
    if not candidates:
        return text
    top = candidates[0]
    payload = top.get("payload") or {}
    citation = _format_one(payload, top.get("doc_id"))
    return f"{text}\n\n{citation}"
