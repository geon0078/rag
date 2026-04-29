"""Outline 데이터셋 audit — 중복 + 구조 + Q&A 패턴 분석.

흐름:
  1. POST /api/documents.list (paginate) — 모든 문서 메타
  2. POST /api/documents.export — 본문 markdown
  3. 분석:
     - 동일/유사 title (정규화 매칭)
     - content sha256 동일
     - 짧은 내용 (50자 미만) — 빈 페이지 후보
     - cross-collection 주제 키워드 (학식당·도서관·기숙사 등)
     - Q&A 패턴 ("Q. ... A. ...", "**질문:**", "**답변:**")
     - 예시 답변 패턴 ("예를 들면", "예시:", "참고")
  4. tmp/outline_audit.json 작성 — 후속 dedup/restructure 입력

Run:
    python scripts/audit_outline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "tmp" / "outline_audit.json"
OUTLINE_BASE = os.environ.get("OUTLINE_URL", "http://localhost:3002").rstrip("/")


def _hdr() -> dict[str, str]:
    tok = os.environ.get("OUTLINE_TOKEN", "")
    if not tok:
        path = PROJECT_ROOT / "tmp" / "outline_api_token.txt"
        if path.exists():
            tok = path.read_text(encoding="utf-8").strip()
    if not tok:
        raise SystemExit("OUTLINE_TOKEN env 또는 tmp/outline_api_token.txt 필요")
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}


_CROSS_TOPICS = {
    "학식당": ["학식", "학식당", "식당", "급식", "뉴밀레니엄"],
    "도서관": ["도서관", "열람실", "도서대출"],
    "기숙사": ["기숙사", "생활관", "외박", "외출"],
    "수강신청": ["수강신청", "수강바구니", "수강 정정"],
    "장학금": ["장학금", "장학"],
    "졸업요건": ["졸업요건", "졸업학점", "졸업인증"],
    "체육관": ["체육관", "헬스장", "운동시설"],
    "휴학": ["휴학", "복학"],
    "교통": ["통학버스", "셔틀", "주차"],
    "증명서": ["증명서", "성적증명", "재학증명"],
}

_QA_PATTERNS = [
    re.compile(r"^Q\.\s", re.MULTILINE),
    re.compile(r"^A\.\s", re.MULTILINE),
    re.compile(r"\*\*질문:\*\*"),
    re.compile(r"\*\*답변:\*\*"),
    re.compile(r"## Q\."),
]

_EXAMPLE_PATTERNS = [
    re.compile(r"예를\s*들면"),
    re.compile(r"예시\s*[::]"),
    re.compile(r"참고\s*[::]"),
    re.compile(r"^>\s+예"),
    re.compile(r"예컨대"),
]


def list_collections(client: httpx.Client) -> dict[str, dict[str, str]]:
    r = client.post(f"{OUTLINE_BASE}/api/collections.list",
                    headers=_hdr(), json={"limit": 100}, timeout=15.0)
    r.raise_for_status()
    out: dict[str, dict[str, str]] = {}
    for c in r.json().get("data") or []:
        out[c["id"]] = {"name": c["name"], "id": c["id"]}
    return out


def list_documents(client: httpx.Client) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    offset = 0
    while True:
        r = client.post(
            f"{OUTLINE_BASE}/api/documents.list",
            headers=_hdr(),
            json={"limit": 100, "offset": offset},
            timeout=20.0,
        )
        r.raise_for_status()
        page = r.json().get("data") or []
        if not page:
            break
        docs.extend(page)
        offset += len(page)
        if len(page) < 100:
            break
    return docs


def export_doc(client: httpx.Client, doc_id: str) -> str:
    """documents.export 가 종종 빈 응답이라 documents.info.text 를 사용."""
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.info",
        headers=_hdr(),
        json={"id": doc_id},
        timeout=15.0,
    )
    if r.status_code != 200:
        return ""
    data = r.json().get("data") or {}
    return data.get("text") or ""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def title_key(title: str) -> str:
    return re.sub(r"\s+", "", (title or "")).lower()


def count_qa(text: str) -> int:
    return sum(len(p.findall(text)) for p in _QA_PATTERNS)


def count_examples(text: str) -> int:
    return sum(len(p.findall(text)) for p in _EXAMPLE_PATTERNS)


def main() -> int:
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with httpx.Client(timeout=60.0) as client:
        cols = list_collections(client)
        print(f"[audit] {len(cols)} collections")
        docs_meta = list_documents(client)
        print(f"[audit] {len(docs_meta)} documents 메타 수집")

        docs: list[dict[str, Any]] = []
        for i, d in enumerate(docs_meta, start=1):
            md = export_doc(client, d["id"])
            text = normalize(md)
            docs.append({
                "id": d["id"],
                "title": d.get("title", ""),
                "collection_id": d.get("collectionId"),
                "collection": cols.get(d.get("collectionId", ""), {}).get("name"),
                "text_len": len(text),
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "qa_markers": count_qa(md),
                "example_markers": count_examples(md),
                "url_id": d.get("urlId"),
                "raw_preview": text[:300],
            })
            if i % 25 == 0:
                print(f"  export {i}/{len(docs_meta)}")
            time.sleep(0.05)

    title_groups: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        title_groups[title_key(d["title"])].append(d["id"])
    duplicate_titles = [
        {"key": k, "ids": v, "titles": [next(d["title"] for d in docs if d["id"] == x) for x in v]}
        for k, v in title_groups.items() if len(v) > 1
    ]

    hash_groups: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        hash_groups[d["text_hash"]].append(d["id"])
    duplicate_content = [
        {"hash": k, "ids": v, "titles": [next(d["title"] for d in docs if d["id"] == x) for x in v]}
        for k, v in hash_groups.items() if len(v) > 1
    ]

    too_short = [
        {"id": d["id"], "title": d["title"], "collection": d["collection"], "len": d["text_len"]}
        for d in docs if d["text_len"] < 50
    ]

    cross_topics: dict[str, list[dict[str, Any]]] = {}
    for topic, kws in _CROSS_TOPICS.items():
        hits = []
        for d in docs:
            blob = (d["title"] + " " + d["raw_preview"]).lower()
            if any(k in blob for k in kws):
                hits.append({"id": d["id"], "title": d["title"], "collection": d["collection"]})
        if len({h["collection"] for h in hits}) > 1:
            cross_topics[topic] = hits

    by_col: dict[str, dict[str, int]] = {}
    for d in docs:
        c = d["collection"] or "?"
        if c not in by_col:
            by_col[c] = {"docs": 0, "total_chars": 0, "qa_pages": 0, "example_pages": 0}
        by_col[c]["docs"] += 1
        by_col[c]["total_chars"] += d["text_len"]
        if d["qa_markers"] > 0:
            by_col[c]["qa_pages"] += 1
        if d["example_markers"] > 0:
            by_col[c]["example_pages"] += 1

    qa_heavy = sorted(
        [d for d in docs if d["qa_markers"] >= 2],
        key=lambda x: -x["qa_markers"],
    )[:30]
    example_heavy = sorted(
        [d for d in docs if d["example_markers"] >= 2],
        key=lambda x: -x["example_markers"],
    )[:30]

    summary = {
        "audited_at": started,
        "totals": {
            "collections": len(cols),
            "documents": len(docs),
            "duplicate_titles": len(duplicate_titles),
            "duplicate_content": len(duplicate_content),
            "too_short": len(too_short),
            "cross_collection_topics": len(cross_topics),
            "qa_heavy_pages": len([d for d in docs if d["qa_markers"] >= 2]),
            "example_heavy_pages": len([d for d in docs if d["example_markers"] >= 2]),
        },
        "by_collection": by_col,
        "duplicate_titles": duplicate_titles[:30],
        "duplicate_content": duplicate_content[:30],
        "too_short": too_short[:30],
        "cross_collection_topics": cross_topics,
        "qa_heavy": [
            {k: v for k, v in d.items() if k != "raw_preview"} for d in qa_heavy
        ],
        "example_heavy": [
            {k: v for k, v in d.items() if k != "raw_preview"} for d in example_heavy
        ],
        "docs": [{k: v for k, v in d.items() if k != "raw_preview"} for d in docs],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[audit] saved {OUT}")
    print(f"  - duplicate titles: {len(duplicate_titles)}")
    print(f"  - duplicate content: {len(duplicate_content)}")
    print(f"  - too short (<50 chars): {len(too_short)}")
    print(f"  - cross-collection topics: {list(cross_topics.keys())}")
    print(f"  - Q&A heavy pages: {summary['totals']['qa_heavy_pages']}")
    print(f"  - example heavy pages: {summary['totals']['example_heavy_pages']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
