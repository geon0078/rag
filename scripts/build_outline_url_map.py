"""corpus.doc_id → Outline document URL 매핑 빌드.

용도: V4 답변의 [출처: doc_id, ...] 마커를 클릭 가능한 markdown 링크로
변환하기 위한 lookup 테이블 사전 생성.

흐름:
  1. tmp/docmost-migration/manifest.json — page (title, file) 목록
  2. tmp/outline_migration_result.json — file → outline_uuid 매핑
  3. Outline API: POST /api/documents.info — outline_uuid → urlId
  4. data/corpus.parquet — corpus chunk metadata 로드해 page_title 또는
     group_key 로 매핑 추론
  5. 결과: tmp/outline_url_map.json

Run:
    python scripts/build_outline_url_map.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "tmp" / "docmost-migration" / "manifest.json"
RESULT = PROJECT_ROOT / "tmp" / "outline_migration_result.json"
CORPUS = PROJECT_ROOT / "data" / "corpus.parquet"
OUT = PROJECT_ROOT / "tmp" / "outline_url_map.json"

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


def _doc_url(client: httpx.Client, outline_uuid: str) -> str | None:
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.info",
        headers=_hdr(),
        json={"id": outline_uuid},
        timeout=15.0,
    )
    if r.status_code != 200:
        return None
    d = r.json().get("data") or {}
    url = d.get("url")
    return f"{OUTLINE_BASE}{url}" if url else None


def _corpus_to_groups() -> dict[str, dict[str, Any]]:
    """corpus.doc_id → {source_collection, page_title, topic_id} 추론.

    migrate_to_docmost.py 의 group_chunks() 와 동일 규칙 사용.
    """
    import pandas as pd

    df = pd.read_parquet(CORPUS)
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        meta = r["metadata"] if isinstance(r["metadata"], dict) else dict(r["metadata"])
        doc_id = str(r["doc_id"])
        sc = meta.get("source_collection", "기타")
        title_raw = str(meta.get("title") or "")
        subcategory = meta.get("subcategory") or "(분류 없음)"

        if sc == "학칙_조항":
            m = re.match(r"제(\d+)조", title_raw)
            article_no = m.group(1) if m else None
            page_title = title_raw
            topic_id = (
                f"topic_school_rules_article_{article_no}" if article_no else None
            )
        elif sc == "강의평가":
            m = re.match(r"lecture_reviews_(\d+)", doc_id)
            lid = m.group(1) if m else None
            page_title = title_raw
            topic_id = f"topic_lecture_lecture_reviews_{lid}" if lid else None
        elif sc == "학사일정":
            m = re.search(r"(\d{4})학년도\s*(\d)학기", title_raw)
            sem = f"{m.group(1)}-{m.group(2)}" if m else "미분류"
            page_title = (
                f"{sem[:4]}학년도 {sem[-1]}학기"
                if "-" in sem
                else "학사일정 (기타)"
            )
            topic_id = f"topic_calendar_{sem}"
        else:
            page_title = subcategory if subcategory != "(분류 없음)" else sc
            topic_id = None
        out[doc_id] = {
            "source_collection": sc,
            "page_title": page_title,
            "topic_id": topic_id,
        }
    return out


def main() -> int:
    if not MANIFEST.exists() or not RESULT.exists():
        raise SystemExit("manifest.json 또는 outline_migration_result.json 없음")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    file_to_uuid: dict[str, str] = result.get("documents") or {}
    print(f"[build] {len(file_to_uuid)} outline documents")

    uuid_to_url: dict[str, str] = {}
    with httpx.Client(timeout=30.0) as client:
        for i, (file, uuid) in enumerate(file_to_uuid.items(), start=1):
            url = _doc_url(client, uuid)
            if url:
                uuid_to_url[uuid] = url
            if i % 25 == 0:
                print(f"  url 조회 {i}/{len(file_to_uuid)}")
            time.sleep(0.05)
    print(f"[build] {len(uuid_to_url)} URLs 조회됨")

    by_title: dict[str, str] = {}
    by_topic_id: dict[str, str] = {}
    file_path_to_url: dict[str, str] = {}
    for page in manifest.get("pages") or []:
        f = page.get("file")
        title = page.get("title")
        topic = page.get("topic_id")
        uuid = file_to_uuid.get(f)
        if uuid:
            url = uuid_to_url.get(uuid)
            if url:
                file_path_to_url[f] = url
                if title:
                    by_title[title] = url
                if topic:
                    by_topic_id[topic] = url

    corpus_groups = _corpus_to_groups()
    by_corpus_doc_id: dict[str, str] = {}
    miss = 0
    for doc_id, info in corpus_groups.items():
        url = None
        if info["topic_id"]:
            url = by_topic_id.get(info["topic_id"])
        if not url and info["page_title"]:
            url = by_title.get(info["page_title"])
        if url:
            by_corpus_doc_id[doc_id] = url
        else:
            miss += 1
    print(
        f"[build] corpus mapping: {len(by_corpus_doc_id)} 매핑, {miss} miss "
        f"({miss/len(corpus_groups)*100:.1f}%)"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "by_corpus_doc_id": by_corpus_doc_id,
                "by_title": by_title,
                "by_topic_id": by_topic_id,
                "by_file_path": file_path_to_url,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[build] saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
