"""Outline → V4 인덱스 (Qdrant + BM25) 재구축.

흐름:
  1. Outline 모든 문서 fetch (documents.list + documents.info)
  2. 페이지 → 청크 분할 (## 섹션 기준, 600자 / 100자 overlap)
  3. Solar 임베딩 (4096-dim) → Qdrant collection 'euljiu_outline'
  4. Okt BM25 인덱스 → data/bm25_outline.pkl
  5. data/outline_chunks.parquet 청크 백업
  6. data/outline_url_map.json 업데이트

Run:
    python scripts/index_outline.py [--limit N] [--collection NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid as uuid_lib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTLINE_BASE = os.environ.get("OUTLINE_URL", "http://localhost:3002").rstrip("/")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
SOLAR_BASE = "https://api.upstage.ai/v1/solar"

NEW_COLLECTION = os.environ.get("OUTLINE_QDRANT_COLLECTION", "euljiu_outline")
EMBED_MODEL = "embedding-passage"
EMBED_DIM = 4096
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
EMBED_BATCH = 50

CHUNKS_OUT = PROJECT_ROOT / "data" / "outline_chunks.parquet"
BM25_OUT = PROJECT_ROOT / "data" / "bm25_outline.pkl"


def _outline_hdr() -> dict[str, str]:
    tok = os.environ.get("OUTLINE_TOKEN", "")
    if not tok:
        path = PROJECT_ROOT / "tmp" / "outline_api_token.txt"
        if path.exists():
            tok = path.read_text(encoding="utf-8").strip()
    if not tok:
        raise SystemExit("OUTLINE_TOKEN 필요")
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}


def _solar_key() -> str:
    key = os.environ.get("UPSTAGE_API_KEY", "")
    if not key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("UPSTAGE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise SystemExit("UPSTAGE_API_KEY 필요")
    return key


def list_collections(client: httpx.Client) -> dict[str, dict[str, str]]:
    r = client.post(
        f"{OUTLINE_BASE}/api/collections.list",
        headers=_outline_hdr(),
        json={"limit": 100},
        timeout=15.0,
    )
    r.raise_for_status()
    return {c["id"]: c for c in (r.json().get("data") or [])}


def list_documents(client: httpx.Client) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    offset = 0
    while True:
        r = client.post(
            f"{OUTLINE_BASE}/api/documents.list",
            headers=_outline_hdr(),
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


def get_doc(client: httpx.Client, doc_id: str) -> dict[str, Any]:
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.info",
        headers=_outline_hdr(),
        json={"id": doc_id},
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json().get("data") or {}


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_SECTION_RE = re.compile(r"^(##\s+.+)$", re.MULTILINE)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    block = m.group(1)
    fm: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    body = text[m.end():]
    return fm, body


def split_into_sections(body: str) -> list[tuple[str, str]]:
    parts = _SECTION_RE.split(body)
    out: list[tuple[str, str]] = []
    if not parts:
        return [("(intro)", body.strip())]
    if parts[0].strip():
        out.append(("(intro)", parts[0].strip()))
    for i in range(1, len(parts), 2):
        header = parts[i].lstrip("# ").strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            out.append((header, content))
    return out


def chunk_section(section: str, content: str, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(content) <= max_size:
        return [content]
    chunks = []
    i = 0
    while i < len(content):
        end = min(i + max_size, len(content))
        nl = content.rfind("\n", i, end)
        sp = content.rfind(" ", i, end)
        cut = nl if nl > i + max_size // 2 else sp if sp > i + max_size // 2 else end
        chunks.append(content[i:cut].strip())
        if cut >= len(content):
            break
        i = max(cut - overlap, i + 1)
    return [c for c in chunks if c]


def _infer_type(col_name: str | None) -> str:
    return {
        "학칙": "regulation",
        "강의평가": "lecture_review",
        "학사일정": "calendar",
        "학사정보": "academic_info",
        "시설": "facility",
        "장학금": "scholarship",
        "학과정보": "department",
        "기타": "misc",
    }.get(col_name or "", "misc")


def _build_parent_path(doc: dict[str, Any], by_id: dict[str, dict[str, Any]], col_name: str | None) -> list[str]:
    """root → ... → doc 의 title 체인 (collection 포함). doc 자신은 제외."""
    chain: list[str] = []
    cur = doc
    visited: set[str] = set()
    while cur and cur.get("parentDocumentId"):
        pid = cur["parentDocumentId"]
        if pid in visited:
            break
        visited.add(pid)
        parent = by_id.get(pid)
        if not parent:
            break
        chain.append(parent.get("title", ""))
        cur = parent
    chain.reverse()
    if col_name:
        chain.insert(0, col_name)
    return [c for c in chain if c]


def build_chunks(
    doc: dict[str, Any],
    col_name: str | None,
    outline_url: str | None,
    by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    text = doc.get("text", "") or ""
    if not text.strip():
        return []
    fm, body = parse_frontmatter(text)
    title = doc.get("title", "")
    parent = f"outline_{doc['id']}"
    parent_titles = _build_parent_path(doc, by_id or {}, col_name)
    breadcrumb = " > ".join(parent_titles + [title]) if parent_titles else title
    chunks: list[dict[str, Any]] = []
    sections = split_into_sections(body)
    if not sections:
        sections = [("(intro)", body.strip())]
    for sec_name, sec_text in sections:
        for piece in chunk_section(sec_name, sec_text):
            idx = len(chunks)
            inner = f"## {sec_name}\n{piece}" if sec_name and sec_name != "(intro)" else piece
            # breadcrumb 를 chunk 본문 앞에 prepend — semantic 매칭에서 부모 키워드 활용
            content = f"[{breadcrumb}]\n{inner}"
            chunks.append({
                "doc_id": f"{parent}_c{idx}",
                "parent_doc_id": parent,
                "contents": content,
                "metadata": {
                    "outline_doc_id": doc["id"],
                    "outline_url": outline_url,
                    "collection": col_name,
                    "title": title,
                    "breadcrumb": breadcrumb,
                    "parent_titles": parent_titles,
                    "outline_parent_id": doc.get("parentDocumentId"),
                    "chunk_section": sec_name,
                    "chunk_index": idx,
                    "campus": fm.get("campus", "전체"),
                    "type": _infer_type(col_name),
                    "frontmatter": fm or None,
                },
            })
    return chunks


def embed_batch(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    r = client.post(
        f"{SOLAR_BASE}/embeddings",
        headers={"Authorization": f"Bearer {_solar_key()}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json().get("data") or []
    return [item["embedding"] for item in data]


def qdrant_recreate_collection(client: httpx.Client) -> None:
    client.delete(f"{QDRANT_URL}/collections/{NEW_COLLECTION}", timeout=30.0)
    r = client.put(
        f"{QDRANT_URL}/collections/{NEW_COLLECTION}",
        json={"vectors": {"size": EMBED_DIM, "distance": "Cosine"}},
        timeout=30.0,
    )
    r.raise_for_status()
    print(f"[qdrant] collection '{NEW_COLLECTION}' recreated (dim={EMBED_DIM})")


def qdrant_upsert(client: httpx.Client, points: list[dict[str, Any]]) -> None:
    r = client.put(
        f"{QDRANT_URL}/collections/{NEW_COLLECTION}/points?wait=true",
        json={"points": points},
        timeout=120.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"qdrant upsert 실패 {r.status_code}: {r.text[:300]}")


def main() -> int:
    global NEW_COLLECTION
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--collection", type=str, default=NEW_COLLECTION)
    args = p.parse_args()
    NEW_COLLECTION = args.collection
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    with httpx.Client(timeout=60.0) as outline_client, \
         httpx.Client(timeout=60.0) as qdrant_client, \
         httpx.Client(timeout=120.0) as solar_client:

        cols = list_collections(outline_client)
        col_name_by_id = {cid: c["name"] for cid, c in cols.items()}
        print(f"[fetch] {len(cols)} collections")

        url_map_path = PROJECT_ROOT / "data" / "outline_url_map.json"
        url_map = (
            json.loads(url_map_path.read_text(encoding="utf-8"))
            if url_map_path.exists() else {"by_corpus_doc_id": {}}
        )

        docs_meta = list_documents(outline_client)
        if args.limit:
            docs_meta = docs_meta[: args.limit]
        print(f"[fetch] {len(docs_meta)} documents 메타")

        # documents.list 응답에 text 필드가 이미 포함됨 (Onyx Outline connector 와 동일).
        # by_id: parent traversal 용 — chunk 에 breadcrumb (parent path) 주입.
        by_id = {d["id"]: d for d in docs_meta}
        all_chunks: list[dict[str, Any]] = []
        outline_url_by_uuid: dict[str, str] = {}
        for i, doc in enumerate(docs_meta, start=1):
            url = doc.get("url")
            outline_url = f"{OUTLINE_BASE}{url}" if url else None
            if outline_url:
                outline_url_by_uuid[doc["id"]] = outline_url
            col_name = col_name_by_id.get(doc.get("collectionId", ""))
            chunks = build_chunks(doc, col_name, outline_url, by_id=by_id)
            all_chunks.extend(chunks)
            if i % 50 == 0:
                print(f"  processed {i}/{len(docs_meta)} (chunks so far: {len(all_chunks)})")
        print(f"[chunk] 생성된 청크: {len(all_chunks)}")

        qdrant_recreate_collection(qdrant_client)

        print(f"[embed] Solar embedding (batch={EMBED_BATCH})")
        for batch_start in range(0, len(all_chunks), EMBED_BATCH):
            batch = all_chunks[batch_start : batch_start + EMBED_BATCH]
            texts = [c["contents"][:7000] for c in batch]
            embs = embed_batch(solar_client, texts)
            points = []
            for c, vec in zip(batch, embs):
                pid = str(uuid_lib.uuid4())
                points.append({
                    "id": pid,
                    "vector": vec,
                    "payload": {
                        "doc_id": c["doc_id"],
                        "parent_doc_id": c["parent_doc_id"],
                        "contents": c["contents"],
                        **(c["metadata"] or {}),
                    },
                })
            qdrant_upsert(qdrant_client, points)
            done = min(batch_start + EMBED_BATCH, len(all_chunks))
            print(f"  embed+upsert {done}/{len(all_chunks)}")

    print(f"[save] chunks parquet")
    import pandas as pd
    rows = [
        {
            "doc_id": c["doc_id"],
            "parent_doc_id": c["parent_doc_id"],
            "contents": c["contents"],
            "metadata": c["metadata"],
        }
        for c in all_chunks
    ]
    pd.DataFrame(rows).to_parquet(CHUNKS_OUT, index=False)
    print(f"  saved {CHUNKS_OUT}")

    print(f"[bm25] Okt 토크나이저로 BM25 인덱스 빌드")
    try:
        from src.retrieval.bm25_okt import OktBM25  # type: ignore
    except ImportError as exc:
        print(f"  [skip] OktBM25 import 실패 (V4 src 마운트 필요): {exc}")
    else:
        bm25 = OktBM25()
        doc_ids = [c["doc_id"] for c in all_chunks]
        contents = [c["contents"] for c in all_chunks]
        payloads = [c["metadata"] or {} for c in all_chunks]
        bm25.build(doc_ids, contents, payloads)
        bm25.save(BM25_OUT)
        print(f"  saved {BM25_OUT} ({len(doc_ids)} docs)")

    url_map_new = {
        "by_outline_doc_uuid": outline_url_by_uuid,
        **{k: v for k, v in url_map.items() if k != "by_outline_doc_uuid"},
    }
    url_map_path.write_text(
        json.dumps(url_map_new, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] started={started}  finished={datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print(f"  collection: {NEW_COLLECTION}, chunks: {len(all_chunks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
