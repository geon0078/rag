"""tmp/docmost-migration/{space}/{page}.md → Outline 일괄 import.

Flow:
  1. tmp/docmost-migration/manifest.json 읽기 (262 페이지, 8 space).
  2. 각 space 별 Outline Collection 생성 (POST /api/collections.create).
  3. 각 markdown 파일 → POST /api/documents.create (title, text, collectionId, publish:true).
  4. tmp/outline_migration_result.json 에 매핑 저장.

Auth:
  Authorization: Bearer ${OUTLINE_TOKEN}
  (없으면 tmp/outline_api_token.txt 에서 자동 로드)

Run:
  OUTLINE_TOKEN=ol_api_... python scripts/outline_migrate.py
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
MIG_DIR = PROJECT_ROOT / "tmp" / "docmost-migration"
MANIFEST = MIG_DIR / "manifest.json"
RESULT = PROJECT_ROOT / "tmp" / "outline_migration_result.json"

COLOR_MAP = {
    "학칙": "#4E5BD0",
    "강의평가": "#9C2BAB",
    "학사일정": "#0095FF",
    "학사정보": "#00A86B",
    "시설": "#F5BE31",
    "장학금": "#FB6E6E",
    "학과정보": "#7B61FF",
    "기타": "#7E7E80",
}


def _base() -> str:
    return os.environ.get("OUTLINE_URL", "http://localhost:3002").rstrip("/")


def _hdr() -> dict[str, str]:
    tok = os.environ.get("OUTLINE_TOKEN", "")
    if not tok:
        path = PROJECT_ROOT / "tmp" / "outline_api_token.txt"
        if path.exists():
            tok = path.read_text(encoding="utf-8").strip()
    if not tok:
        raise SystemExit("OUTLINE_TOKEN env 또는 tmp/outline_api_token.txt 필요")
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}


def _strip_frontmatter(md: str) -> tuple[str, dict[str, str]]:
    """frontmatter 제거 + 메타 dict 반환."""
    meta: dict[str, str] = {}
    if md.startswith("---\n"):
        end = md.find("\n---\n", 4)
        if end > 0:
            block = md[4:end]
            for line in block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            md = md[end + 5:]
    return md.lstrip("\n"), meta


def list_existing_collections(client: httpx.Client) -> dict[str, str]:
    r = client.post(
        f"{_base()}/api/collections.list",
        headers=_hdr(),
        json={"limit": 100},
        timeout=15.0,
    )
    r.raise_for_status()
    out: dict[str, str] = {}
    for c in (r.json().get("data") or []):
        out[c["name"]] = c["id"]
    return out


def create_collection(client: httpx.Client, name: str, description: str) -> str:
    body = {
        "name": name,
        "description": description,
        "color": COLOR_MAP.get(name, "#4E5BD0"),
        "permission": "read_write",
        "private": False,
    }
    r = client.post(
        f"{_base()}/api/collections.create",
        headers=_hdr(),
        json=body,
        timeout=15.0,
    )
    if r.status_code != 200:
        raise SystemExit(
            f"collection '{name}' 생성 실패 {r.status_code}: {r.text[:300]}"
        )
    cid = r.json()["data"]["id"]
    print(f"[migrate] collection '{name}' created id={cid}")
    return cid


def create_document(
    client: httpx.Client,
    collection_id: str,
    title: str,
    text: str,
    max_retries: int = 4,
) -> str | None:
    body = {
        "collectionId": collection_id,
        "title": title[:100],
        "text": text,
        "publish": True,
    }
    for attempt in range(max_retries):
        r = client.post(
            f"{_base()}/api/documents.create",
            headers=_hdr(),
            json=body,
            timeout=30.0,
        )
        if r.status_code == 200:
            return r.json()["data"]["id"]
        if r.status_code == 429:
            wait = 65 + attempt * 10
            print(f"  [rate-limit] '{title[:30]}' wait {wait}s (attempt {attempt+1})")
            time.sleep(wait)
            continue
        print(f"  [warn] '{title[:40]}' 실패 {r.status_code}: {r.text[:150]}")
        return None
    print(f"  [give-up] '{title[:40]}' 429 retry exhausted")
    return None


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit(f"manifest 없음: {MANIFEST}.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    spaces = manifest["spaces"]
    pages = manifest["pages"]
    print(f"[migrate] 대상: {len(spaces)} collections, {len(pages)} documents")

    result: dict[str, Any] = {"collections": {}, "documents": {}}

    with httpx.Client(timeout=60.0) as client:
        existing = list_existing_collections(client)
        print(f"[migrate] 기존 collections: {list(existing.keys())}")

        for space_name, info in spaces.items():
            if space_name in existing:
                cid = existing[space_name]
                print(f"[migrate] '{space_name}' 이미 존재 (id={cid})")
            else:
                desc = f"source_collections: {', '.join(info['source_collections'])}"
                cid = create_collection(client, space_name, desc)
            result["collections"][space_name] = cid

        # Resume support — 이미 import 된 문서는 skip
        prior: dict[str, str] = {}
        if RESULT.exists():
            try:
                prior_data = json.loads(RESULT.read_text(encoding="utf-8"))
                prior = prior_data.get("documents") or {}
                if prior:
                    print(f"[migrate] resume: {len(prior)} 개 이미 import 됨, skip")
                    result["documents"].update(prior)
            except Exception:
                pass

        ok = len(prior)
        fail = 0
        for i, p in enumerate(pages, 1):
            if p["file"] in prior:
                continue
            space = p["space"]
            cid = result["collections"].get(space)
            if not cid:
                fail += 1
                continue
            md_path = MIG_DIR / p["file"]
            if not md_path.exists():
                fail += 1
                continue
            raw = md_path.read_text(encoding="utf-8")
            text, _meta = _strip_frontmatter(raw)
            text = re.sub(r"^# .+\n+", "", text, count=1)
            doc_id = create_document(client, cid, p["title"], text)
            if doc_id:
                result["documents"][p["file"]] = doc_id
                ok += 1
                # 진행 중간 결과 저장 (rate-limit 끊겨도 resume 가능)
                if ok % 10 == 0:
                    RESULT.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            else:
                fail += 1
            if i % 10 == 0:
                print(f"  progress {i}/{len(pages)} (ok={ok} fail={fail})")
            time.sleep(2.6)  # 25/min rate-limit 회피 — 23/min 페이스

    RESULT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[migrate] 완료: {ok} pages / {fail} 실패")
    print(f"[migrate] 결과: {RESULT}")
    print(f"[migrate] 확인: {_base()}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
