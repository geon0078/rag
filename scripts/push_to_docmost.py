"""tmp/docmost-zip/*.zip → Docmost workspace 일괄 업로드.

선행: scripts/migrate_to_docmost.py 로 ZIP 생성.

Flow:
  1. 기존 Spaces 조회 (POST /api/spaces) → 기존 이름 dict.
  2. ZIP 파일마다:
     - 동일 이름 Space 가 없으면 POST /api/spaces/create 로 생성.
     - POST /api/pages/import-zip (multipart) — spaceId, file.

Auth:
  Authorization: Bearer <JWT>  — DOCMOST_TOKEN env var.

Run:
    DOCMOST_TOKEN=eyJ... python scripts/push_to_docmost.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_DIR = PROJECT_ROOT / "tmp" / "docmost-zip"
MANIFEST = PROJECT_ROOT / "tmp" / "docmost-migration" / "manifest.json"


def _hdr() -> dict[str, str]:
    tok = os.environ.get("DOCMOST_TOKEN", "")
    if not tok:
        raise SystemExit("DOCMOST_TOKEN 환경변수 필요 (Docmost JWT).")
    return {"Authorization": f"Bearer {tok}"}


def _base() -> str:
    return os.environ.get("DOCMOST_URL", "http://localhost:3001").rstrip("/")


def list_spaces(client: httpx.Client) -> dict[str, str]:
    """이름 → spaceId 맵."""
    r = client.post(
        f"{_base()}/api/spaces",
        headers={**_hdr(), "Content-Type": "application/json"},
        json={"page": 1, "limit": 100},
        timeout=15.0,
    )
    r.raise_for_status()
    out: dict[str, str] = {}
    body = r.json()
    items = body.get("items") or (body.get("data") or {}).get("items") or []
    for s in items:
        out[s["name"]] = s["id"]
    return out


SLUG_MAP = {
    "학칙": "regulations",
    "학사일정": "calendar",
    "학사정보": "academic",
    "시설": "facility",
    "장학금": "scholarship",
    "강의평가": "lecturereview",
    "학과정보": "department",
    "기타": "misc",
}


def create_space(client: httpx.Client, name: str) -> str:
    slug = SLUG_MAP.get(name, "space" + str(abs(hash(name)) % 100000))
    body = {"name": name, "description": f"{name} 자동 생성", "slug": slug}
    r = client.post(
        f"{_base()}/api/spaces/create",
        headers={**_hdr(), "Content-Type": "application/json"},
        json=body,
        timeout=15.0,
    )
    if r.status_code not in (200, 201):
        raise SystemExit(f"space '{name}' 생성 실패 {r.status_code}: {r.text[:300]}")
    body = r.json()
    sid = (body.get("data") or body).get("id")
    if not sid:
        raise SystemExit(f"space '{name}' id 없음: {body}")
    print(f"[push] space '{name}' created id={sid}")
    return sid


def import_zip(client: httpx.Client, space_id: str, zip_path: Path) -> int:
    with zip_path.open("rb") as f:
        files = {"file": (zip_path.name, f, "application/zip")}
        data = {"spaceId": space_id, "source": "generic"}
        r = client.post(
            f"{_base()}/api/pages/import-zip",
            headers=_hdr(),
            files=files,
            data=data,
            timeout=600.0,
        )
    if r.status_code not in (200, 201, 202):
        print(f"[push] import-zip {zip_path.name} 실패 {r.status_code}: {r.text[:300]}")
        return 0
    try:
        body = r.json()
    except Exception:
        body = {}
    return int(body.get("count") or body.get("imported") or 1)


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit(f"manifest 없음: {MANIFEST}. migrate_to_docmost.py 먼저 실행.")
    if not ZIP_DIR.exists():
        raise SystemExit(f"ZIP dir 없음: {ZIP_DIR}.")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spaces_to_create = list(manifest.get("spaces", {}).keys())

    with httpx.Client(timeout=60.0) as client:
        existing = list_spaces(client)
        print(f"[push] 기존 Spaces: {list(existing.keys())}")

        space_ids: dict[str, str] = {}
        for name in spaces_to_create:
            if name in existing:
                space_ids[name] = existing[name]
                print(f"[push] '{name}' 이미 존재 (id={existing[name]})")
            else:
                space_ids[name] = create_space(client, name)

        total_imported = 0
        for zip_file in sorted(ZIP_DIR.glob("*.zip")):
            space_name = zip_file.stem
            sid = space_ids.get(space_name)
            if sid is None:
                print(f"[push] {zip_file.name}: matching Space 없음, skip")
                continue
            print(f"[push] uploading {zip_file.name} -> Space '{space_name}' ...")
            n = import_zip(client, sid, zip_file)
            print(f"[push]   imported {n} pages")
            total_imported += n

        print()
        print(f"[push] 완료: {total_imported} pages across {len(space_ids)} spaces")
        print(f"[push] 확인: {_base()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
