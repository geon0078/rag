"""Onyx 에 우리 corpus ZIP 을 프로그래매틱하게 import (UI 업로드 우회).

Flow:
  1. POST /api/manage/admin/connector/file/upload (ZIP, unzip=true)
     → file_paths (FileStore UUID), file_names
  2. POST /api/manage/credential       — empty FILE credential
  3. POST /api/manage/admin/connector  — FILE source, file_locations + file_names
  4. PUT  /api/manage/connector/{cid}/credential/{credid} — link, triggers indexing
  5. (poll) POST /api/manage/admin/connector/indexing-status

Run:
    ONYX_API_KEY=on_... python scripts/onyx_import_corpus.py

Required env:
    ONYX_API_KEY     — admin scope key
    ONYX_BASE_URL    — default http://localhost:3010
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

log = get_logger("onyx_import")

ONYX_BASE = os.environ.get("ONYX_BASE_URL", "http://localhost:3010")
ZIP_PATH = PROJECT_ROOT / "tmp" / "onyx-import.zip"
CONNECTOR_NAME = "eulji_corpus"


def _hdr() -> dict[str, str]:
    key = os.environ.get("ONYX_API_KEY", "")
    if not key:
        raise SystemExit("ONYX_API_KEY 환경변수 필요 — admin scope.")
    return {"Authorization": f"Bearer {key}"}


CACHED_UPLOAD = PROJECT_ROOT / "tmp" / "onyx_upload_response.json"


def upload_zip(client: httpx.Client) -> tuple[list[str], list[str]]:
    if CACHED_UPLOAD.exists():
        import json
        log.info(f"reuse cached upload response: {CACHED_UPLOAD}")
        data = json.loads(CACHED_UPLOAD.read_text(encoding="utf-8"))
        return data["file_paths"], data["file_names"]
    if not ZIP_PATH.exists():
        raise SystemExit(f"ZIP 없음: {ZIP_PATH}. 먼저 export_corpus_for_onyx.py 실행.")
    log.info(f"upload {ZIP_PATH.name} ({ZIP_PATH.stat().st_size/1024:.1f} KB)")
    with ZIP_PATH.open("rb") as f:
        files = {"files": (ZIP_PATH.name, f, "application/zip")}
        r = client.post(
            f"{ONYX_BASE}/api/manage/admin/connector/file/upload",
            params={"unzip": "true"},
            headers=_hdr(),
            files=files,
            timeout=300.0,
        )
    r.raise_for_status()
    data = r.json()
    paths = data["file_paths"]
    names = data["file_names"]
    log.info(f"uploaded {len(paths)} files (unzipped)")
    return paths, names


def create_credential(client: httpx.Client) -> int:
    body = {
        "credential_json": {},
        "admin_public": True,
        "source": "file",
        "name": f"{CONNECTOR_NAME}_credential",
        "curator_public": False,
        "groups": [],
    }
    r = client.post(
        f"{ONYX_BASE}/api/manage/credential",
        headers={**_hdr(), "Content-Type": "application/json"},
        json=body,
        timeout=30.0,
    )
    r.raise_for_status()
    cid = int(r.json()["id"])
    log.info(f"credential id={cid}")
    return cid


def create_connector(
    client: httpx.Client, file_paths: list[str], file_names: list[str]
) -> int:
    body = {
        "name": CONNECTOR_NAME,
        "source": "file",
        "input_type": "load_state",
        "connector_specific_config": {
            "file_locations": file_paths,
            "file_names": file_names,
            "zip_metadata": {},
        },
        "refresh_freq": None,
        "prune_freq": None,
        "indexing_start": None,
        "access_type": "public",
        "groups": [],
    }
    r = client.post(
        f"{ONYX_BASE}/api/manage/admin/connector",
        headers={**_hdr(), "Content-Type": "application/json"},
        json=body,
        timeout=30.0,
    )
    if r.status_code != 200:
        log.error(f"connector create failed {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
    cid = int(r.json()["id"])
    log.info(f"connector id={cid}")
    return cid


def associate(client: httpx.Client, connector_id: int, credential_id: int) -> int:
    body = {
        "name": f"{CONNECTOR_NAME}_pair",
        "access_type": "public",
        "auto_sync_options": None,
        "groups": [],
        "processing_mode": "REGULAR",
    }
    r = client.put(
        f"{ONYX_BASE}/api/manage/connector/{connector_id}/credential/{credential_id}",
        headers={**_hdr(), "Content-Type": "application/json"},
        json=body,
        timeout=30.0,
    )
    if r.status_code != 200:
        log.error(f"associate failed {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
    cc_pair_id = int(r.json()["data"])
    log.info(f"cc-pair id={cc_pair_id} (indexing triggered)")
    return cc_pair_id


def poll_status(client: httpx.Client, max_minutes: int = 60) -> None:
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        r = client.post(
            f"{ONYX_BASE}/api/manage/admin/connector/indexing-status",
            headers={**_hdr(), "Content-Type": "application/json"},
            json={"source": "file", "get_all_connectors": True},
            timeout=30.0,
        )
        if r.status_code == 200:
            payload = r.json()
            for src in payload:
                summ = src.get("summary", {})
                log.info(
                    "status: total_docs_indexed="
                    f"{summ.get('total_docs_indexed', '?')}  "
                    f"active={summ.get('active_connectors', '?')}/"
                    f"{summ.get('total_connectors', '?')}"
                )
            done = any(
                src.get("summary", {}).get("total_docs_indexed", 0) >= 2000
                for src in payload
            )
            if done:
                log.info("indexing reached 2000+ docs")
                return
        time.sleep(30)
    log.warning(f"{max_minutes}min within indexing not complete — proceeding")


def main() -> int:
    with httpx.Client(timeout=300.0) as client:
        file_paths, file_names = upload_zip(client)
        cred_id = create_credential(client)
        conn_id = create_connector(client, file_paths, file_names)
        associate(client, conn_id, cred_id)
        log.info("indexing started — monitoring (up to 60min)")
        poll_status(client, max_minutes=60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
