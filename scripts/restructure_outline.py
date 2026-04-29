"""Outline 페이지 재구조화 — 중복 제거 + Q&A → 사실 데이터 변환.

순서:
  1) 본문 머리글 제거 ([학사 | doc_id | ...] 패턴)
  2) Q&A 패턴 → 사실 데이터로 변환 (Solar Pro 사용)
  3) 동일 title 중복 페이지 병합

cross-collection 통합 (학식당 등) 은 별도 작업.

Run:
    python scripts/restructure_outline.py [--dry-run] [--limit N]
스킵 옵션:
    --skip-noise / --skip-qa / --skip-merge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT = PROJECT_ROOT / "tmp" / "outline_audit.json"
LOG = PROJECT_ROOT / "tmp" / "restructure_log.json"

OUTLINE_BASE = os.environ.get("OUTLINE_URL", "http://localhost:3002").rstrip("/")
SOLAR_BASE = "https://api.upstage.ai/v1"


def _outline_hdr() -> dict[str, str]:
    tok = os.environ.get("OUTLINE_TOKEN", "")
    if not tok:
        path = PROJECT_ROOT / "tmp" / "outline_api_token.txt"
        if path.exists():
            tok = path.read_text(encoding="utf-8").strip()
    if not tok:
        raise SystemExit("OUTLINE_TOKEN env 필요")
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


# ─────────────────────────────────────────────
# 1) 본문 머리글 제거
# ─────────────────────────────────────────────


_HEADER_PATTERNS = [
    re.compile(r"^\\\[[^\]]+\|[^\]]+\\\]\s*", re.MULTILINE),
    re.compile(r"^\[[^\]]+\|[^\]]+\]\s*", re.MULTILINE),
    re.compile(r"^\[학사[^\]]*\]\s*", re.MULTILINE),
    re.compile(r"^\[강의평가[^\]]*\]\s*", re.MULTILINE),
    re.compile(r"^\[기타 시설[^\]]*\]\s*", re.MULTILINE),
    re.compile(r"^\[FAQ[^\]]*\]\s*", re.MULTILINE),
    re.compile(r"^\[학생서비스[^\]]*\]\s*", re.MULTILINE),
    re.compile(r"^\[시설 예약[^\]]*\]\s*", re.MULTILINE),
]

_NOISE_PATTERNS = [
    re.compile(r"\*\*질문:\*\*\s*", re.IGNORECASE),
    re.compile(r"\*\*답변:\*\*\s*", re.IGNORECASE),
    re.compile(r"\*\*\s*좋은\s*답변:\*\*\s*", re.IGNORECASE),
    re.compile(r"\*\*\s*부족한\s*답변:\*\*\s*", re.IGNORECASE),
]


def strip_noise(text: str) -> str:
    for p in _HEADER_PATTERNS:
        text = p.sub("", text)
    for p in _NOISE_PATTERNS:
        text = p.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────
# 2) Q&A → 사실 데이터 (Solar Pro)
# ─────────────────────────────────────────────


CONVERT_PROMPT = """다음 한국어 학사 페이지 본문을 RAG 검색용 \"사실 데이터 시트\"로 재구성하세요.

규칙 (반드시 준수):
1. Q&A 형식 (Q. ... A. ...) 제거. 답변 내용만 추출하여 ## 섹션 제목 + 항목별 사실 list 로 재배치.
2. \"예를 들어\", \"예시:\", \"참고:\" 등 narrative 제거. 사실만 유지.
3. \"**질문:**\", \"**답변:**\", \"**좋은 답변:**\" 같은 마커 제거.
4. 정보를 의미 그룹별 ## 섹션으로 정리 (예: ## 위치, ## 운영시간, ## 절차, ## 자격, ## 신청 방법).
5. 각 섹션 내용은 가능하면 항목 list (- 항목) 또는 짧은 단락.
6. 출처 언급/메타 코멘트 (\"본 페이지는 ...\") 제거. 단 강의평가의 디스클레이머는 유지.
7. 표/수치는 그대로 유지.
8. 학과명·금액·날짜·전화번호 등 구체 데이터 절대 변경하지 말 것.
9. 결과는 markdown. 첫 줄에 # 페이지 제목 (전달받은 제목 그대로).

원본 페이지 제목: {title}

원본 본문:
{body}

재구성된 사실 데이터 (markdown):"""


def convert_qa_to_facts(client: httpx.Client, title: str, body: str) -> str:
    r = client.post(
        f"{SOLAR_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {_solar_key()}", "Content-Type": "application/json"},
        json={
            "model": "solar-pro",
            "messages": [
                {"role": "system", "content": "당신은 학사 데이터 정리 전문가입니다. RAG 검색에 적합한 사실 데이터 시트만 출력합니다."},
                {"role": "user", "content": CONVERT_PROMPT.format(title=title, body=body[:6000])},
            ],
            "temperature": 0.0,
            "max_tokens": 2000,
        },
        timeout=90.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Solar 호출 실패 {r.status_code}: {r.text[:200]}")
    msg = (r.json().get("choices") or [{}])[0].get("message", {})
    return (msg.get("content") or "").strip()


# ─────────────────────────────────────────────
# Outline API helpers
# ─────────────────────────────────────────────


def get_doc(client: httpx.Client, doc_id: str) -> dict[str, Any]:
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.info",
        headers=_outline_hdr(),
        json={"id": doc_id},
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json().get("data") or {}


def update_doc(client: httpx.Client, doc_id: str, text: str, dry: bool = False) -> bool:
    if dry:
        return True
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.update",
        headers=_outline_hdr(),
        json={"id": doc_id, "text": text, "publish": True, "append": False},
        timeout=30.0,
    )
    if r.status_code != 200:
        print(f"  [warn] update {doc_id} 실패 {r.status_code}: {r.text[:150]}")
        return False
    return True


def delete_doc(client: httpx.Client, doc_id: str, dry: bool = False) -> bool:
    if dry:
        return True
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.delete",
        headers=_outline_hdr(),
        json={"id": doc_id, "permanent": False},
        timeout=15.0,
    )
    return r.status_code == 200


# ─────────────────────────────────────────────
# 3) 동일 title 병합
# ─────────────────────────────────────────────


def merge_duplicate_titles(
    client: httpx.Client,
    audit: dict[str, Any],
    dry: bool,
    log: list[dict[str, Any]],
) -> int:
    merged = 0
    for grp in audit.get("duplicate_titles", []):
        ids = grp["ids"]
        if len(ids) < 2:
            continue
        bodies = []
        for did in ids:
            try:
                d = get_doc(client, did)
                bodies.append((did, d.get("text", "")))
            except Exception:
                continue
        if len(bodies) < 2:
            continue
        bodies.sort(key=lambda x: -len(x[1]))
        keep_id, keep_body = bodies[0]
        merge_bodies = [b for did, b in bodies[1:] if b]
        seen = set(keep_body.splitlines())
        for body in merge_bodies:
            for line in body.splitlines():
                if line.strip() and line not in seen:
                    keep_body += "\n" + line
                    seen.add(line)
        keep_body = keep_body.strip()
        ok = update_doc(client, keep_id, keep_body, dry=dry)
        if ok:
            for did, _ in bodies[1:]:
                if delete_doc(client, did, dry=dry):
                    merged += 1
                    log.append({"action": "merge_duplicate_title", "kept": keep_id, "deleted": did})
        time.sleep(0.5)
    return merged


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="처리 페이지 수 제한 (0=전체)")
    p.add_argument("--skip-noise", action="store_true")
    p.add_argument("--skip-qa", action="store_true")
    p.add_argument("--skip-merge", action="store_true")
    args = p.parse_args()

    if not AUDIT.exists():
        raise SystemExit("audit_outline.py 먼저 실행")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    log: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    n_noise = 0
    n_qa = 0
    n_merge = 0

    with httpx.Client(timeout=120.0) as client:
        if not args.skip_noise:
            print(f"[step1] 본문 머리글 제거 시작 ({len(audit['docs'])} 페이지)")
            for i, doc_meta in enumerate(audit["docs"], start=1):
                if args.limit and i > args.limit:
                    break
                doc_id = doc_meta["id"]
                try:
                    d = get_doc(client, doc_id)
                except Exception as exc:
                    log.append({"action": "noise_get_failed", "doc_id": doc_id, "error": str(exc)})
                    continue
                old_text = d.get("text", "") or ""
                new_text = strip_noise(old_text)
                if new_text != old_text and new_text.strip():
                    if update_doc(client, doc_id, new_text, dry=args.dry_run):
                        n_noise += 1
                        log.append({
                            "action": "strip_noise",
                            "doc_id": doc_id,
                            "before_len": len(old_text),
                            "after_len": len(new_text),
                        })
                if i % 25 == 0:
                    print(f"  noise {i}/{len(audit['docs'])}  cleaned={n_noise}")
                time.sleep(0.4)
            print(f"  → {n_noise} 페이지 노이즈 제거")

        if not args.skip_qa:
            qa_targets = audit.get("qa_heavy", [])
            if args.limit:
                qa_targets = qa_targets[: args.limit]
            print(f"[step2] Q&A → 사실 변환 시작 ({len(qa_targets)} 페이지)")
            for i, doc_meta in enumerate(qa_targets, start=1):
                doc_id = doc_meta["id"]
                title = doc_meta.get("title", "")
                try:
                    d = get_doc(client, doc_id)
                    body = d.get("text", "")
                    if not body or len(body) < 100:
                        continue
                    new_body = convert_qa_to_facts(client, title, body)
                    if not new_body.strip() or len(new_body) < 50:
                        log.append({"action": "qa_convert_empty", "doc_id": doc_id})
                        continue
                    if update_doc(client, doc_id, new_body, dry=args.dry_run):
                        n_qa += 1
                        log.append({
                            "action": "qa_to_facts",
                            "doc_id": doc_id,
                            "before_len": len(body),
                            "after_len": len(new_body),
                        })
                        print(f"  qa[{i}/{len(qa_targets)}] {title[:30]}: {len(body)} → {len(new_body)}")
                except Exception as exc:
                    log.append({"action": "qa_failed", "doc_id": doc_id, "error": str(exc)})
                    print(f"  [warn] qa {doc_id}: {exc}")
                time.sleep(2.5)
            print(f"  → {n_qa} 페이지 Q&A 변환")

        if not args.skip_merge:
            print(f"[step3] 동일 title 병합 ({len(audit.get('duplicate_titles',[]))} 그룹)")
            n_merge = merge_duplicate_titles(client, audit, args.dry_run, log)
            print(f"  → {n_merge} 페이지 병합/삭제")

    LOG.write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "dry_run": args.dry_run,
                "totals": {"noise_cleaned": n_noise, "qa_converted": n_qa, "merged": n_merge},
                "log": log,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"[restructure] saved {LOG}")
    print(f"  noise={n_noise} qa={n_qa} merge={n_merge} (dry={args.dry_run})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
