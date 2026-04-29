"""Outline 페이지 트리 구조화 — 큰 페이지 split + 학칙 chapter grouping.

전략:
  1. **large**: 큰 페이지(>3000자)를 Solar Pro 로 sub-topic 분리 → 자식 페이지 생성.
     부모 페이지는 overview + 각 자식 링크 + 공통 정보로 갱신.
  2. **chapters**: 학칙 90 페이지를 corpus.parquet 의 chapter metadata 기반으로
     장(章) 단위 부모 페이지 생성 + 기존 조항들을 자식으로 move.

Run:
    python scripts/restructure_subpages.py --target large
    python scripts/restructure_subpages.py --target chapters
    python scripts/restructure_subpages.py --target all
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
LOG = PROJECT_ROOT / "tmp" / "restructure_subpages_log.json"
AUDIT = PROJECT_ROOT / "tmp" / "outline_audit.json"
OUTLINE_BASE = os.environ.get("OUTLINE_URL", "http://localhost:3002").rstrip("/")
SOLAR_BASE = "https://api.upstage.ai/v1/solar"
LARGE_THRESHOLD = 3000


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


def get_doc(client: httpx.Client, doc_id: str) -> dict[str, Any]:
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.info",
        headers=_outline_hdr(),
        json={"id": doc_id},
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json().get("data") or {}


def update_doc(client: httpx.Client, doc_id: str, text: str, dry: bool) -> bool:
    if dry:
        return True
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.update",
        headers=_outline_hdr(),
        json={"id": doc_id, "text": text, "publish": True, "append": False},
        timeout=30.0,
    )
    return r.status_code == 200


def create_child_doc(
    client: httpx.Client,
    title: str,
    text: str,
    collection_id: str,
    parent_id: str,
    dry: bool,
) -> str | None:
    if dry:
        return "dry-run-id"
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.create",
        headers=_outline_hdr(),
        json={
            "title": title,
            "text": text,
            "collectionId": collection_id,
            "parentDocumentId": parent_id,
            "publish": True,
        },
        timeout=30.0,
    )
    if r.status_code != 200:
        print(f"  [warn] child create '{title[:30]}' 실패 {r.status_code}: {r.text[:200]}")
        return None
    return (r.json().get("data") or {}).get("id")


def move_doc(
    client: httpx.Client,
    doc_id: str,
    parent_id: str,
    collection_id: str,
    dry: bool,
) -> bool:
    if dry:
        return True
    r = client.post(
        f"{OUTLINE_BASE}/api/documents.move",
        headers=_outline_hdr(),
        json={
            "id": doc_id,
            "parentDocumentId": parent_id,
            "collectionId": collection_id,
        },
        timeout=15.0,
    )
    if r.status_code != 200:
        print(f"  [warn] move {doc_id[:8]} 실패 {r.status_code}: {r.text[:150]}")
        return False
    return True


SPLIT_PROMPT = """다음 한국어 페이지를 자연스러운 sub-topic 단위로 분할하세요.

조건:
1. 각 sub-topic = 의미적으로 독립적인 단위 (예: 장학금 종류, 인증제 영역, 시설 종류)
2. 부모 페이지는 overview (각 sub 의 제목 list + 공통 안내 + 공통 문의처)
3. sub-page title: 명확한 명사구 (의문문 X)
4. 각 sub-page 본문: ## 섹션 headers + 사실 list. Q&A 형식 X.
5. 원본의 모든 사실 데이터를 보존 (절대 누락 X). 특히 전화번호·금액·날짜·자격 조건.
6. 3-8개 sub-page 권장.
7. 모든 sub-page 가 동일 컬렉션. parent 는 부모 페이지.

원본 페이지 제목: {title}
원본 본문 (총 {n}자):
{body}

JSON 으로만 응답:
{{
  "parent_overview": "부모 페이지 본문 — 각 sub-page 링크 + 공통 안내",
  "children": [
    {{"title": "...", "content": "..."}},
    {{"title": "...", "content": "..."}}
  ]
}}"""


def split_with_solar(client: httpx.Client, title: str, body: str) -> dict[str, Any] | None:
    r = client.post(
        f"{SOLAR_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {_solar_key()}", "Content-Type": "application/json"},
        json={
            "model": "solar-pro",
            "messages": [
                {"role": "system", "content": "당신은 학사 데이터 정리 전문가입니다. JSON 만 출력합니다."},
                {"role": "user", "content": SPLIT_PROMPT.format(title=title, body=body[:8000], n=len(body))},
            ],
            "temperature": 0.0,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"},
        },
        timeout=120.0,
    )
    if r.status_code != 200:
        print(f"  [warn] Solar 호출 실패 {r.status_code}: {r.text[:200]}")
        return None
    msg = (r.json().get("choices") or [{}])[0].get("message", {})
    raw = (msg.get("content") or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        print(f"  [warn] JSON 파싱 실패: {raw[:200]}")
        return None


def split_large_pages(
    client: httpx.Client, audit: dict[str, Any], threshold: int, dry: bool, limit: int
) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    targets = sorted(
        [d for d in audit["docs"] if d["text_len"] > threshold and d["collection"] != "강의평가"],
        key=lambda d: -d["text_len"],
    )
    if limit:
        targets = targets[:limit]
    print(f"[split] 대상 {len(targets)} 페이지 (>{threshold}자, 강의평가 제외)")

    for i, dm in enumerate(targets, start=1):
        try:
            doc = get_doc(client, dm["id"])
            body = doc.get("text", "")
            if not body or len(body) < threshold:
                continue
            print(f"\n[{i}/{len(targets)}] '{doc.get('title','')}' ({len(body)} 자) → split 분석 중...")
            split = split_with_solar(client, doc.get("title", ""), body)
            if not split or not split.get("children"):
                print("  [skip] split 결과 없음")
                continue
            children = split["children"]
            parent_text = split.get("parent_overview", "").strip() or doc.get("title", "")
            print(f"  → {len(children)} 자식 페이지 생성")

            ok = update_doc(client, dm["id"], parent_text, dry=dry)
            if not ok:
                continue
            child_ids: list[str] = []
            for ch in children:
                ch_title = (ch.get("title") or "").strip()
                ch_content = (ch.get("content") or "").strip()
                if not ch_title or not ch_content:
                    continue
                cid = create_child_doc(
                    client,
                    title=ch_title,
                    text=ch_content,
                    collection_id=doc.get("collectionId"),
                    parent_id=dm["id"],
                    dry=dry,
                )
                if cid:
                    child_ids.append(cid)
                time.sleep(2.5)
            log.append({
                "action": "split",
                "parent_id": dm["id"],
                "parent_title": doc.get("title"),
                "before_len": len(body),
                "child_count": len(child_ids),
                "child_ids": child_ids,
            })
            print(f"  ✓ {len(child_ids)} children 생성")
            time.sleep(1.0)
        except Exception as exc:
            log.append({"action": "split_failed", "doc_id": dm["id"], "error": str(exc)})
            print(f"  [err] {exc}")
    return log


def chapter_grouping(
    client: httpx.Client, audit: dict[str, Any], dry: bool
) -> list[dict[str, Any]]:
    """학칙 collection 의 페이지를 corpus.parquet chapter 기반으로 grouping."""
    import pandas as pd

    log: list[dict[str, Any]] = []
    df = pd.read_parquet(PROJECT_ROOT / "data" / "corpus.parquet")
    rul = df[df["metadata"].apply(lambda m: m.get("source_collection") == "학칙_조항")]

    article_to_chapter: dict[str, str] = {}
    for _, r in rul.iterrows():
        meta = r["metadata"] if isinstance(r["metadata"], dict) else dict(r["metadata"])
        title = str(meta.get("title", ""))
        chapter = meta.get("chapter")
        if not chapter:
            continue
        m = re.match(r"제(\d+)조", title)
        if m:
            article_to_chapter[m.group(1)] = chapter

    chapters: dict[str, list[dict[str, Any]]] = {}
    rul_collection_id: str | None = None
    for d in audit["docs"]:
        if d["collection"] != "학칙":
            continue
        title = d.get("title", "")
        m = re.match(r"제(\d+)조", title)
        if not m:
            continue
        chapter = article_to_chapter.get(m.group(1))
        if not chapter:
            continue
        chapters.setdefault(chapter, []).append(d)
        if not rul_collection_id:
            rul_collection_id = d["collection_id"]

    print(f"[chapters] {len(chapters)} 장(章) 식별, 학칙 collection: {rul_collection_id}")
    for ch, arts in sorted(chapters.items()):
        print(f"  {ch}: {len(arts)} 조")

    if dry or not rul_collection_id:
        return log

    for chapter_title, articles in sorted(chapters.items()):
        sorted_arts = sorted(
            articles,
            key=lambda x: int(re.match(r"제(\d+)조", x["title"]).group(1))
            if re.match(r"제(\d+)조", x["title"]) else 0,
        )
        overview_lines = [f"# {chapter_title}", "", "본 장은 다음 조항을 포함합니다:", ""]
        overview_lines.extend(f"- {a['title']}" for a in sorted_arts)
        overview = "\n".join(overview_lines)

        r = httpx.post(
            f"{OUTLINE_BASE}/api/documents.create",
            headers=_outline_hdr(),
            json={
                "title": chapter_title,
                "text": overview,
                "collectionId": rul_collection_id,
                "publish": True,
            },
            timeout=30.0,
        )
        if r.status_code != 200:
            print(f"  [warn] chapter '{chapter_title}' 생성 실패")
            continue
        chapter_id = (r.json().get("data") or {}).get("id")
        log.append({"action": "chapter_create", "title": chapter_title, "id": chapter_id, "n_articles": len(articles)})
        moved = 0
        for art in sorted_arts:
            ok = move_doc(client, art["id"], chapter_id, rul_collection_id, dry=False)
            if ok:
                moved += 1
            time.sleep(0.3)
        log.append({"action": "chapter_moved", "chapter": chapter_title, "moved": moved})
        print(f"  ✓ {chapter_title}: {moved}/{len(articles)} 조 이동")
        time.sleep(1.0)
    return log


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["large", "chapters", "all"], default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threshold", type=int, default=LARGE_THRESHOLD)
    args = p.parse_args()

    if not AUDIT.exists():
        raise SystemExit("audit_outline.py 먼저 실행")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    log: list[dict[str, Any]] = []

    with httpx.Client(timeout=180.0) as client:
        if args.target in ("large", "all"):
            log.extend(split_large_pages(client, audit, args.threshold, args.dry_run, args.limit))
        if args.target in ("chapters", "all"):
            log.extend(chapter_grouping(client, audit, args.dry_run))

    LOG.write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "dry_run": args.dry_run,
                "log": log,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[done] {LOG}")
    print(f"  log entries: {len(log)} (dry={args.dry_run})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
