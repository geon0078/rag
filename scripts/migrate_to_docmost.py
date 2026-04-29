"""corpus.parquet → Docmost markdown 페이지 빌드 (CSV_STORAGE_STRATEGY 사양).

Phase 1 — 자연 메타데이터 기반 그룹핑 (BERTopic 없이).

흐름:
  1. data/corpus.parquet 읽기 (2,382 청크).
  2. source_collection 별 다른 규칙으로 그룹핑:
     - 학칙_조항    → 조항 단위 (title 의 '제N조' 추출)
     - 강의평가     → lecture_id (doc_id 'lecture_reviews_N')
     - 학사일정     → 학기 단위 (title 의 'YYYY학년도 N학기')
     - FAQ/시설_연락처/장학금/학사정보/학과정보/교육과정/기타 → subcategory
  3. 그룹마다 마크다운 페이지 생성 (per-collection rule).
  4. tmp/docmost-migration/{space_safe}/{page_safe}.md 작성
     + frontmatter 메타데이터 (topic_id, source_collection, campus 등)
  5. tmp/docmost-migration/manifest.json (Space/Page 매핑)
  6. tmp/docmost-zip/{space}.zip — Space 별 import-zip 용 ZIP

Run:
    python scripts/migrate_to_docmost.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "tmp" / "docmost-migration"
ZIP_DIR = PROJECT_ROOT / "tmp" / "docmost-zip"

_SAFE = re.compile(r"[^\w가-힣\-·]+")


SPACE_NAMES = {
    "학칙_조항": "학칙",
    "학사일정": "학사일정",
    "FAQ": "학사정보",
    "시설_연락처": "시설",
    "장학금": "장학금",
    "강의평가": "강의평가",
    "학사정보": "학사정보",
    "학과정보": "학과정보",
    "교육과정": "학과정보",
    "기타": "기타",
}


def _safe(s: str, max_len: int = 80) -> str:
    return (_SAFE.sub("_", str(s)).strip("_"))[:max_len] or "untitled"


def _frontmatter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for k, v in meta.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if isinstance(v, list):
            v_str = ", ".join(str(x) for x in v)
        else:
            v_str = str(v).replace("\n", " ")
        lines.append(f"{k}: {v_str}")
    lines.append("---")
    return "\n".join(lines)


def _extract_article_no(title: str) -> str | None:
    m = re.match(r"제(\d+)조", str(title or ""))
    return m.group(1) if m else None


def _extract_paragraph_no(title: str) -> int | None:
    m = re.search(r"제\d+조\s*(\d+)\s*항", str(title or ""))
    return int(m.group(1)) if m else None


def _extract_lecture_id(doc_id: str) -> str | None:
    m = re.match(r"lecture_reviews_(\d+)", str(doc_id or ""))
    return m.group(1) if m else None


def _extract_semester(title: str) -> str | None:
    m = re.search(r"(\d{4})학년도\s*(\d)학기", str(title or ""))
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def _calendar_group(title: str, subcategory: str) -> str:
    t = str(title or "")
    sc = str(subcategory or "")
    if any(k in t or k in sc for k in ["수강신청", "수강바구니", "등록", "정정"]):
        return "등록·수강신청"
    if any(k in t for k in ["개강", "입학", "OT"]):
        return "개강·입학"
    if any(k in t for k in ["중간고사", "기말고사", "시험"]):
        return "시험"
    if any(k in t for k in ["방학", "종강"]):
        return "종강·방학"
    if any(k in t for k in ["휴학", "복학"]):
        return "휴학·복학"
    if any(k in t for k in ["졸업"]):
        return "졸업"
    return "기타 일정"


def convert_school_rules(chunks: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    """학칙: 같은 제N조의 청크들 → 1 페이지 (## N항 구조)."""
    head_meta = chunks[0]["metadata"]
    title_raw = str(head_meta.get("title") or "")
    article_no = _extract_article_no(title_raw)
    page_title = title_raw

    body_parts: list[str] = []
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            _extract_paragraph_no(c["metadata"].get("title", "")) or 0
        ),
    )
    for r in sorted_chunks:
        para_no = _extract_paragraph_no(r["metadata"].get("title", ""))
        body = (r["contents"] or "").strip()
        if not body:
            continue
        if para_no:
            body_parts.append(f"## {para_no}항\n\n{body}")
        else:
            body_parts.append(body)

    md = "\n\n".join(body_parts)
    meta = {
        "topic_id": f"topic_school_rules_article_{article_no}" if article_no else None,
        "topic_name": page_title,
        "source_collection": "학칙_조항",
        "category": "학칙",
        "chapter": head_meta.get("chapter"),
        "article_number": f"제{article_no}조" if article_no else None,
        "campus": head_meta.get("campus", "전체"),
        "owner": "교무처",
    }
    return page_title, md, meta


def convert_lecture_review(chunks: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    """강의평가: 같은 lecture_id 의 청크들 → 1 페이지 (디스클레이머 + 섹션)."""
    head_meta = chunks[0]["metadata"]
    page_title = str(head_meta.get("title") or "강의")
    subject_area = str(head_meta.get("subcategory") or "기타")

    disclaimer = "> ⚠️ 본 페이지는 학생들의 의견이며 객관적 사실이 아닙니다."

    body_parts: list[str] = []
    seen_sections: set[str] = set()
    for r in chunks:
        section = str(r["metadata"].get("section") or r["metadata"].get("chapter") or "").strip()
        body = (r["contents"] or "").strip()
        if not body:
            continue
        if section and section not in seen_sections and section not in ("(본문)", "(intro)", page_title):
            body_parts.append(f"## {section}\n\n{body}")
            seen_sections.add(section)
        else:
            body_parts.append(body)

    md = disclaimer + "\n\n" + "\n\n".join(body_parts)
    lecture_id = _extract_lecture_id(head_meta.get("doc_id", ""))
    meta = {
        "topic_id": f"topic_lecture_lecture_reviews_{lecture_id}" if lecture_id else None,
        "topic_name": page_title,
        "source_collection": "강의평가",
        "category": "강의평가",
        "lecture_id": f"lecture_reviews_{lecture_id}" if lecture_id else None,
        "subject_area": subject_area,
        "campus": head_meta.get("campus", "전체"),
        "confidence": "medium",
    }
    return page_title, md, meta


def convert_calendar(chunks: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    """학사일정: 같은 학기의 모든 이벤트 → 1 페이지 (의미 그룹별 ## 섹션)."""
    head_meta = chunks[0]["metadata"]
    semester = _extract_semester(head_meta.get("title", "")) or "미분류"
    if "-" in semester:
        page_title = f"{semester[:4]}학년도 {semester[-1]}학기"
    else:
        page_title = "학사일정 (기타)"

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in chunks:
        m = r["metadata"]
        g = _calendar_group(m.get("title", ""), m.get("subcategory", ""))
        grouped[g].append(r)

    order = ["등록·수강신청", "개강·입학", "시험", "휴학·복학", "졸업", "종강·방학", "기타 일정"]
    body_parts: list[str] = []
    for g in order:
        if g not in grouped:
            continue
        body_parts.append(f"## {g}")
        events = sorted(
            grouped[g],
            key=lambda r: str(r["metadata"].get("start_date") or ""),
        )
        for r in events:
            m = r["metadata"]
            sd = m.get("start_date")
            ed = m.get("end_date")
            if sd and ed and sd != ed:
                date_str = f"{sd} ~ {ed}"
            elif sd:
                date_str = str(sd)
            else:
                date_str = "(날짜 미정)"
            body_parts.append(f"- **{date_str}** | {m.get('title', '(제목 없음)')}")

    md = "\n".join(body_parts)
    meta = {
        "topic_id": f"topic_calendar_{semester}",
        "topic_name": page_title,
        "source_collection": "학사일정",
        "category": "학사일정",
        "semester": semester,
        "campus": head_meta.get("campus", "성남"),
    }
    return page_title, md, meta


def convert_subcategory_group(
    chunks: list[dict[str, Any]],
    source_collection: str,
    subcategory: str,
) -> tuple[str, str, dict[str, Any]]:
    """공통 변환: subcategory 단위 — FAQ/시설/장학금/학사정보 등."""
    page_title = subcategory or source_collection
    head_meta = chunks[0]["metadata"]

    body_parts: list[str] = []
    titles: list[str] = []
    for r in chunks:
        m = r["metadata"]
        title = str(m.get("title") or "").strip()
        body = (r["contents"] or "").strip()
        if not body:
            continue
        title = re.sub(r"^Q\.\s*", "", title)
        body_no_a = re.sub(r"^A\.\s*", "", body)
        if title:
            body_parts.append(f"## {title}\n\n{body_no_a}")
            titles.append(title)
        else:
            body_parts.append(body_no_a)

    md = "\n\n".join(body_parts)
    meta = {
        "topic_id": f"topic_{_safe(source_collection)}_{_safe(subcategory or 'all', 40)}",
        "topic_name": page_title,
        "source_collection": source_collection,
        "category": head_meta.get("category"),
        "subcategory": subcategory,
        "campus": head_meta.get("campus", "전체"),
        "questions_included": titles[:20],
    }
    return page_title, md, meta


def group_chunks(df: pd.DataFrame) -> dict[str, list[tuple[str, list[dict[str, Any]]]]]:
    """source_collection → list of (page_key, chunks)."""
    out: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    rows = []
    for _, r in df.iterrows():
        meta = r["metadata"] if isinstance(r["metadata"], dict) else dict(r["metadata"])
        rows.append({
            "doc_id": r["doc_id"],
            "contents": r["contents"],
            "metadata": meta,
        })

    sc_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in rows:
        sc_buckets[c["metadata"].get("source_collection", "기타")].append(c)

    for sc in ["학칙_조항"]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in sc_buckets.get(sc, []):
            ano = _extract_article_no(c["metadata"].get("title", ""))
            if ano is None:
                ano = c["metadata"].get("title", "기타")
            groups[ano].append(c)
        out[sc] = list(groups.items())

    for sc in ["강의평가"]:
        groups = defaultdict(list)
        for c in sc_buckets.get(sc, []):
            lid = _extract_lecture_id(c["metadata"].get("doc_id", ""))
            if lid is None:
                lid = c["metadata"].get("title", "기타")
            groups[lid].append(c)
        out[sc] = list(groups.items())

    for sc in ["학사일정"]:
        groups = defaultdict(list)
        for c in sc_buckets.get(sc, []):
            sem = _extract_semester(c["metadata"].get("title", "")) or "미분류"
            groups[sem].append(c)
        out[sc] = list(groups.items())

    for sc in ["FAQ", "시설_연락처", "장학금", "학사정보", "학과정보", "교육과정", "기타"]:
        groups = defaultdict(list)
        for c in sc_buckets.get(sc, []):
            sub = c["metadata"].get("subcategory") or "(분류 없음)"
            groups[sub].append(c)
        out[sc] = list(groups.items())

    return out


def main() -> int:
    corpus = PROJECT_ROOT / "data" / "corpus.parquet"
    if not corpus.exists():
        print(f"[migrate] corpus 없음: {corpus}")
        return 1

    df = pd.read_parquet(corpus)
    print(f"[migrate] read {len(df)} chunks")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    grouped = group_chunks(df)
    manifest: dict[str, Any] = {"spaces": {}, "pages": []}
    space_files: dict[str, list[Path]] = defaultdict(list)
    n_pages = 0

    for sc, groups in grouped.items():
        space_name = SPACE_NAMES.get(sc, sc)
        manifest["spaces"].setdefault(space_name, {"source_collections": []})
        if sc not in manifest["spaces"][space_name]["source_collections"]:
            manifest["spaces"][space_name]["source_collections"].append(sc)

        space_dir = OUT_DIR / _safe(space_name, 40)
        space_dir.mkdir(parents=True, exist_ok=True)

        for key, chunks in groups:
            if not chunks:
                continue
            if sc == "학칙_조항":
                title, md, meta = convert_school_rules(chunks)
            elif sc == "강의평가":
                title, md, meta = convert_lecture_review(chunks)
            elif sc == "학사일정":
                title, md, meta = convert_calendar(chunks)
            else:
                title, md, meta = convert_subcategory_group(chunks, sc, key if key != "(분류 없음)" else "")

            if not md.strip():
                continue

            page_md = _frontmatter(meta) + "\n\n# " + title + "\n\n" + md + "\n"
            fname = _safe(title, 100) + ".md"
            fpath = space_dir / fname
            fpath.write_text(page_md, encoding="utf-8")
            space_files[space_name].append(fpath)
            manifest["pages"].append({
                "space": space_name,
                "source_collection": sc,
                "title": title,
                "file": str(fpath.relative_to(OUT_DIR)),
                "n_chunks": len(chunks),
                "topic_id": meta.get("topic_id"),
            })
            n_pages += 1

    print(f"[migrate] wrote {n_pages} pages across {len(manifest['spaces'])} spaces")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for space_name, files in space_files.items():
        zip_path = ZIP_DIR / f"{_safe(space_name, 40)}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        print(
            f"[migrate] zipped {space_name}: {len(files)} pages "
            f"-> {zip_path.name} ({zip_path.stat().st_size/1024:.1f} KB)"
        )

    print()
    print("다음 단계:")
    print("  1) Docmost UI: http://localhost:3001 → Space 생성 → Import → ZIP 업로드")
    print("  2) 또는 토큰 발급 후 push_to_docmost.py 실행")
    print(f"  생성: {OUT_DIR}")
    print(f"  ZIPs: {ZIP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
