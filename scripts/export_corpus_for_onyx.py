"""우리 corpus → Onyx File Connector 업로드용 ZIP 패키지.

onyx+docmost 개발.md A2 시나리오 — Onyx 풀 stack 위에 우리 RAG 구축.

흐름:
  1. data/corpus.parquet 읽기 (2,382 청크)
  2. parent_doc_id 기준 그룹화 — 같은 부모 청크들을 1 markdown 파일로 결합
  3. Onyx File Connector 가 받아들이는 markdown + frontmatter 생성
  4. tmp/onyx-import/{source_collection}/ 디렉토리에 저장
  5. tmp/onyx-import.zip 으로 압축 → Onyx UI 업로드용

Frontmatter 형식 (Onyx file connector 가 metadata 로 인식):
  ---
  doc_id: si_static_info_75
  title: 졸업학점
  source_collection: 학사정보
  campus: 성남
  ---
  # 졸업학점
  ...

Run:
    python scripts/export_corpus_for_onyx.py [--limit 0]
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "tmp" / "onyx-import"
OUT_ZIP = PROJECT_ROOT / "tmp" / "onyx-import.zip"


_SAFE = re.compile(r"[^\w가-힣\-]+")


def _safe_filename(s: str) -> str:
    return _SAFE.sub("_", s).strip("_")[:120] or "untitled"


def _frontmatter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for k in ("doc_id", "title", "source_collection", "category", "campus", "path"):
        v = meta.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        lines.append(f"{k}: {str(v).replace(chr(10), ' ')}")
    lines.append("---")
    return "\n".join(lines)


def _build_markdown(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """parent_doc_id 가 같은 청크들을 1 markdown 으로."""
    head_meta = rows[0]["metadata"]
    title = head_meta.get("title") or rows[0].get("doc_id") or "(제목 없음)"

    body_parts: list[str] = []
    for r in sorted(rows, key=lambda x: int(x["metadata"].get("chunk_index", 0))):
        section = (
            r["metadata"].get("topic_section")
            or r["metadata"].get("section")
            or r["metadata"].get("chapter")
            or ""
        )
        body = (r["contents"] or "").strip()
        if not body:
            continue
        if section and section not in ("(본문)", "(intro)", title):
            body_parts.append(f"## {section}\n\n{body}")
        else:
            body_parts.append(body)

    md = _frontmatter(head_meta) + f"\n\n# {title}\n\n" + "\n\n".join(body_parts)
    return md, head_meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(PROJECT_ROOT / "data" / "corpus.parquet"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"[export] corpus 없음: {corpus}")
        return 1

    df = pd.read_parquet(corpus)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    print(f"[export] read {len(df)} chunks")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in df.iterrows():
        meta = row["metadata"] if isinstance(row["metadata"], dict) else dict(row["metadata"])
        rec = {"doc_id": row["doc_id"], "contents": row["contents"], "metadata": meta}
        parent = meta.get("parent_doc_id") or row["doc_id"]
        grouped[str(parent)].append(rec)

    print(f"[export] grouped into {len(grouped)} pages")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for parent, rows in grouped.items():
        md, head_meta = _build_markdown(rows)
        coll = (head_meta.get("source_collection") or "기타").replace("/", "_")
        title = head_meta.get("title") or parent
        coll_dir = OUT_DIR / _safe_filename(coll)
        coll_dir.mkdir(parents=True, exist_ok=True)
        fname = _safe_filename(f"{parent}__{title}") + ".md"
        (coll_dir / fname).write_text(md, encoding="utf-8")
        n_files += 1

    print(f"[export] wrote {n_files} markdown files to {OUT_DIR}")

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUT_DIR.rglob("*.md"):
            zf.write(f, arcname=f.relative_to(OUT_DIR))
    size_kb = OUT_ZIP.stat().st_size / 1024
    print(f"[export] zipped → {OUT_ZIP} ({size_kb:.1f} KB)")
    print()
    print("다음 단계:")
    print("  1. Onyx UI: http://localhost:3010/admin/connectors")
    print("  2. + Add Connector → File → 위 ZIP 파일 업로드")
    print("  3. Connector 인덱싱 완료 대기 (~10-30분)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
