"""
을지대학교 지식관리 데이터 RAG 전처리 파이프라인 (v2)

원본 대비 변경:
1. calendar 소스: CSV → JSON (CSV는 64/69 날짜 결측)
2. start_date/end_date 파싱: 단일 날짜는 end_date=None 유지 (사용자 요구)
3. low_confidence NaN-truthy 버그 수정
4. 단계별 검증 출력 + 중간 산출물 저장 (_check/)
"""
import re
import sys
import json
import pandas as pd
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "구DB"
OUTPUT_DIR = BASE_DIR / "data"
CHECK_DIR = BASE_DIR / "_check"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "collections").mkdir(exist_ok=True)
CHECK_DIR.mkdir(exist_ok=True)

MAX_CHUNK_CHARS = 400
CHUNK_OVERLAP = 50

DATE_TOKEN = re.compile(r"(\d{4}-\d{2}-\d{2})\([^)]+\)")
RANGE_PAT = re.compile(
    r"(\d{4}-\d{2}-\d{2})\([^)]+\)\s*[~～∼]\s*(\d{4}-\d{2}-\d{2})\([^)]+\)"
)


def add_prefix(text, category, campus, source_id, title=""):
    parts = [category]
    if title and title not in text[:50]:
        parts.append(title)
    if campus and campus != "전체":
        parts.append(campus)
    parts.append(source_id)
    prefix = "[" + " | ".join(parts) + "]"
    return f"{prefix}\n{text}"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def verify_chunks(name, df, sample_n=3):
    """단계별 검증: stats + sample 출력 + 중간 CSV 저장"""
    print(f"\n  [verify] {name}")
    print(f"    rows: {len(df)}")
    if len(df) == 0:
        print("    EMPTY")
        return

    required = {"chunk_id", "doc_id", "title", "category", "subcategory",
                "campus", "source_collection", "raw_content"}
    missing = required - set(df.columns)
    if missing:
        print(f"    WARN missing cols: {missing}")

    if "raw_content" in df.columns:
        L = df["raw_content"].fillna("").str.len()
        print(f"    chars: min={L.min()}, mean={L.mean():.0f}, p95={L.quantile(0.95):.0f}, max={L.max()}")
        empty = (L == 0).sum()
        if empty:
            print(f"    WARN empty raw_content: {empty}")

    if "chunk_id" in df.columns:
        dup = len(df) - df["chunk_id"].nunique()
        if dup:
            print(f"    WARN duplicate chunk_id: {dup}")

    for col in ["category", "campus", "source_collection"]:
        if col in df.columns:
            null = df[col].isna().sum()
            if null:
                print(f"    WARN null {col}: {null}")

    out_path = CHECK_DIR / f"step_{name}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    saved: {out_path.name}")

    print(f"    samples (first {sample_n}):")
    for i in range(min(sample_n, len(df))):
        row = df.iloc[i]
        cid = row.get("chunk_id", "?")
        title = str(row.get("title", ""))[:40]
        body = str(row.get("raw_content", ""))[:120].replace("\n", " ")
        print(f"      [{cid}] {title} -> {body}")


def process_학칙(df_학칙):
    print(f"\n[1/7] 학칙_대학 처리... 원본 {len(df_학칙)}행")
    df = df_학칙.copy()
    df["내용"] = df["내용"].fillna("").astype(str)

    is_deleted = df["내용"].str.strip().isin(["(삭제)", "삭제"])
    print(f"  삭제 행 제거: {is_deleted.sum()}")
    df = df[~is_deleted].reset_index(drop=True)

    def parse_category(cat):
        if pd.isna(cat):
            return None, None
        parts = [p.strip() for p in str(cat).split(">")]
        chapter = next((p for p in parts if re.match(r"제\d+장", p)), None)
        article = next((p for p in parts if re.match(r"제\d+조", p)), None)
        return chapter, article

    df[["장명", "조항명"]] = df["카테고리"].apply(lambda c: pd.Series(parse_category(c)))

    has_article = df["조항명"].notna()
    rows = []
    LONG = 500
    for article, group in df[has_article].groupby("조항명", sort=False):
        chapter = group["장명"].iloc[0]
        first_id = group["ID"].iloc[0]
        items = []
        for _, r in group.iterrows():
            t = r["제목"]
            content = clean_text(r["내용"])
            if not content:
                continue
            label = f"【{t}】" if re.match(r"^\d+항$", str(t)) else ""
            items.append((label, content, r["ID"]))
        if not items:
            continue

        total = sum(len(c) for _, c, _ in items)
        if total <= LONG or len(items) == 1:
            body = "\n".join(f"{lbl} {c}".strip() for lbl, c, _ in items)
            rows.append({
                "chunk_id": f"학칙_{first_id}",
                "doc_id": first_id,
                "title": article,
                "chapter": chapter or "",
                "category": "학칙",
                "subcategory": chapter or "",
                "campus": "전체",
                "source_collection": "학칙_조항",
                "raw_content": body,
            })
        else:
            for lbl, content, src_id in items:
                body = f"{article} {lbl} {content}".strip()
                rows.append({
                    "chunk_id": f"학칙_{src_id}",
                    "doc_id": first_id,
                    "title": f"{article} {lbl}".strip(),
                    "chapter": chapter or "",
                    "category": "학칙",
                    "subcategory": chapter or "",
                    "campus": "전체",
                    "source_collection": "학칙_조항",
                    "raw_content": body,
                })

    no_article = df[~has_article]
    for _, r in no_article.iterrows():
        content = clean_text(r["내용"])
        if not content or len(content) < 5:
            continue
        rows.append({
            "chunk_id": f"학칙_{r['ID']}",
            "doc_id": r["ID"],
            "title": r["제목"],
            "chapter": "",
            "category": "학칙",
            "subcategory": str(r["카테고리"]).split(">")[-1].strip() if pd.notna(r["카테고리"]) else "",
            "campus": "전체",
            "source_collection": "학칙_조항",
            "raw_content": content,
        })

    result = pd.DataFrame(rows)
    verify_chunks("1_학칙", result)
    return result


def process_regulations(df_reg, hakchik_titles):
    print(f"\n[2/7] regulations 처리... 원본 {len(df_reg)}행")
    df = df_reg.copy()
    df["내용"] = df["내용"].fillna("").astype(str)

    rows = []
    dup_count = 0
    for _, r in df.iterrows():
        content = clean_text(r["내용"])
        if not content or len(content) < 5:
            continue
        if r["제목"] in hakchik_titles:
            dup_count += 1
        rows.append({
            "chunk_id": f"reg_{r['ID']}",
            "doc_id": r["ID"],
            "title": r["제목"],
            "chapter": "",
            "category": "학칙",
            "subcategory": "본문",
            "campus": "전체",
            "source_collection": "학칙_조항",
            "raw_content": content,
        })
    print(f"  학칙_대학과 제목 중복: {dup_count} (보존)")

    result = pd.DataFrame(rows)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n【", "\n\n", "\n", ". ", " "]
    )
    expanded = []
    for _, r in result.iterrows():
        if len(r["raw_content"]) <= MAX_CHUNK_CHARS:
            expanded.append(r.to_dict())
        else:
            for i, ch in enumerate(splitter.split_text(r["raw_content"])):
                d = r.to_dict()
                d["chunk_id"] = f"{r['chunk_id']}_part{i}"
                d["raw_content"] = ch
                expanded.append(d)
    result = pd.DataFrame(expanded)
    verify_chunks("2_regulations", result)
    return result


def process_static_info(df_static):
    print(f"\n[3/7] static_info 처리... 원본 {len(df_static)}행")
    df = df_static.copy()
    df["내용"] = df["내용"].fillna("").astype(str)

    null_mask = df["카테고리"].isna()
    is_faq = df["제목"].fillna("").str.startswith("Q.")
    auto_faq = null_mask & is_faq
    print(f"  카테고리 NULL → FAQ 자동 분류: {auto_faq.sum()}")
    df.loc[auto_faq, "카테고리"] = "FAQ"
    df.loc[df["카테고리"].isna(), "카테고리"] = "기타"
    df["캠퍼스"] = df["캠퍼스"].fillna("전체")

    collection_map = {
        "FAQ": "FAQ",
        "전화번호부": "시설_연락처", "기숙사": "시설_연락처",
        "학생식당": "시설_연락처", "편의시설": "시설_연락처",
        "체육시설": "시설_연락처", "시설대여": "시설_연락처",
        "시설 예약": "시설_연락처", "기타 시설": "시설_연락처",
        "디지털 서비스": "시설_연락처", "특수 서비스": "시설_연락처",
        "캠퍼스": "시설_연락처",
        "장학금": "장학금",
        "학과": "학과정보", "졸업요건": "학과정보",
        "수강신청": "학사정보", "수강 변경": "학사정보",
        "학적 변동": "학사정보", "학생서비스": "학사정보",
        "국제교류": "학사정보", "시스템": "학사정보",
        "기타": "기타",
    }

    rows = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )

    for _, r in df.iterrows():
        content = clean_text(r["내용"])
        if not content or len(content) < 3:
            continue

        cat = str(r["카테고리"])
        collection = collection_map.get(cat, "기타")
        is_faq_row = (cat == "FAQ")

        base = {
            "doc_id": r["ID"],
            "title": str(r["제목"]),
            "chapter": "",
            "category": cat,
            "subcategory": str(r["서브카테고리"]) if pd.notna(r["서브카테고리"]) else "",
            "campus": str(r["캠퍼스"]),
            "source_collection": collection,
        }

        if is_faq_row or len(content) <= MAX_CHUNK_CHARS * 1.2:
            base["chunk_id"] = f"si_{r['ID']}"
            base["raw_content"] = content
            rows.append(base)
        else:
            for i, ch in enumerate(splitter.split_text(content)):
                d = dict(base)
                d["chunk_id"] = f"si_{r['ID']}_part{i}"
                d["raw_content"] = ch
                rows.append(d)

    result = pd.DataFrame(rows)
    print(f"  컬렉션별: {result['source_collection'].value_counts().to_dict()}")
    verify_chunks("3_static_info", result)
    return result


def process_academic_info(df_aca):
    print(f"\n[4/7] academic_info 처리... 원본 {len(df_aca)}행")
    df = df_aca.copy()
    df["내용"] = df["내용"].fillna("").astype(str)

    rows = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )

    for _, r in df.iterrows():
        content = r["내용"]
        title = str(r["제목"])
        first_line = content.split("\n", 1)[0].strip()
        if first_line == title or first_line.startswith(title):
            content = content.split("\n", 1)[1] if "\n" in content else content
        content = clean_text(content)

        if not content:
            continue

        base = {
            "doc_id": r["ID"],
            "title": title,
            "chapter": "",
            "category": str(r["카테고리"]),
            "subcategory": str(r["서브카테고리"]) if pd.notna(r["서브카테고리"]) else "",
            "campus": str(r["캠퍼스"]) if pd.notna(r["캠퍼스"]) else "전체",
            "source_collection": "학사정보",
        }

        if "■" in content:
            sections = re.split(r"(?=^■)", content, flags=re.MULTILINE)
            sections = [s.strip() for s in sections if s.strip()]
            for i, sec in enumerate(sections):
                if len(sec) > MAX_CHUNK_CHARS * 1.5:
                    for j, sub in enumerate(splitter.split_text(sec)):
                        d = dict(base)
                        d["chunk_id"] = f"aca_{r['ID']}_s{i}_p{j}"
                        d["raw_content"] = sub
                        rows.append(d)
                else:
                    d = dict(base)
                    d["chunk_id"] = f"aca_{r['ID']}_s{i}"
                    d["raw_content"] = sec
                    rows.append(d)
        else:
            if len(content) <= MAX_CHUNK_CHARS:
                d = dict(base)
                d["chunk_id"] = f"aca_{r['ID']}"
                d["raw_content"] = content
                rows.append(d)
            else:
                for i, ch in enumerate(splitter.split_text(content)):
                    d = dict(base)
                    d["chunk_id"] = f"aca_{r['ID']}_p{i}"
                    d["raw_content"] = ch
                    rows.append(d)

    result = pd.DataFrame(rows)
    verify_chunks("4_academic_info", result)
    return result


def process_curriculum(df_cur):
    print(f"\n[5/7] curriculum 처리... 원본 {len(df_cur)}행")
    df = df_cur.copy()
    df["내용"] = df["내용"].fillna("").astype(str)

    rows = []
    LOW_CONF_THRESHOLD = 100
    for _, r in df.iterrows():
        content = clean_text(r["내용"])
        if not content:
            continue
        text_only = re.sub(r"https?://\S+", "", content).strip()
        is_low = len(content) < LOW_CONF_THRESHOLD or len(text_only) < 50
        rows.append({
            "chunk_id": f"cur_{r['ID']}",
            "doc_id": r["ID"],
            "title": str(r["제목"]),
            "chapter": "",
            "category": "교육과정",
            "subcategory": "로드맵",
            "campus": str(r["캠퍼스"]),
            "source_collection": "교육과정",
            "raw_content": content,
            "low_confidence": bool(is_low),
        })

    result = pd.DataFrame(rows)
    n_low = result["low_confidence"].sum() if len(result) else 0
    print(f"  low_confidence=True: {n_low}/{len(result)}")
    verify_chunks("5_curriculum", result)
    return result


def process_calendar_from_json():
    print(f"\n[6/7] calendar 처리... (JSON 소스)")
    json_path = INPUT_DIR / "calendar.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  원본 JSON: {len(data)} entries")

    rows = []
    n_range = n_single = n_text_only = 0
    for item in data:
        cid = item.get("id") or item.get("ID")
        title = str(item.get("title") or item.get("제목") or "").strip()
        content = str(item.get("content") or item.get("내용") or "").strip()
        sub = str(item.get("sub_category") or item.get("서브카테고리") or "").strip()
        campus = str(item.get("campus") or item.get("캠퍼스") or "전체").strip()
        if not campus:
            campus = "전체"

        start_date = None
        end_date = None
        m_range = RANGE_PAT.search(content)
        if m_range:
            start_date = m_range.group(1)
            end_date = m_range.group(2)
            n_range += 1
        else:
            tokens = DATE_TOKEN.findall(content)
            if tokens:
                start_date = tokens[0]
                if len(tokens) >= 2 and tokens[-1] != tokens[0]:
                    end_date = tokens[-1]
                    n_range += 1
                else:
                    n_single += 1
            else:
                n_text_only += 1

        if content:
            synthetic = f"{title} ({sub}, {campus}캠퍼스): {content}"
        else:
            synthetic = f"{title} ({sub}, {campus}캠퍼스). 자세한 일정은 학사일정 페이지를 확인하세요."

        rows.append({
            "chunk_id": f"cal_{cid}",
            "doc_id": cid,
            "title": title,
            "chapter": "",
            "category": "학사일정",
            "subcategory": sub,
            "campus": campus,
            "source_collection": "학사일정",
            "raw_content": synthetic,
            "start_date": start_date,
            "end_date": end_date,
        })

    print(f"  날짜 추출: 범위={n_range}, 단일={n_single}, 없음={n_text_only}")
    result = pd.DataFrame(rows)
    verify_chunks("6_calendar", result)
    has_start = result["start_date"].notna().sum()
    has_end = result["end_date"].notna().sum()
    print(f"  start_date 보유: {has_start}/{len(result)}, end_date 보유: {has_end}/{len(result)}")
    return result


def classify_lecture_subject(title):
    title = str(title)
    if any(k in title for k in ["영어", "토익", "중국어", "스페인어", "일본어", "한자", "독일어"]):
        return "어학"
    if any(k in title for k in ["간호", "의료", "병리", "방사선", "치위생", "물리치료", "응급", "약리", "보건"]):
        return "보건의료"
    if any(k in title for k in ["철학", "역사", "인문", "문화", "이미지", "한시"]):
        return "인문"
    if any(k in title for k in ["스포츠", "체육"]):
        return "체육"
    if any(k in title for k in ["코딩", "프로그래밍", "AI", "데이터", "IT", "컴퓨터"]):
        return "공학·IT"
    if any(k in title for k in ["경영", "회계", "마케팅", "경제"]):
        return "경상"
    return "교양"


def process_lecture_reviews(df_lec):
    print(f"\n[7/7] lecture_reviews 처리... 원본 {len(df_lec)}행")
    df = df_lec.copy()
    df["내용"] = df["내용"].fillna("").astype(str)

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "강의명_헤더"),
            ("##", "section"),
            ("###", "subsection"),
        ],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "- ", ". ", " "]
    )

    rows = []
    for _, r in df.iterrows():
        title = str(r["제목"])
        content = r["내용"]
        subject = classify_lecture_subject(title)

        try:
            md_chunks = md_splitter.split_text(content)
        except Exception:
            md_chunks = [type("Doc", (), {"page_content": content, "metadata": {}})()]

        chunk_idx = 0
        for md_chunk in md_chunks:
            section_name = md_chunk.metadata.get("section", "기본")
            sub_name = md_chunk.metadata.get("subsection", "")
            section_label = f"{section_name}-{sub_name}" if sub_name else section_name
            sec_content = md_chunk.page_content.strip()
            if not sec_content or len(sec_content) < 10:
                continue

            pieces = char_splitter.split_text(sec_content) if len(sec_content) > MAX_CHUNK_CHARS * 1.5 else [sec_content]

            for piece in pieces:
                rows.append({
                    "chunk_id": f"lec_{r['ID']}_c{chunk_idx}",
                    "doc_id": r["ID"],
                    "title": title,
                    "chapter": section_label,
                    "category": "강의평가",
                    "subcategory": subject,
                    "campus": "전체",
                    "source_collection": "강의평가",
                    "raw_content": piece,
                    "section": section_label,
                })
                chunk_idx += 1

    result = pd.DataFrame(rows)
    n_lectures = result["doc_id"].nunique() if len(result) else 0
    print(f"  강의 수: {n_lectures}, 평균 청크/강의: {len(result)/max(n_lectures,1):.1f}")
    print(f"  분야: {result['subcategory'].value_counts().to_dict()}")
    verify_chunks("7_lecture_reviews", result)
    return result


def make_metadata(row):
    """3개 버그 수정 - 모두 명시적 NaN 검사"""
    meta = {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "category": row["category"],
        "subcategory": row.get("subcategory", "") if pd.notna(row.get("subcategory")) else "",
        "campus": row.get("campus", "") if pd.notna(row.get("campus")) else "",
        "source_collection": row["source_collection"],
    }
    chap = row.get("chapter")
    if pd.notna(chap) and str(chap).strip():
        meta["chapter"] = str(chap)
    sec = row.get("section")
    if pd.notna(sec) and str(sec).strip():
        meta["section"] = str(sec)
    sd = row.get("start_date")
    if pd.notna(sd) and sd:
        meta["start_date"] = str(sd)
    ed = row.get("end_date")
    if pd.notna(ed) and ed:
        meta["end_date"] = str(ed)
    lc = row.get("low_confidence")
    if isinstance(lc, bool) and lc is True:
        meta["low_confidence"] = True
    return meta


def main():
    print("=" * 70)
    print("을지대학교 RAG 전처리 v2")
    print(f"INPUT:  {INPUT_DIR}")
    print(f"OUTPUT: {OUTPUT_DIR}")
    print(f"CHECK:  {CHECK_DIR}")
    print("=" * 70)

    files = {
        "학칙_대학": "지식관리_학칙_대학_2026-04-26.csv",
        "regulations": "지식관리_regulations_2026-04-26.csv",
        "static_info": "지식관리_static_info_2026-04-26.csv",
        "academic_info": "지식관리_academic_info_2026-04-26.csv",
        "curriculum": "지식관리_curriculum_2026-04-26.csv",
        "lecture_reviews": "지식관리_lecture_reviews_2026-04-26.csv",
    }
    dfs = {k: pd.read_csv(INPUT_DIR / v) for k, v in files.items()}
    학칙_titles = set(dfs["학칙_대학"]["제목"].dropna().astype(str))

    chunks = [
        process_학칙(dfs["학칙_대학"]),
        process_regulations(dfs["regulations"], 학칙_titles),
        process_static_info(dfs["static_info"]),
        process_academic_info(dfs["academic_info"]),
        process_curriculum(dfs["curriculum"]),
        process_calendar_from_json(),
        process_lecture_reviews(dfs["lecture_reviews"]),
    ]

    print("\n[8/8] 통합 corpus 생성...")
    all_chunks = pd.concat(chunks, ignore_index=True, sort=False)

    for col in ["title", "chapter", "category", "subcategory", "campus"]:
        if col in all_chunks.columns:
            all_chunks[col] = all_chunks[col].fillna("").astype(str)

    all_chunks["contents"] = all_chunks.apply(
        lambda r: add_prefix(r["raw_content"], r["category"], r.get("campus", ""),
                             r["chunk_id"], r.get("title", "")),
        axis=1
    )
    all_chunks["chunk_chars"] = all_chunks["contents"].str.len()
    all_chunks["est_tokens"] = (all_chunks["chunk_chars"] * 1.3).round().astype(int)

    all_chunks["metadata"] = all_chunks.apply(make_metadata, axis=1)

    autorag_corpus = pd.DataFrame({
        "doc_id": all_chunks["chunk_id"],
        "contents": all_chunks["contents"],
        "metadata": all_chunks["metadata"],
    })

    print(f"\n[저장]")
    print(f"  총 청크: {len(autorag_corpus)}")

    autorag_corpus.to_parquet(OUTPUT_DIR / "corpus.parquet", index=False)
    print(f"  - corpus.parquet ({len(autorag_corpus)} rows)")

    review_cols = ["chunk_id", "source_collection", "category", "subcategory",
                   "campus", "title", "chunk_chars", "est_tokens", "raw_content", "contents"]
    review_cols = [c for c in review_cols if c in all_chunks.columns]
    all_chunks[review_cols].to_csv(OUTPUT_DIR / "corpus_review.csv", index=False, encoding="utf-8-sig")
    print(f"  - corpus_review.csv")

    coll_dir = OUTPUT_DIR / "collections"
    for name, group in all_chunks.groupby("source_collection"):
        safe = name.replace("·", "_").replace(" ", "_")
        cols = [c for c in review_cols if c in group.columns]
        group[cols].to_csv(coll_dir / f"{safe}.csv", index=False, encoding="utf-8-sig")
        print(f"  - collections/{safe}.csv ({len(group)})")

    stats = {
        "total_chunks": int(len(all_chunks)),
        "avg_chars": float(all_chunks["chunk_chars"].mean()),
        "max_chars": int(all_chunks["chunk_chars"].max()),
        "min_chars": int(all_chunks["chunk_chars"].min()),
        "p95_chars": float(all_chunks["chunk_chars"].quantile(0.95)),
        "over_512_tokens": int((all_chunks["est_tokens"] > 512).sum()),
        "by_collection": all_chunks["source_collection"].value_counts().to_dict(),
        "by_category": all_chunks["category"].value_counts().to_dict(),
        "by_campus": all_chunks["campus"].value_counts().to_dict(),
    }

    n_lc = all_chunks["metadata"].apply(
        lambda m: isinstance(m, dict) and m.get("low_confidence") is True
    ).sum()
    n_sd = all_chunks["metadata"].apply(
        lambda m: isinstance(m, dict) and m.get("start_date") is not None
    ).sum()
    n_ed = all_chunks["metadata"].apply(
        lambda m: isinstance(m, dict) and m.get("end_date") is not None
    ).sum()
    stats["low_confidence_chunks"] = int(n_lc)
    stats["with_start_date"] = int(n_sd)
    stats["with_end_date"] = int(n_ed)

    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  - stats.json")

    print("\n" + "=" * 70)
    print(f"총 청크: {stats['total_chunks']}")
    print(f"low_confidence=True: {n_lc}")
    print(f"start_date 보유: {n_sd}, end_date 보유: {n_ed}")
    print(f"평균: {stats['avg_chars']:.0f}자, p95: {stats['p95_chars']:.0f}, 최대: {stats['max_chars']}")
    print(f"512 토큰 초과: {stats['over_512_tokens']}청크 ({stats['over_512_tokens']/stats['total_chunks']*100:.1f}%)")
    print("\n컬렉션별:")
    for k, v in stats["by_collection"].items():
        print(f"  {k}: {v}")
    print("=" * 70)
    return stats


if __name__ == "__main__":
    main()
