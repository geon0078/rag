# Outline 데이터 스키마 v3

> **목적**: Outline 페이지를 RAG 엔진(V4)이 일관되게 검색 + 인용 가능하도록 강제 형식 정의.
> **작성일**: 2026-04-29
> **선행**: outline_audit.json (262 페이지, 7 cross-collection 중복, 23 Q&A heavy)

## 핵심 원칙

1. **Single Source of Truth (SSOT)**: 한 주제 = 한 페이지. 같은 정보가 여러 페이지에 산재하지 않게.
2. **사실 데이터만**: Q&A 형식 (Q. ... A. ...) 금지. 항목별 사실 나열.
3. **예시 답변 X**: "예를 들면", "예시:", "참고:" 같은 narrative 제거.
4. **frontmatter 강제**: 모든 페이지 상단에 type 별 메타데이터.
5. **Outline 자체가 RAG의 master**: corpus.parquet 의존성 제거.

## Collection 구조 (8개)

| Collection | Type | 페이지 수 | 비고 |
|---|---|---:|---|
| 학칙 | regulation | 90 | 조항별 |
| 강의평가 | lecture_review | 121 | 강의별 |
| 학사일정 | calendar | 4 | 학기별 |
| 학사정보 | academic_info | 36 | 주제별 — Q&A 정리 필요 |
| 시설 | facility | 5 | 시설별 — 각 페이지 내 통합 |
| 장학금 | scholarship | 2 | 종류별 |
| 학과정보 | department | 3 | 학과별 |
| 기타 | misc | 1 | 분류 어려운 항목 |

---

## Type 1: regulation (학칙 조항)

```yaml
---
type: regulation
chapter: "제2장 조직과 학생정원"
article_number: "제2조"
article_title: "소재 및 교육조직"
campus: "전체"
effective_start: "2024-03-01"
related: ["제3조", "제4조"]
---

# 제2조 (소재 및 교육조직)

## 1항
본교는 대전광역시에 소재하는 대전캠퍼스와 ...

## 2항
대전캠퍼스에는 의과대학을 두고, ...
```

**필수 필드**: `type, chapter, article_number, article_title, campus`
**선택**: `effective_start, related`

---

## Type 2: lecture_review (강의평가)

```yaml
---
type: lecture_review
lecture_id: "lecture_reviews_0"
lecture_title: "글로벌문화영어"
subject_area: "어학"
campus: "전체"
disclaimer: true
---

# 글로벌문화영어

## 강의 스타일
- 팝송 활용 PPT 기반 수업
- 비대면 영상 강의 (20~30분)

## 시험·평가
- 중간/기말: 객관식 + 단답
- 출석: 매주 퀴즈 포함

## 과제
- 학기당 3회 (보고서)

## 학점 분포
- A: 30%, B: 50%, C: 20%
```

**필수**: `type, lecture_id, lecture_title, subject_area, campus, disclaimer:true`
**섹션 (있는 것만)**: 강의 스타일, 시험·평가, 과제, 학점 분포, 출석

> ⚠️ 본 페이지는 학생 의견이며 객관적 사실이 아닙니다.

---

## Type 3: facility (시설 — 학식당, 도서관 등 통합)

```yaml
---
type: facility
name: "학식당"
aliases: ["학식", "학생식당", "급식"]
campus: "성남"
building: "뉴밀레니엄센터"
floor: "지하 1층"
phone: "031-740-7727"
operating_hours:
  weekday: "11:00-19:00"
  weekend: "휴무"
---

# 학식당

## 위치
- 캠퍼스: 성남
- 건물: 뉴밀레니엄센터
- 층: 지하 1층

## 운영시간
- 평일: 11:00 ~ 19:00
- 주말·공휴일: 휴무

## 메뉴
- 학교 홈페이지 → 공지사항 → 식당메뉴
- 카카오워크 → 학생식당메뉴

## 가격
- 일반식: 5,500원
- 식권: 기숙사 A동 입소생 120장, B동 입소생 100장 무료 제공

## 부가 서비스
- 테이크아웃 가능 (도시락 용기 무료 제공)
- 7대 알레르기 식품 표시
- 결제 수단: 식권, 카드

## 문의
- 구내식당 및 매점: 031-740-7727
```

**필수**: `type, name, campus`
**선택**: `aliases, building, floor, phone, operating_hours`

---

## Type 4: academic_info (학사정보 — Q&A 변환됨)

```yaml
---
type: academic_info
topic: "졸업요건"
campus: "전체"
related: ["학점", "졸업인증제"]
applies_to: "all_except_의학과"
---

# 졸업요건

## 학점 요구
- 단일전공: 130학점 이상
- 다전공: 주전공 42학점 + 복수전공 등 포함 70학점 이상
- 교양: 30학점 이상
- 핵심영역 교양 교과목 이수 필수

## 추가 요구
- 졸업논문/시험 통과 (학과별 운영)
- 졸업인증제 이수

## 절차
1. 학점 충족 확인 (학생포털)
2. 졸업논문/시험 신청
3. 졸업인증제 등록
4. 학위수여 신청

## 예외 학과
- 의학과: 별도 졸업요건 (의학과 행정실 문의)
- 간호학과: 130학점 + 임상실습 별도
```

**필수**: `type, topic, campus`
**선택**: `related, applies_to`

> ⚠️ 변환 규칙 (FAQ → academic_info):
> - "Q. ..." → 섹션 헤더 (## 학점 요구, ## 절차)
> - "A. 예를 들어..." → 사실만 추출 (예시 제거)
> - 여러 Q&A → 한 페이지 내 의미 그룹별 ## 섹션

---

## Type 5: scholarship (장학금)

```yaml
---
type: scholarship
name: "성적우수 장학금"
campus: "전체"
eligibility:
  gpa: 3.8
  semester: "직전 학기"
amount: "등록금 20%"
application_window:
  semester_1: "2월 1-28일"
  semester_2: "8월 1-31일"
contact: "학생지원팀"
contact_phone: "031-740-XXXX"
---

# 성적우수 장학금

## 자격
- 직전 학기 평점 3.8 이상
- 재학 상태 (휴학 X)

## 금액
- 등록금의 20% 면제

## 신청 방법
1. 학생포털 → 장학 → 장학신청
2. 매 학기 시작 전 신청 기간 내

## 신청 기간
- 1학기: 매년 2월 1일 ~ 2월 28일
- 2학기: 매년 8월 1일 ~ 8월 31일

## 결과 통보
- 신청 후 약 3주 이내
- 본인 계좌로 통보

## 문의
- 학생지원팀
```

---

## Type 6: calendar (학사일정)

```yaml
---
type: calendar
semester: "2026-1"
year: 2026
campus: "성남"
---

# 2026학년도 1학기 학사일정

## 등록·수강신청
- 2026-02-03 ~ 2026-02-05: 재학생 수강바구니(1차)
- 2026-02-19 ~ 2026-02-22: 재학생 수강바구니(2차)
- 2026-02-23 ~ 2026-02-27: 등록
- 2026-03-03 ~ 2026-03-09: 수강정정

## 개강·입학
- 2026-03-03: 개강

## 시험
- 2026-04-21 ~ 04-25: 중간고사
- 2026-06-09 ~ 06-15: 기말고사

## 종강·방학
- 2026-06-21: 종강
- 2026-06-22 ~ 08-31: 여름방학
```

**필수**: `type, semester, year, campus`

---

## Type 7: department (학과정보)

```yaml
---
type: department
name: "의료IT학과"
college: "보건과학대학"
campus: "성남"
location: "뉴밀레니엄센터 4층"
phone: "031-740-7310"
---

# 의료IT학과

## 소속
- 보건과학대학

## 위치
- 캠퍼스: 성남
- 건물: 뉴밀레니엄센터 4층

## 연락처
- 학과 사무실: 031-740-7310

## 졸업요건
- 130학점 이상
- 전공 70학점 (전필 포함)
- 교양 30학점

## 교육과정
- 교육과정 로드맵 페이지 참조 (상세 과목 list)
```

**필수**: `type, name, college, campus`

---

## Type 8: misc (기타)

```yaml
---
type: misc
title: "자연계열학부"
campus: "성남"
---

# 자연계열학부 (기타)
...
```

---

## 변환 규칙 — 기존 → v3

| 변경 | 작업 |
|---|---|
| 본문 시작에 박힌 머리글 (`[학사 \| doc_id \| si_static_info_X]`) | 제거 |
| `Q. ... A. ...` | `## 의미 섹션` + `- 사실 항목` |
| `예를 들어, ...`, `예시:` | 제거 (사실만 유지) |
| `**질문:**`, `**답변:**` | 제거 |
| 같은 주제 cross-collection | 단일 페이지로 병합 (학식당 → 시설/학식당) |
| 동일 title 페이지 (3건) | 1개로 병합 |

## 새 V4 인덱스 chunking 전략

- **chunk size**: 600 chars (한국어 ~150 토큰)
- **overlap**: 100 chars
- **chunk 단위**: page → `## 섹션`별 1차 split → 600자 초과 시 2차 split
- **metadata 보존**:
  - frontmatter 의 모든 필드 → chunk metadata
  - chunk 내 `##` 섹션 이름 → `chunk_section`
  - `outline_doc_id`, `outline_url`, `collection` 항상 포함

```python
{
    "doc_id": "outline_<outline_doc_uuid>_c0",
    "parent_doc_id": "outline_<outline_doc_uuid>",
    "contents": "<chunk text>",
    "metadata": {
        "type": "facility",
        "topic_name": "학식당",
        "campus": "성남",
        "outline_url": "http://localhost:3002/doc/...",
        "chunk_section": "운영시간",
        "frontmatter": {"name": "학식당", "phone": "031-740-7727"}
    }
}
```

## 동기화 정책 (Outline ↔ V4)

| 이벤트 | V4 동작 |
|---|---|
| `documents.create` | 새 청크 추가 (Qdrant + BM25 += chunks) |
| `documents.update` | 기존 `parent_doc_id` 청크 삭제 → 새로 청크 |
| `documents.delete` | `parent_doc_id` 청크 삭제 |
| `documents.move` | metadata.collection 갱신 (delete+upsert) |

webhook URL: `http://host.docker.internal:8000/api/sync/outline`
