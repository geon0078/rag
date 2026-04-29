# 을지대 RAG 시스템 — Onyx + Docmost 통합 개발 명세서

> **이 문서의 대상**: Claude Code (구현 담당)
> **작성 목적**: 지금까지 모든 분석·결정·교훈을 통합하여 Onyx(학생 챗봇) + Docmost(운영자 docs) 기반 최종 시스템 구축
> **선행 문서**: 모든 이전 명세서를 본 문서가 통합·대체
> **소요 시간**: 2~3주 (Phase 1: 1주, Phase 2: 조건부 1~2주)
> **작성일**: 2026-04-28

---

## 0. 한 페이지 요약

```
┌─────────────────────────────────────────────────────────────────────┐
│  학생 (chat.euljiu.ac.kr)              운영자 (docs.euljiu.ac.kr)    │
│         ↓                                       ↓                    │
│  ┌──────────────┐                       ┌──────────────┐           │
│  │ Onyx          │                       │ Docmost      │           │
│  │ (Next.js)     │                       │ (NestJS)     │           │
│  │ - 채팅 UI      │                       │ - docs 편집   │           │
│  │ - 디자인 수정  │                       │ - 트리 구조   │           │
│  │ - white-label │                       │ - 협업·이력   │           │
│  └──────┬───────┘                       └──────┬───────┘           │
│         │ 검색·답변 API                         │ webhook + API      │
│         ↓                                       ↓                    │
│         └──────────────┬─────────────────────┘                      │
│                        ↓                                            │
│             ┌────────────────────────┐                              │
│             │ 우리 RAG 백엔드           │                              │
│             │ (FastAPI - 검증 완료)     │                              │
│             │ ├── Solar Pro + 임베딩    │                              │
│             │ ├── HyDE + Groundedness  │                              │
│             │ ├── Qdrant (4096-dim)    │                              │
│             │ ├── BM25 (ko_okt)        │                              │
│             │ └── recall@5 0.852 ✓    │                              │
│             └────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘

핵심: 우리 검증된 RAG는 그대로 유지, Onyx와 Docmost는 UI·docs 도구로만 사용
시간: Phase 1 1주 + Phase 2 조건부 1~2주
라이선스: Onyx MIT + Docmost AGPL + 우리 자체 = 학내 사용 자유
```

---

## 1. 명세서 통합 — 어떤 결정이 어디서 왔나

이전 명세서들의 모든 결정을 반영했습니다. 충돌하는 부분은 최신 결정 우선.

### 1.1 살아있는 결정 (이 명세서 그대로 유지)

| 출처 명세서 | 결정 | 본 문서 위치 |
|---|---|---|
| PROJECT_SPEC.md | Solar Pro·임베딩 사용, Qdrant 벡터 DB, BM25 + ko_okt | §2 (RAG 백엔드 보존) |
| EVAL_TRUST_SPEC.md | 3-Tier 평가셋, Golden Set 신뢰도, Drop ≤ 15pt | §7 (평가) |
| ADVERSARIAL_EVAL_SPEC.md | 250건 Adversarial Set, 통과 기준 | §7 (평가) |
| METADATA_AND_WEB_SPEC.md | 메타데이터 v3, 평탄 dict, 3-Layer 구조 | §3 (데이터) |
| TOPIC_STRUCTURE_SPEC.md | Topic 단위 구조화 (학식당 케이스) | §4 (Topic) |
| ONYX_INTEGRATION_SPEC.md | Onyx의 디자인 수정·LLM 주입 가능성 | §5 (Onyx) |
| PLATFORM_INTEGRATION_SPEC.md | Docmost docs 플랫폼 사용 | §6 (Docmost) |
| `현재 상태 (2026-04-28)` | V4 채택 (cc_w=0.6), recall@10 0.85 완화 | §2.4 (설정) |

### 1.2 폐기된 결정 (이전 명세서에서 변경)

| 폐기된 것 | 폐기 이유 | 대체 |
|---|---|---|
| 자체 docs UI 개발 (METADATA_AND_WEB_SPEC §6) | 시간·비용 큼 | Docmost 사용 |
| Streamlit MVP (METADATA_AND_WEB_SPEC §6) | 폐기 비효율 | 바로 Onyx + Docmost |
| 처방 1: lecture_grouper | 회귀 발생 (-2.2pt) | 롤백, Onyx Vespa 사용 |
| 처방 2: Query Rewriter 단독 | 회귀 발생 (-2.6pt) | 사용 안 함 (V12에서만 효과) |
| recall@10 ≥ 0.95 | 시스템 한계 | 0.85로 완화 |
| USS 8차원 학술 메트릭 | 학생 체감 못 | Adversarial Score 사용 |

### 1.3 새로 결정한 것

| 새 결정 | 근거 |
|---|---|
| **Onyx + Docmost Hybrid** | Onyx는 학생 챗봇, Docmost는 운영자 docs |
| **우리 RAG 백엔드 그대로 유지** | 검증된 점수 보존, 매몰비용 0% |
| **Phase 1 → 평가 → Phase 2 조건부** | 위험 분산, 빠른 진입 + 품질 보장 |
| **Vespa 사용 안 함** | 우리 Qdrant 검증되어 있어 교체 비효율 |
| **Onyx Frontend만 사용** | 백엔드는 우리 FastAPI로 호출 |

---

## 2. 시스템 아키텍처

### 2.1 컴포넌트 구성

```
┌────────────────────────────────────────────────────────────────────┐
│ 학생용 시스템                                                        │
│                                                                    │
│ ┌──────────────────────────────────┐                              │
│ │ Onyx Frontend (Next.js)          │                              │
│ │ - URL: chat.euljiu.ac.kr         │                              │
│ │ - 채팅 UI (검증된 디자인)          │                              │
│ │ - White-label: 을지대 브랜딩      │                              │
│ │ - 백엔드 호출: /api/onyx/* → 우리 RAG│                              │
│ └──────────────────────────────────┘                              │
└────────────────────────────────────────────────────────────────────┘
                          ↕ REST API
┌────────────────────────────────────────────────────────────────────┐
│ 운영자용 시스템                                                       │
│                                                                    │
│ ┌──────────────────────────────────┐                              │
│ │ Docmost (NestJS + React)          │                              │
│ │ - URL: docs.euljiu.ac.kr          │                              │
│ │ - Spaces (영역) → Pages (Topic)   │                              │
│ │ - TipTap 위지윅 편집기             │                              │
│ │ - 변경 이력, 협업                  │                              │
│ │ - webhook: 페이지 수정 시 RAG 알림  │                              │
│ └──────────────────────────────────┘                              │
└────────────────────────────────────────────────────────────────────┘
                          ↕ webhook + REST API
┌────────────────────────────────────────────────────────────────────┐
│ 공통 RAG 백엔드 (우리 시스템 - 100% 보존)                              │
│                                                                    │
│ ┌──────────────────────────────────────────────────────────────┐  │
│ │ FastAPI                                                       │  │
│ │ ├── /api/onyx/chat                  (Onyx 호출 호환)           │  │
│ │ ├── /api/onyx/search                (Onyx 호출 호환)           │  │
│ │ ├── /api/sync/docmost               (Docmost webhook 수신)    │  │
│ │ ├── /api/admin/*                    (운영 도구)                │  │
│ │ └── Solar Pro + HyDE + Groundedness                           │  │
│ └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ ┌──────────────┐  ┌─────────────┐  ┌────────────────────────┐     │
│ │ PostgreSQL   │  │ Qdrant      │  │ Redis (Celery + cache) │     │
│ │ - 메타·콘텐츠  │  │ - 벡터 인덱스│  │                        │     │
│ │ - Topic 구조  │  │ - BM25 인덱스│  │                        │     │
│ └──────────────┘  └─────────────┘  └────────────────────────┘     │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 docker-compose 전체 구성

```yaml
services:
  # ===== 우리 RAG 백엔드 =====
  rag-api:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - UPSTAGE_API_KEY=${UPSTAGE_API_KEY}
      - DB_URL=postgresql://rag:rag@postgres:5432/euljiu_rag
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - DOCMOST_API_URL=http://docmost:3000
      - DOCMOST_API_KEY=${DOCMOST_API_KEY}
    depends_on: [postgres, qdrant, redis]

  rag-worker:
    build: ./backend
    command: celery -A worker worker --loglevel=info
    environment: # rag-api와 동일

  postgres:
    image: postgres:16
    volumes: [postgres_data:/var/lib/postgresql/data]

  qdrant:
    image: qdrant/qdrant:latest
    volumes: [qdrant_data:/qdrant/storage]

  redis:
    image: redis:7-alpine

  # ===== Onyx Frontend (학생 챗봇) =====
  # full Onyx가 아니라 Frontend만 fork해서 사용
  onyx-web:
    build: ./onyx-web   # Onyx frontend fork
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://rag-api:8000/api/onyx
      - NEXT_PUBLIC_BRAND_NAME=을지대 학사 도우미

  # ===== Docmost (운영자 docs) =====
  docmost:
    image: docmost/docmost:latest
    ports: ["3001:3000"]
    environment:
      - DATABASE_URL=postgresql://docmost:docmost@docmost-postgres:5432/docmost
      - REDIS_URL=redis://redis:6379
      - APP_URL=http://localhost:3001
      - APP_SECRET=${DOCMOST_SECRET}
      - WEBHOOK_URL=http://rag-api:8000/api/sync/docmost
    depends_on: [docmost-postgres, redis]

  docmost-postgres:
    image: postgres:16
    volumes: [docmost_pg_data:/var/lib/postgresql/data]

volumes:
  postgres_data:
  qdrant_data:
  docmost_pg_data:
```

### 2.3 라이선스 정리 (학내 사용 모두 OK)

| 컴포넌트 | 라이선스 | 학내 사용 | 외부 학교 SaaS |
|---|---|---|---|
| 우리 RAG 백엔드 | 자체 (비공개 OK) | ✅ | ✅ |
| Onyx Frontend (CE) | MIT | ✅ | ✅ |
| Docmost CE | AGPL 3.0 | ✅ | ⚠️ 코드 공개 의무 |
| Solar Pro/임베딩 | API 사용 | ✅ | ✅ (API 키 별도) |

학내 운영만 하면 모두 자유. 외부 학교에 SaaS 제공할 경우만 Docmost AGPL 주의.

### 2.4 RAG 핵심 설정 (V4 — 검증 완료)

```python
# src/config.py — 변경 금지
top_k_dense = 30
top_k_sparse = 30
top_k_rerank_final = 5
hybrid_method = "cc"
hybrid_cc_weight = 0.6        # ★ V4 베스트
hybrid_cc_normalize = "mm"
reranker_enabled = False       # GPU 환경이면 True 검토
hyde_enabled = True            # off 시 grounded -3.3pt
rewrite_enabled = False        # 회귀, 사용 안 함
default_campus = "성남"
```

**현재 검증된 점수** (Adversarial 250건):
- recall@5: 0.852 ✓
- recall@10: 0.868 (목표 0.85 완화 후 통과)
- MRR: 0.678 ✓
- citation: 1.000 ✓
- grounded: 0.960 ✓

---

## 3. 데이터 구조 — 메타데이터 v3 + Topic

이전 명세서의 결정을 그대로 유지.

### 3.1 메타데이터 v3 (METADATA_AND_WEB_SPEC.md §3 그대로)

3-Layer 구조 + 평탄 dict:

```python
# 청크 1개 예시 (학식당 운영시간)
{
    # Layer 1: Core (필수, 모든 청크)
    "doc_id": "topic_facility_cafeteria_section_hours",
    "parent_doc_id": "topic_facility_cafeteria",
    "path": "시설/학식당/운영시간",
    "breadcrumb": ["시설", "학식당", "운영시간"],
    "schema_version": "v3",
    "source_collection": "시설_연락처",
    "category": "시설",
    "subcategory": "식음료",
    "title": "학식당 운영시간",
    "campus": "성남",
    "language": "ko",
    "chunk_index": 1,
    "chunk_count": 7,
    "depth": 3,

    # Layer 1 추가: Topic 정보
    "topic_id": "topic_facility_cafeteria",
    "topic_name": "학식당",
    "topic_section": "운영시간",
    "topic_section_order": 1,

    # Layer 2: Domain (시설 컬렉션 전용)
    "phone": null,
    "building": "본관",
    "floor": "1층",
    "facility_type": "식당",

    # Layer 3: Operations
    "effective_start": "2024-03-01",
    "effective_end": null,
    "created_at": "2024-02-15T00:00:00+09:00",
    "indexed_at": "2026-04-28T10:00:00+09:00",
    "last_verified_at": "2026-04-15",
    "version": 3,
    "owner": "총무팀",
    "confidence": "high",

    # Docmost 연결 (새 추가)
    "docmost_page_id": "page_abc123",
    "docmost_space_id": "space_facility"
}
```

### 3.2 Topic 구조 (TOPIC_STRUCTURE_SPEC.md 그대로)

```
[Topic]              ← Docmost Page (예: "학식당")
  ├─ [Section]       ← Page 내 ## 헤더 (예: "운영시간", "메뉴")
  │   └─ [Chunk]     ← 검색·임베딩 단위
  └─ [Section]
      └─ [Chunk]
```

전체 약 **320개 Topic** (학식당·기숙사·도서관 등).

### 3.3 Docmost ↔ 우리 메타 매핑

| Docmost 개념 | 우리 메타 |
|---|---|
| Space | topic_area (영역) |
| Page | topic_id, topic_name |
| Page 제목 | title |
| Page 내 ## 헤더 | topic_section |
| Page URL slug | path 일부 |
| Page 본문 | contents (마크다운) |

---

## 4. Phase 분할 — 단계적 진입

### 4.1 전체 흐름

```
[현재] 우리 RAG 백엔드 검증 완료 (recall@5 0.852)
   ↓
Phase 1 (Week 1): UI 통합 + 평가
   ├── Onyx Frontend white-label
   ├── Docmost 셋업 + 데이터 마이그레이션
   ├── webhook + API 연결
   └── Adversarial 250 재평가
   ↓
Day 7: 결정 포인트
   ├── 통과 → 운영 진입 (Phase 2 불필요)
   ├── 부분 미달 → Phase 2 (Topic 보강만)
   └── 다 미달 → 디버그 또는 자체 UI 복귀
   ↓
Phase 2 (Week 2-3, 조건부): 깊은 통합
   ├── Topic 단위 Auto-Merging Retriever
   ├── Onyx 컴포넌트 fork (Topic 트리 추가)
   └── 최종 평가
   ↓
운영 진입
```

### 4.2 Phase 별 산출물

| Phase | 산출물 | 평가 기준 |
|---|---|---|
| Phase 1 | UI 통합·기본 동작 | Adversarial 점수 V4 대비 ±5pt 이내 |
| Phase 2 (선택) | Topic 검색·UI 보강 | 학식당 케이스 통합 답변, recall +5pt |
| 운영 | 익명 피드백 누적 | (운영 시점에 추가 평가) |

---

## 5. Phase 1 — Week 1 상세 일정

### Day 1: 인프라 셋업 (8시간)

#### 오전 (4시간)

- [ ] 프로젝트 디렉토리 구조 결정
  ```
  euljiu-rag/
  ├── backend/                # 우리 RAG (그대로)
  ├── onyx-web/               # Onyx frontend fork
  ├── docker-compose.yml
  └── .env
  ```
- [ ] Docmost docker-compose 띄우기 + 학식당 페이지 1개 수동 생성, UX 검증
- [ ] Onyx full version (Lite 아님) docker-compose로 확인 (frontend 코드 분석용)

#### 오후 (4시간)

- [ ] Onyx 프론트엔드 코드 fork
  ```bash
  git clone https://github.com/onyx-dot-app/onyx
  cp -r onyx/web ./onyx-web
  cd onyx-web && npm install
  ```
- [ ] Onyx 프론트엔드의 백엔드 API 호출 부분 분석
  - `web/src/lib/chat/*.ts` (채팅 API 호출)
  - `web/src/lib/search/*.ts` (검색 API 호출)
- [ ] 환경변수 정리, NEXT_PUBLIC_API_URL 우리 FastAPI로 매핑

### Day 2: 우리 RAG → Onyx API 호환 어댑터 (8시간)

Onyx 프론트가 기대하는 API 형식에 맞춰 우리 FastAPI에 어댑터 엔드포인트 추가.

#### 핵심 엔드포인트 매핑

| Onyx 프론트 호출 | 우리 어댑터 → 내부 호출 |
|---|---|
| `POST /chat/send-message` | `/api/onyx/chat` → 우리 RAG `run()` |
| `POST /search` | `/api/onyx/search` → 우리 retriever |
| `GET /chat/sessions` | `/api/onyx/sessions` → PostgreSQL chat_sessions |
| `POST /chat/create-chat-session` | `/api/onyx/sessions/create` |
| `GET /persona` | `/api/onyx/persona` → 단일 persona 응답 |

#### 주요 변환 로직

- [ ] `/api/onyx/chat` 엔드포인트 추가
  - Onyx 형식 요청 → 우리 RAG 형식 변환
  - 우리 RAG 답변 → Onyx 형식 응답 (citations 포함)
- [ ] streaming 응답 호환 (Onyx는 SSE 사용)
- [ ] citations 형식 매핑
  - 우리: `[출처: doc_id, 카테고리, 캠퍼스]`
  - Onyx: `{document_id, link, source_type}` 객체 배열
- [ ] PostgreSQL에 `onyx_compatible` 채팅 세션 테이블 추가

### Day 3: Docmost ↔ RAG 동기화 (8시간)

#### 마이그레이션 스크립트 (오전, 4시간)

- [ ] `scripts/migrate_to_docmost.py`
  - 우리 PostgreSQL의 청크들을 Topic 단위로 그룹화 (BERTopic 결과 활용)
  - 각 Topic을 Docmost Space + Page로 변환
  - Topic Section을 Page 내 ## 헤더로 변환
  - Docmost API로 일괄 생성

#### 마이그레이션 매핑

```
Topic "학식당" + 7 sections
   ↓
Docmost Page "학식당" (Space "시설" 안에)
   본문:
   ## 위치
   학식당은 본관 1층...
   ## 운영시간
   평일 11:00~19:00...
   ## 메뉴
   ...
```

#### Webhook 엔드포인트 (오후, 4시간)

- [ ] `POST /api/sync/docmost` 추가
- [ ] Docmost webhook 페이로드 검증
- [ ] 페이지 변경 감지 → Docmost API에서 전체 마크다운 가져옴
- [ ] 마크다운 → ## 헤더 단위로 청크 분할
- [ ] 메타데이터 v3 변환 (`docmost_page_id` 채움)
- [ ] Celery 큐에 인덱싱 작업 등록 (Qdrant + BM25 갱신)

### Day 4: Onyx Frontend 디자인 적용 (6시간)

#### Level 1: White-label (1시간)

Onyx Admin Panel 설정 또는 환경변수:
- 로고 (을지대)
- App Name: "을지대 학사 도우미"
- Primary Color (을지대 브랜드)
- favicon

#### Level 2: Tailwind 테마 (3시간)

- [ ] `onyx-web/tailwind.config.js` 색상 팔레트
- [ ] `onyx-web/app/globals.css` CSS 변수
- [ ] Pretendard 폰트 추가 (한국어 친화)
- [ ] 다크모드 색상 보정

#### Level 2: 불필요 컴포넌트 숨김 (2시간)

- [ ] Slack/Drive/GitHub 커넥터 UI 제거 또는 숨김
- [ ] AI Agent 생성 메뉴 단순화
- [ ] 학생용에 불필요한 Admin 기능 비활성

### Day 5: HyDE + Groundedness 통합 검증 (4시간)

이미 우리 RAG 백엔드에 있는 기능. Onyx 프론트에서 정상 작동하는지 확인.

- [ ] 학생 질문 → Onyx → 우리 RAG → HyDE → Qdrant → 답변 → Groundedness
- [ ] 답변 형식 (citations) 확인
- [ ] Onyx UI에서 출처 클릭 시 Docmost 페이지로 이동 (URL 매핑)
  ```
  [출처: topic_facility_cafeteria_section_hours]
       ↓ 클릭
  https://docs.euljiu.ac.kr/s/space_facility/p/page_cafeteria
  ```

### Day 6: Adversarial 250건 재평가 (4시간)

이전 명세서의 평가 그대로 사용.

- [ ] eval_adversarial.py 실행 (Onyx Frontend 우회, 우리 백엔드 직접)
- [ ] 결과 비교 → 우리 V4 baseline (recall@5 0.852) 대비
- [ ] Onyx Frontend 통한 e2e 테스트 (수동 30건)
- [ ] Docmost에서 페이지 수정 → webhook → 재인덱싱 → 검색 결과 갱신 확인

### Day 7: 결정 + 운영 준비 (4시간)

#### 평가 통과 기준 (모두 충족 시 운영 진입)

| 메트릭 | 기준 | 의미 |
|---|---|---|
| Adversarial recall@5 | ≥ 0.85 | V4 baseline 유지 |
| Adversarial citation | = 1.0 | 출처 표기 정상 |
| Adversarial grounded | ≥ 0.90 | 환각 방지 |
| Onyx UI e2e (수동 30건) | ≥ 28건 OK | 실 사용 가능 |
| Docmost ↔ RAG 동기화 | 정상 | webhook 동작 |

#### 결정 분기

**모두 통과 시**:
- ✅ 운영 진입
- 학내 학생 30명 베타 테스트 (1주)
- 익명 피드백 버튼 추가
- 운영 모니터링 시작

**부분 미달 시 (recall만 약간)**:
- Phase 2의 일부만 진행 (Topic 보강)
- 1~2주 추가 작업

**다 미달 시**:
- 디버그
- 또는 시나리오 변경 (자체 UI 복귀)

---

## 6. Phase 2 — Week 2-3 (조건부)

Phase 1 통과하면 **건너뛰세요.** Phase 1에서 부분 미달인 경우만 진행.

### 6.1 Phase 2-A: Topic 단위 Auto-Merging Retriever (3일)

검색 정확도가 부족할 때:

- [ ] LlamaIndex Auto-Merging Retriever 패턴 적용
  - top-30 청크 → topic_id로 그룹화 → top-5 Topic
  - 각 Topic의 가장 관련된 Section 1~2 expand
- [ ] 학식당 같은 통합 답변 검증
- [ ] Adversarial 재평가

### 6.2 Phase 2-B: Onyx UI에 Topic 트리 추가 (3일)

운영자 요청 또는 학생 사용성 부족 시:

- [ ] 좌측 사이드바에 Topic 트리 추가 (자체 컴포넌트)
- [ ] Topic 클릭 시 해당 Topic의 통합 답변 자동 표시
- [ ] react-arborist로 320 Topic 가상화

### 6.3 Phase 2-C: 평가 + 운영 진입 (1일)

- [ ] 최종 Adversarial 250건 평가
- [ ] 통과 시 운영 진입
- [ ] 미통과 시 보고 + 의사결정자 결정

---

## 7. 평가 (EVAL_TRUST_SPEC.md + ADVERSARIAL_EVAL_SPEC.md 통합)

### 7.1 사용할 평가셋

| 셋 | 규모 | 용도 |
|---|---|---|
| **Adversarial Set** (이미 있음) | 250건 (수작업) | **통과 기준** |
| Synthetic Set (Trial F) | 50건 (paraphrase) | 회귀 감지만 |
| 향후 Production Set | 운영 후 누적 | 6개월 후 |

### 7.2 통과 기준 (변경 금지)

```
Adversarial 250건 기준:
  recall@5  ≥ 0.85  (V4 검증값 0.852 유지)
  recall@10 ≥ 0.85  (이전 0.95에서 완화, 산업 표준 근거)
  MRR       ≥ 0.65
  citation  ≥ 0.90
  grounded  ≥ 0.85

Robustness Drop (Synthetic vs Adversarial):
  ≤ 15pt
```

### 7.3 평가 도구

- [ ] `scripts/eval_adversarial.py` (이미 있음)
- [ ] `scripts/compare_phase1_vs_baseline.py` (Phase 1 후 비교)
- [ ] `scripts/eval_e2e_onyx.py` (Onyx UI 거친 e2e 평가, 신규)

---

## 8. 운영 모니터링 (개발 단계 외 추가)

> **개발 단계에서는 KPI·SSO·RBAC 제외.**
> 본 절은 **운영 진입 후 추가**.

### 8.1 운영 진입 후 추가 항목

- [ ] 학내 SSO (OIDC) — 학교 인증 시스템 연동
- [ ] 익명 피드백 버튼 (👍/👎)
- [ ] 모든 쿼리·답변 익명 로그
- [ ] 6개월 후 Production Set 누적 (200건 목표)
- [ ] 운영 KPI 대시보드 (응답시간 p95, 일일 쿼리, 만족도)

이건 **별도 명세서**로 운영 진입 후 작성.

---

## 9. 컴포넌트별 책임 정리

### 9.1 Onyx Frontend (학생용)

| 책임 | 비책임 |
|---|---|
| 채팅 UI 표시 | 검색 로직 (우리 RAG가) |
| 사용자 입력 받기 | LLM 호출 (우리 RAG가) |
| 답변·출처 표시 | 데이터 저장 (우리 PostgreSQL) |
| 디자인·테마 | 권한 관리 (운영 시 SSO) |
| 채팅 이력 표시 | 콘텐츠 편집 (Docmost가) |

### 9.2 Docmost (운영자용)

| 책임 | 비책임 |
|---|---|
| 페이지 편집 (TipTap) | RAG 검색 (우리 RAG가) |
| 트리 구조 (Spaces) | 학생 질문 처리 |
| 변경 이력 | 답변 생성 |
| 협업 편집 | 학생 UI |
| webhook 발송 | 인덱싱 (우리 RAG가) |

### 9.3 우리 RAG 백엔드 (코어)

| 책임 |
|---|
| Solar Pro LLM 호출 |
| Solar 임베딩 생성 |
| HyDE 쿼리 보강 |
| Qdrant 벡터 검색 |
| BM25 (ko_okt) 검색 |
| RRF (cc w=0.6) fusion |
| Groundedness Check |
| 메타데이터 관리 |
| Topic 구조 관리 |
| Onyx API 어댑터 |
| Docmost webhook 처리 |
| Celery 인덱싱 큐 |

### 9.4 책임 분리의 이점

각 컴포넌트가 **단일 책임**: Onyx 업데이트가 우리 RAG에 영향 없음, Docmost 업데이트도 마찬가지. 우리 RAG 변경해도 둘 다 영향 없음.

---

## 10. 구현 시 자주 하는 실수 8가지

지금까지 분석에서 발견한 함정들. 모두 회피.

### 1. Onyx 백엔드까지 사용

Onyx full을 띄우면 Vespa·MinIO 등 무거운 인프라 강제. **Frontend만 fork해서 우리 RAG에 연결**.

### 2. Outline 선택

BSL 1.1 라이선스로 외부 학교 SaaS 시 위반. **Docmost(AGPL)가 학내·외부 모두 안전**.

### 3. 자체 docs UI 개발

이전 명세서의 자체 개발 안은 4~6주 걸림. **Docmost로 1주에 동등 기능**.

### 4. 검증된 RAG 폐기 후 Onyx 채택

매몰비용 큼. recall@5 0.852 검증된 시스템을 Vespa로 재현하는 건 위험. **우리 백엔드 100% 보존**.

### 5. 처방 1·2 재시도

이전 두 시도 모두 회귀. **건드리지 말 것**. V4 설정 그대로.

### 6. recall@10 0.95 목표 유지

시스템 한계. **0.85로 완화 결정 사항**.

### 7. Phase 1 평가 없이 Phase 2 진행

부분 미달 진단 없이 Phase 2 들어가면 또 회귀. **Day 7 평가 결과로 결정**.

### 8. 메타데이터 v3 무시

Docmost와 우리 RAG의 일관성 핵심. **모든 청크에 v3 메타 강제**.

---

## 11. 참고할 핵심 오픈소스

이전 명세서들에서 정리한 모든 참고 자료를 한 곳에.

### 11.1 핵심 (Phase 1 사용)

| 프로젝트 | 용도 | URL |
|---|---|---|
| **Onyx (Danswer)** | 학생 챗봇 frontend | https://github.com/onyx-dot-app/onyx |
| **Docmost** | 운영자 docs | https://github.com/docmost/docmost |
| **TipTap** | Docmost의 편집기 (Docmost 내부) | https://github.com/ueberdosis/tiptap |
| **LiteLLM** | Solar Pro LLM 연결 (선택) | https://github.com/BerriAI/litellm |

### 11.2 RAG 핵심 (이미 사용 중)

| 프로젝트 | URL |
|---|---|
| Qdrant | https://github.com/qdrant/qdrant |
| KoNLPy (Okt) | https://github.com/konlpy/konlpy |
| bge-reranker-v2-m3-ko | https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko |

### 11.3 Phase 2 (조건부)

| 프로젝트 | 용도 | URL |
|---|---|---|
| **BERTopic** | Topic 자동 발견 (이미 적용) | https://github.com/MaartenGr/BERTopic |
| **LlamaIndex Auto-Merging** | 계층적 검색 | https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/ |

### 11.4 평가

| 프로젝트 | URL |
|---|---|
| **AutoRAG** (한국 Marker-Inc-Korea) | https://github.com/Marker-Inc-Korea/AutoRAG |
| RAGAS | https://github.com/explodinggradients/ragas |

---

## 12. Claude Code 핵심 지시 요약

> **Onyx Frontend (학생 챗봇) + Docmost (운영자 docs) + 우리 RAG 백엔드(그대로 유지)** 통합 시스템 구축. Phase 1 (1주) → 평가 → Phase 2 조건부.
>
> **반드시 지킬 것**:
> - 우리 RAG 백엔드 (Solar Pro + HyDE + Qdrant + ko_okt BM25 + V4 cc_w=0.6) **100% 보존**
> - Onyx **Frontend만** 사용 (full Onyx 띄우지 말 것 — Vespa·MinIO 부담 큼)
> - Docmost는 그대로 사용 (AGPL, 학내 사용 OK)
> - 메타데이터 v3 평탄 dict (이전 명세서)
> - Topic 구조 320개 (학식당 케이스 통합)
> - Adversarial 250건이 통과 기준 (Trial F는 회귀 감지만)
> - V4 default 설정 변경 금지 (`hybrid_cc_weight=0.6`)
> - recall@10 목표 0.85 (0.95 아님)
>
> **Phase 1 (Week 1, Day 1~7)**:
> - Day 1: 인프라 셋업, Onyx Frontend fork, Docmost 띄우기
> - Day 2: 우리 RAG에 Onyx API 호환 어댑터 추가 (`/api/onyx/*`)
> - Day 3: 우리 corpus → Docmost 마이그레이션 + webhook 동기화
> - Day 4: Onyx Frontend 디자인 (white-label + Tailwind)
> - Day 5: HyDE·Groundedness e2e 검증
> - Day 6: Adversarial 250건 재평가
> - Day 7: 결정 (운영 진입 / Phase 2 / 디버그)
>
> **Phase 2 (Week 2-3, 조건부)**:
> - Phase 1 부분 미달 시만 진행
> - Topic Auto-Merging Retriever 또는 Onyx UI 컴포넌트 fork
>
> **하지 말 것**:
> - Onyx full version (Vespa 포함) 사용
> - 처방 1·2 재시도 (확정 회귀)
> - 우리 RAG 컴포넌트를 Onyx Vespa로 교체
> - 자체 docs UI 개발 (Docmost로 대체)
> - Adversarial 평가 없이 운영 진입
> - Outline 사용 (BSL 1.1)
> - Onyx EE 기능 사용 (라이선스 비용)
>
> **Day 7 결정 분기**:
> - 모든 평가 통과 → 운영 진입
> - recall만 미달 → Phase 2-A (Topic 검색 보강)
> - UI 부족 → Phase 2-B (Onyx 컴포넌트 fork)
> - 다 미달 → 디버그 또는 자체 UI 복귀

---

## 13. 명세서 사용 가이드

### 13.1 본 명세서가 통합·대체하는 문서

다음 문서들은 **본 명세서로 통합**되었습니다:

- `PROJECT_SPEC.md` — RAG 핵심은 §2.4에 통합
- `EVAL_SPEC.md` — 평가는 §7에 통합
- `PATCH_SPEC.md` — V4 설정 §2.4에 반영
- `METADATA_AND_WEB_SPEC.md` — Docmost로 대체 §6
- `EVAL_TRUST_SPEC.md` — 평가 신뢰도 §7
- `ADVERSARIAL_EVAL_SPEC.md` — Adversarial 250건 §7
- `USS_AUTORAG_SPEC.md` — 폐기 (피드백 없는 환경 부적합)
- `TOPIC_STRUCTURE_SPEC.md` — Topic 구조 §3-4
- `PLATFORM_INTEGRATION_SPEC.md` — Docmost 채택 §6
- `ONYX_INTEGRATION_SPEC.md` — Onyx Frontend §5

### 13.2 본 명세서로 부족한 경우

**더 자세한 설명이 필요하면 원본 명세서 참조**:

- 메타데이터 v3 디테일 → METADATA_AND_WEB_SPEC.md §3
- Topic 발견 BERTopic 절차 → TOPIC_STRUCTURE_SPEC.md §4
- Onyx 디자인 수정 Level 3 → ONYX_INTEGRATION_SPEC.md §3
- Docmost docker-compose → PLATFORM_INTEGRATION_SPEC.md §5
- Adversarial Tier 분류 → ADVERSARIAL_EVAL_SPEC.md §2

본 명세서는 **마스터 가이드**, 원본은 **상세 참고자료**.

---

## 14. 일정 요약 (한 페이지)

```
Week 1 (Phase 1):
  Day 1: 인프라 셋업 (Onyx fork + Docmost 셋업)
  Day 2: 우리 RAG에 Onyx API 어댑터 추가
  Day 3: corpus → Docmost 마이그레이션 + webhook
  Day 4: Onyx Frontend 디자인 (white-label + Tailwind)
  Day 5: HyDE·Groundedness e2e 검증
  Day 6: Adversarial 250건 재평가
  Day 7: 결정 + 운영 준비

Week 2-3 (Phase 2, 조건부):
  Phase 2-A: Topic 단위 Auto-Merging Retriever (3일)
  Phase 2-B: Onyx UI Topic 트리 추가 (3일)
  Phase 2-C: 평가 + 운영 진입 (1일)

운영 진입 후:
  Week 4-: 베타 테스트 (학내 30명)
  Week 5+: 익명 피드백 누적
  6개월 후: Production Set 200건 → 진짜 성능 측정
```

---

**문서 작성일**: 2026-04-28
**적용 대상**: Onyx Frontend + Docmost + 우리 RAG 백엔드 통합
**기대 효과**: 1~3주에 운영 진입, 검증된 RAG 품질(recall@5 0.852) 보존, 학생용 챗봇 + 운영자 docs 분리 시스템