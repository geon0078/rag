# 을지대 RAG 시스템 — Claude Code 컨텍스트

> 이 파일은 Claude Code 세션이 자동으로 읽음. 프로젝트의 핵심 구조 + 운영 명령 +
> 컨벤션을 한눈에 제공.

## 프로젝트

을지대학교 학사 정보 RAG 챗봇. 한국어 학사 코퍼스 (학칙·강의평가·시설·장학금·FAQ 등)
검색 + Solar Pro 답변 생성. 운영자가 Outline 에서 콘텐츠 편집, V4 RAG 가 검색,
Onyx UI 가 운영자/개발자 도구.

## 아키텍처

```
┌─ 운영자 ────────────────────┐  ┌─ 학생 ──────────────────────┐
│ http://localhost:3002        │  │ http://localhost:5173/chat   │
│ Outline (운영 wiki)         │  │ React frontend (chat UI)     │
│ - 페이지 작성/편집           │  │ - 채팅 + OCR 업로드          │
└──────────────┬───────────────┘  └──────────────┬───────────────┘
               │ index_outline.py                │
               ↓                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  V4 RAG (euljiu-api, port 8000)                                  │
│  - Solar 4096-dim embedding + KoNLPy Okt BM25 + cc fusion        │
│  - /v1/chat/completions (OpenAI compat) ★                        │
│  - /api/ocr (Solar Document Parse)                                │
│  - Qdrant (euljiu_outline) + bm25_outline.pkl                     │
│  - 답변에 breadcrumb 인용 + outline URL 자동                     │
└──────────────────────────────────────────────────────────────────┘
               ↑ LLM provider
               │
┌──────────────────────────────────────────────────────────────────┐
│  Onyx (개발/admin, port 3010)                                     │
│  - V4 를 LLM provider 로 등록 (api_base = host.docker.internal:8000/v1) │
│  - Persona "을지대 학사 도우미" (도구 없음 — V4 가 RAG)          │
└──────────────────────────────────────────────────────────────────┘
```

## 디렉토리

```
AI-ver-4/
├── src/                          # V4 RAG 핵심 (pipeline, retrieval, generation)
│   ├── pipeline/rag_pipeline.py  # RagPipeline.run() — 전체 orchestration
│   ├── retrieval/                # qdrant_store, bm25_okt, hybrid, router, reranker
│   ├── generation/               # solar_llm, prompts, citation, groundedness
│   └── config.py                 # Settings (qdrant_collection, bm25 path 등)
├── backend/                      # FastAPI admin/V4 entrypoint (port 8000)
│   ├── main.py                   # app + router 등록
│   └── routers/
│       ├── openai_compat.py     # /v1/chat/completions (Onyx 가 호출)
│       ├── ocr.py                # /api/ocr (Solar Document Parse 프록시)
│       ├── onyx.py               # legacy /api/onyx/* (이전 어댑터)
│       └── ...                   # chunks, tree, preview, indexing, sync
├── frontend/                     # React + TypeScript + Vite (port 5173)
│   └── src/pages/ChatPage.tsx    # 학생 챗봇 UI (/chat)
├── outline/                      # 분리된 Outline docker stack
│   ├── docker-compose.yml        # outline + postgres + redis + mailpit
│   ├── .env                      # OUTLINE_SECRET_KEY 등
│   └── README.md                 # 운영 가이드
├── data/
│   ├── corpus.parquet            # 옛 corpus (마이그레이션 source)
│   ├── outline_chunks.parquet    # Outline 기반 청크 (현재 사용)
│   ├── bm25_outline.pkl          # Okt BM25 인덱스
│   └── outline_url_map.json      # corpus_doc_id → outline_url + by_outline_doc_uuid
├── scripts/
│   ├── index_outline.py          # Outline → Qdrant + BM25 (재인덱싱)
│   ├── audit_outline.py          # 데이터 audit (중복, Q&A 패턴)
│   ├── restructure_outline.py    # 노이즈 제거 + Q&A → 사실 변환
│   ├── restructure_subpages.py   # 큰 페이지 split + 학칙 chapter grouping
│   ├── outline_migrate.py        # 초기 Outline import (1회용, 완료)
│   ├── build_outline_url_map.py  # URL 매핑 빌드
│   ├── eval_outline_250.py       # Adversarial 250 평가
│   └── ...
├── docs/
│   └── DATA_SCHEMA.md            # Outline 페이지 schema (8 type 정의)
├── docker-compose.yml            # 메인 V4 스택 (postgres, qdrant, redis, api, worker, web, gradio)
├── docker-compose.onyx.yml       # Onyx 풀 스택 (--project-name onyx)
├── PIPELINE_DETAILED.md          # V4 RAG 파이프라인 상세
└── 운영웹통합명세서.md            # 운영자 web admin 명세
```

## 주요 포트

| 포트 | 서비스 | 비고 |
|---:|---|---|
| 3002 | Outline UI | 운영자 wiki (`outline/` 컨테이너) |
| 3010 | Onyx UI | 개발/admin 도구 |
| 5173 | React frontend | 학생 챗봇 (`/chat`) |
| 7861 | Gradio demo | RAG 데모 (개발용) |
| 8000 | V4 backend | `/v1/chat/completions`, `/api/ocr`, `/api/onyx/*` |
| 8025 | Mailpit UI | Outline 매직 링크 캐쳐 |
| 6333 | Qdrant | 벡터 DB |
| 5432 | PostgreSQL | V4 메타데이터 |

## 자주 쓰는 명령

### 인덱스 재구축 (Outline 변경 후)
```bash
python scripts/index_outline.py
docker exec euljiu-redis redis-cli FLUSHDB  # 답변 캐시 비우기
docker restart euljiu-api                    # BM25 새로 로드
```

### V4 + Onyx 빠른 테스트
```bash
# V4 직접
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"solar-pro","messages":[{"role":"user","content":"졸업학점"}],"stream":false}'

# Onyx 통과
curl -X POST http://localhost:3010/api/chat/send-chat-message \
  -H 'Authorization: Bearer <ONYX_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"chat_session_info":{},"message":"졸업학점","stream":false}'
```

### Outline 스택 시작/중단
```bash
# 시작
docker compose -f outline/docker-compose.yml --env-file outline/.env --project-name outline up -d

# 중단 (volumes 보존)
docker compose -f outline/docker-compose.yml --project-name outline stop
```

### Adversarial 250 평가
```bash
python scripts/eval_outline_250.py
# 결과: reports/eval_outline_250.{json,md}
```

### Outline 매직 링크 받기 (mailpit 통해)
```bash
# 1. UI 에서 이메일 입력 → 매직 링크 발송
# 2. mailpit (http://localhost:8025) 에서 링크 확인 후 브라우저에 붙여넣기
```

## 핵심 컨벤션

### 1. 하드코딩 패턴 매칭 금지 (CLAUDE 글로벌 규칙)
```python
# ❌ 금지
if "맛집" in query:
    query = query.replace("맛집", "식당")

# ✅ 올바름 — LLM 활용
prompt = "동의어 추가하여 검색 쿼리 확장: {query}"
```

### 2. Outline 페이지 작성 규칙 (`docs/DATA_SCHEMA.md`)
- Frontmatter 필수 (type, campus, ...)
- `## 섹션` 단위 분할 (chunk 와 일치)
- 사실 데이터만 (Q&A·예시·"예를 들면" narrative 금지)
- 한 페이지 = 한 주제 (cross-collection 분산 금지)
- 큰 페이지는 parent-child 트리로 분할 (`restructure_subpages.py`)

### 3. RAG 파이프라인 (V4 cc_w_high — recall@5 0.866)
- Solar 4096-dim embedding (`embedding-passage`/`-query`)
- KoNLPy Okt BM25
- Hybrid cc fusion (mm-norm, semantic_weight=0.6)
- Reranker: PassthroughReranker (default), bge-reranker-v2-m3-ko (옵션)
- HyDE retry (groundedness fail 시)

### 4. 청크 메타데이터 (Outline 기반)
```python
{
  "doc_id": "outline_<uuid>_c<idx>",
  "parent_doc_id": "outline_<uuid>",
  "metadata": {
    "outline_doc_id": "<uuid>",
    "outline_url": "http://localhost:3002/doc/...",
    "outline_parent_id": "<chapter_uuid>",  # 트리 부모
    "collection": "학칙",
    "title": "제20조(수강신청)",
    "breadcrumb": "학칙 > 제5장 수강신청 및 수업 > 제20조(수강신청)",  # ★
    "parent_titles": ["학칙", "제5장 수강신청 및 수업"],
    "chunk_section": "1항",
    "campus": "전체",
    "type": "regulation"
  }
}
```

### 5. 인용 형식 (breadcrumb)
LLM 답변 끝:
```
[출처: 학칙 > 제5장 수강신청 및 수업 > 제20조(수강신청)](http://localhost:3002/doc/...)
```
- breadcrumb 표시 → 학생이 클릭 없이 위치 인지
- markdown 링크 자동 (post-processor)

## 평가 지표 (검증된 baseline)

| 지표 | V4 (legacy corpus) | V4 + Outline | Onyx native |
|---|---:|---:|---:|
| recall@5 | 0.852 | **0.866** | 0.454 |
| recall@10 | 0.868 | 0.866 | 0.518 |
| MRR | 0.678 | **0.707** | 0.368 |
| nDCG@5 | 0.716 | **0.743** | 0.380 |

## 알려진 제약 / TODO

- **실시간 sync 없음**: Outline 변경 → 수동 `index_outline.py` 필요. webhook 미구현.
- **chapter mapping 부분 누락**: 학칙 일부 조항은 chapter parent 매칭 안됨.
- **부모 단위 필터링 X**: chapter 안에서만 검색 같은 facet 미구현.
- **Onyx Outline native connector 미사용**: Onyx 자동 폴링 가능하나 한국어 검색 품질 떨어져서 V4 만 사용.

## 추가 운영 노트

- V4 RAG 캐시: Redis 24시간 (`cache_ttl_default_sec`). 같은 query 답변 캐시.
- Onyx LLM provider: `host.docker.internal:8000/v1` (Windows Docker), api_key bytea encrypted.
- Outline auth: 매직 링크 (mailpit 캐쳐). 실제 SMTP 운영 시 Gmail/SendGrid 등으로 전환.
- Solar Pro 2/3: tool_calls + streaming 호환성 버그 → solar-pro (older) 사용 권장.
- 청크 본문 prepend: `[breadcrumb]\n## 섹션\n본문` 형식 (semantic 매칭 향상).
