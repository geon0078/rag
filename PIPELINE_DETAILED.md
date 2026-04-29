# EulJi RAG 파이프라인 상세 문서

> 작성일: 2026-04-28
> 코드 버전: V4_cc_w_high (sweep 베스트, hybrid_cc_weight=0.6)
> 본 문서는 **모든 프롬프트 / 변수 / 의사결정 규칙**을 단일 출처로 정리한 reference.

---

## 0. 한눈에 보기

```
사용자 query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ [Optional] QueryRewriter (default: OFF)                      │
│   Solar Pro 1회 호출 → {single|multi|vague|normal}            │
│   multi → 서브쿼리 N개 / vague → original + best-guess        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Router (configs/routing_rules.yaml)                          │
│   campus 추출 (성남/의정부/대전 키워드) → 없으면 default 성남  │
│   collection boost (장학금=2.0, 학칙=1.8, FAQ=1.5 등)         │
│   campus_was_inferred 플래그 → UI 알림용                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ HybridRetriever (병렬 dense+sparse → fusion)                  │
│   Dense:  Qdrant + solar-embedding-1-large-query (top 30)     │
│   Sparse: BM25 + KoNLPy Okt (top 30)                          │
│   Fusion: cc (mm-norm + w=0.6 dense weight)  OR  RRF (k=60)   │
│   Boost: collection_boost × score                             │
│   campus filter: {chunk.campus IN [질의캠퍼스, "전체"]}        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Reranker (default: Passthrough — bge cross-encoder 옵션)      │
│   Passthrough: 점수 정렬 그대로 → top 5                       │
│   bge-reranker-v2-m3-ko (GPU 필요): cross-encoder 재정렬      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Solar LLM 답변 생성 (solar-pro3, temp=0)                      │
│   format_context(candidates) → "[문서 1] doc_id=... 카테고리=..." │
│   USER_PROMPT_TEMPLATE 적용                                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ ensure_citation(answer, candidates)                          │
│   pure post-processor — LLM 호출 없음                         │
│   [출처: doc_id, 카테고리, 캠퍼스] 누락 시 top1 로부터 합성    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ GroundednessChecker.verify(context, answer)                  │
│   Solar Pro 심사관 → grounded / notSure / notGrounded         │
└─────────────────────────────────────────────────────────────┘
    │
    ├─ grounded → 답변 그대로 반환
    │
    ├─ notSure + relaxable=True → 통과 (multi-hop 등)
    │
    └─ notGrounded OR (notSure + non-relaxable):
         │
         ▼
       HyDE 확장 (hyde_enabled=True 인 경우):
         Solar Pro 1회 → 가상 답변 → query+가상답변 으로 재검색
         (top_k_dense=50, top_k_rerank_retry=10)
         │
         ▼
       2차 GroundednessChecker
         │
         ├─ grounded → 답변 반환 (retry=True 플래그)
         │
         └─ notGrounded → FALLBACK_ANSWER + 인용 부착
```

---

## 1. 사용 모델

### 1.1 Solar (Upstage)

| 역할 | 모델 ID | 차원 / 토큰 | 위치 |
|---|---|---|---|
| Passage 임베딩 | `solar-embedding-1-large-passage` | 4096-dim | corpus 인덱싱 시 |
| Query 임베딩 | `solar-embedding-1-large-query` | 4096-dim | retrieval 매번 |
| 답변 생성 | `solar-pro3` | temp=0, no max_tokens | RagPipeline.run |
| HyDE 확장 | `solar-pro3` | temp=0, max_tokens=200 | retry 시 |
| Groundedness 심사 | `solar-pro3` | temp=0, max_tokens=10 | 답변 후 검증 |
| Claim faithfulness (옵션) | `solar-pro3` | temp=0, max_tokens=600/80 | eval 시 |
| Query Rewriter (옵션) | `solar-pro3` | temp=0, max_tokens=300 | rewrite_enabled=True 시 |

### 1.2 Sparse 모델

| 컴포넌트 | 구성 |
|---|---|
| 토크나이저 | KoNLPy `Okt` (한국어 형태소) |
| BM25 인덱스 | `rank_bm25` (data/bm25_okt.pkl, schema_v2, 2382 docs) |
| 검색 함수 | OktBM25.search(query, top_k, metadata_filter) |

### 1.3 Reranker (옵션)

| 항목 | 값 |
|---|---|
| 모델 ID | `dragonkue/bge-reranker-v2-m3-ko` |
| 종류 | Cross-encoder (Sigmoid 활성) |
| 디바이스 자동 선택 | cuda → mps → cpu 순 |
| 활성 조건 | `settings.reranker_enabled=True` (env: `RERANKER_ENABLED=true`) |
| 비활성 시 | `PassthroughReranker` (점수 정렬 그대로 슬라이스) |
| 응답 시간 | GPU ~1.7s/q · CPU ~15-30s/q (2-core) |

---

## 2. 모든 변수 (settings, default)

`src/config.py` 의 `Settings` BaseModel.

```python
# Solar API
upstage_api_key       = os.getenv("UPSTAGE_API_KEY", "")
upstage_base_url      = "https://api.upstage.ai/v1/solar"
embedding_model_passage = "solar-embedding-1-large-passage"
embedding_model_query   = "solar-embedding-1-large-query"
embedding_dim         = 4096
llm_model_pro         = "solar-pro3"
llm_model_mini        = "solar-mini"
llm_temperature       = 0.0
llm_timeout_sec       = 60.0     # 단일 Solar 호출 타임아웃 (60s ↑ 시 worst-case 24분 hang 사례 있음)

# Qdrant
qdrant_url            = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key        = os.getenv("QDRANT_API_KEY", "")
qdrant_collection     = "euljiu_knowledge"

# Redis (Celery broker / 캐시)
redis_url             = os.getenv("REDIS_URL", "redis://localhost:6379")

# Hybrid retrieval
top_k_dense           = 30        # Qdrant top-K
top_k_sparse          = 30        # BM25 top-K
top_k_rerank_final    = 5         # 최종 reranker 출력
top_k_rerank_retry    = 10        # HyDE retry 시 reranker 출력 (확장)
hybrid_method         = "cc"      # "cc" (default) | "rrf"
hybrid_cc_weight      = 0.6       # ★ V4 베스트 (이전 0.4) — semantic 비중
hybrid_cc_normalize   = "mm"      # mm | tmm | z | dbsf
rrf_k                 = 60        # method="rrf" 시 (k+rank) 분모

# Reranker
reranker_model        = "dragonkue/bge-reranker-v2-m3-ko"
reranker_enabled      = bool(os.getenv("RERANKER_ENABLED", "false")=="true")

# BM25
bm25_tokenizer        = "okt"

# Routing
default_campus        = "성남"     # 쿼리에 캠퍼스 신호 없으면 fallback

# Embedding
embed_batch_size      = 100
embed_retry_max       = 5

# API
api_host              = "0.0.0.0"
api_port              = 8000
log_level             = os.getenv("LOG_LEVEL", "INFO")
```

### 2.1 RagPipeline 인자

```python
RagPipeline(
    retriever        = HybridRetriever() | None,
    reranker         = KoReranker | PassthroughReranker | None,
    llm              = SolarLLM() | None,
    groundedness     = GroundednessChecker() | None,
    hyde_enabled     = True,        # ★ A/B 테스트 결과 유지 권장
    rewriter         = QueryRewriter() | None,
    rewrite_enabled  = False,       # ★ 단독 사용 시 회귀 발견 → off 기본값
)
```

---

## 3. 모든 프롬프트

### 3.1 SYSTEM_PROMPT (답변 생성)
`src/generation/prompts.py:8-22`

```
당신은 을지대학교 학사 정보 안내 전문가입니다.
아래 [참고 문서]만을 근거로 사용자 질문에 정확하게 답하세요.

규칙:
1. 반드시 참고 문서에 있는 정보만 사용하세요. 추측하지 마세요.
2. **[질문 핵심에 집중]** 답변의 첫 문장은 반드시 질문이 묻는 핵심 정보를
   직접 응답하세요. 배경 설명은 그 뒤에 추가하세요.
3. **[정답 우선 인용]** [참고 문서]는 검색 점수 순으로 정렬되어 있습니다.
   **상위 문서(문서 1, 2)에 답이 있다면 그 정보를 우선 인용하세요.**
   하위 문서는 보충용으로만 사용하세요. 여러 문서가 충돌하면 상위 문서를 따르세요.
4. **[중요]** 답변 마지막 줄에 반드시 출처를 표기하세요. 형식은 정확히 다음과 같아야 합니다:
   `[출처: doc_id, 카테고리, 캠퍼스]`
   - doc_id, 카테고리, 캠퍼스 값은 [참고 문서] 메타데이터에서 그대로 가져오세요.
   - 여러 문서를 인용해야 할 경우 같은 형식을 줄을 바꿔 추가하세요.
   - 이 출처 줄을 절대 생략하거나 다른 형식(예: "출처:", "[ref:", "Sources:")으로 바꾸지 마세요.
5. 정보가 부족하거나 확실하지 않으면 "제공된 자료에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
6. 학칙·졸업요건 같은 공식 정보는 원문 표현을 가능한 유지하세요.
7. 강의평가는 학생 의견이며 객관적 사실이 아님을 명시하세요.
```

### 3.2 USER_PROMPT_TEMPLATE
`src/generation/prompts.py:25-31`

```
[참고 문서]
{retrieved_contents}

[질문]
{query}

[답변]
```

`retrieved_contents` 는 `format_context()` 가 빌드:

```
[문서 1] doc_id=... | 카테고리=... | 하위=... | 캠퍼스=... | 출처=...
{청크 본문}

[문서 2] doc_id=... | 카테고리=... | ...
{청크 본문}
```

### 3.3 INFERRED_CAMPUS_NOTICE (campus inferred 시)
`src/generation/prompts.py:37`

```
[{campus}캠퍼스 기준 답변입니다]
```

답변 앞에 prepend. router 가 explicit campus 신호 못 찾고 default 성남 사용한 경우.

### 3.4 HYDE_PROMPT_TEMPLATE
`src/generation/prompts.py:40-46`

```
다음 질문에 답하는 가상의 짧은 문서를 한국어로 작성하세요.
사실 여부보다는 검색에 도움이 되는 키워드와 문맥을 풍부하게 포함하세요.
2~4문장으로 작성하고 출처는 표기하지 마세요.

질문: {query}

가상 문서:
```

→ 결과를 `query + "\n\n" + hyde_doc` 으로 결합 후 retrieve 재시도.

### 3.5 Groundedness Judge
`src/generation/groundedness.py:24-39`

**System:**
```
당신은 RAG 응답의 사실 일치성을 평가하는 심사관입니다.
주어진 [참고 문서]를 근거로, [답변]의 핵심 주장(main claims)이 문서에 의해 뒷받침되는지 판정하세요.

판정 원칙 (관대한 기준 — 검색이 정답을 가져왔다면 답변은 대부분 grounded):
- 답변의 핵심 주장이 문서에 직접 등장하거나 paraphrase·요약 형태로 존재하면 grounded
- **답변이 다루는 주제가 문서와 같고, 답변 내용이 문서로부터 합리적으로 도출 가능하면 grounded** (정확히 일치하지 않아도 됨)
- 여러 문서를 종합한 합리적 추론, 명시된 정보로부터의 직접적 계산, 그리고 단순 적용·재해석은 모두 grounded
- 표현 차이, 단어 선택, 어순, 문장 분할 차이는 모두 무시
- "정보 없음" 등 거부 답변도 문서와 모순되지 않으면 grounded
- **notGrounded는 오직 답변이 문서와 명백히 모순되거나 문서에 없는 새로운 구체적 사실(고유명사/숫자/날짜)을 발명한 경우에만**
- notSure는 답변이 핵심에서 크게 벗어나거나 모호한 경우에만 보수적으로 사용

응답 형식: 반드시 다음 셋 중 하나의 단어만 출력하세요. 다른 설명이나 부가 텍스트를 포함하지 마세요.
grounded
notGrounded
notSure
```

**User:**
```
[참고 문서]
{context}

[답변]
{answer}

[판정]
```

`max_tokens=10` 으로 강제. 한국어 변형 ("근거 없음" 등) 도 normalize 함수에서 매핑.

### 3.6 Claim Faithfulness — Extract
`src/eval/claim_faithfulness.py:58-67`

```
다음 한국어 답변을 atomic 사실 단위(claim)로 쪼개세요.

규칙:
1. 각 claim 은 한 문장의 짧은 사실 진술 (예: "졸업학점은 130학점이다").
2. 한 claim 안에 여러 사실이 들어가지 않게 (and 로 묶지 말 것).
3. 출처 표기 (`[출처: ...]`), 인사말, 메타 코멘트는 제외.
4. 빈 답변·거부 답변("정보를 찾을 수 없습니다" 등)은 빈 list 반환.

반드시 다음 JSON 한 줄만 출력. 다른 텍스트 금지:
{"claims": ["claim 1", "claim 2", ...]}
```

### 3.7 Claim Faithfulness — Verify
`src/eval/claim_faithfulness.py:70-78`

```
주어진 [참고 문서]에 비추어 [주장] 이 사실로 뒷받침되는지 판정하세요.

판정 기준:
- "supported"   = 주장이 문서에 직접 등장하거나 paraphrase·요약으로 존재
- "partial"     = 일부만 뒷받침되거나, 주장 일부에 표현 차이가 있어 불확실
- "not_supported" = 문서와 모순되거나 문서에 전혀 없는 새로운 정보

반드시 다음 JSON 한 줄만 출력. 다른 텍스트 금지:
{"verdict": "supported|partial|not_supported", "rationale": "<한 문장>"}
```

### 3.8 QueryRewriter (옵션, default off)
`src/pipeline/query_rewriter.py:43-72`

**System:**
```
당신은 한국어 대학 학사 챗봇의 쿼리 분석기입니다.
사용자 입력 쿼리를 분석해 검색에 더 유리한 형태로 재작성합니다.
```

**User template:**
```
다음 쿼리의 의도를 분류하고 검색 가능한 형태로 재작성하세요.

분류 기준:
- "single": 명확한 단일 의도. 그대로 검색 가능.
- "multi": 두 개 이상의 독립적인 질문이 결합됨 (예: "X는 언제이고 Y는 어디?"). 분해 필요.
- "vague": 지시어("이거", "그거"), 짧은 구어체, 의도 불명. 명료화 필요.
- "normal": single 과 같음. 단어 길이가 정상이고 모호하지 않음.

규칙:
1. multi 인 경우: 독립적인 서브쿼리 2-3개로 분해. 각 서브쿼리는 그 자체로 검색 가능.
2. vague 인 경우: 가장 가능성 높은 의도로 best-guess 재작성 1개.
3. single/normal 인 경우: rewrites = [원본] 그대로.

JSON 으로만 출력 (다른 설명 금지):
{"type": "single|multi|vague|normal", "rewrites": ["...", "..."]}

쿼리: {query}
```

---

## 4. Routing 규칙

### 4.1 Campus 추출
`src/retrieval/router.py:45-57` + `configs/routing_rules.yaml:4-7`

```yaml
campus_keywords:
  성남:    ["성남", "성남캠퍼스"]
  의정부:  ["의정부", "의정부캠퍼스"]
  대전:    ["대전", "대전캠퍼스"]
```

쿼리에 명시적 키워드 없으면 `settings.default_campus="성남"` 사용 + `campus_was_inferred=True`.

### 4.2 Collection Boosts
`configs/routing_rules.yaml:9-36`

| 키워드 | Boost 컬렉션 | weight |
|---|---|---|
| 장학금, 장학 | 장학금 | 2.0 |
| 전화번호, 연락처, 사무실, 팩스 | 시설_연락처 | 2.0 |
| 수강신청, 개강, 휴학, 복학, 졸업식, 학위수여식, 방학, 공휴일 | 학사일정 | 2.0 |
| 학칙, 조항 | 학칙_조항 | 1.8 |
| FAQ, 자주, 질문 | FAQ | 1.5 |
| 강의평가, 수업후기, 교수님, 시험 | 강의평가 | 1.5 |
| 커리큘럼, 교육과정, 로드맵 | 교육과정 | 1.5 |
| 졸업요건, 학점, 이수 | 학사정보 | 1.5 |
| 학과, 전공, 단과대학 | 학과정보 | 1.3 |

→ Hybrid `_fuse()` 에서 `score = score × boost` 로 적용.

### 4.3 Per-collection BM25/Dense 가중치 (옵션)
`configs/routing_rules.yaml:39-49`

```yaml
collection_weights:
  학칙_조항:    {bm25: 0.5, dense: 0.5}
  FAQ:          {bm25: 0.6, dense: 0.4}
  시설_연락처:  {bm25: 0.7, dense: 0.3}
  장학금:       {bm25: 0.4, dense: 0.6}
  학사정보:     {bm25: 0.4, dense: 0.6}
  학사일정:     {bm25: 0.5, dense: 0.5}
  강의평가:     {bm25: 0.3, dense: 0.7}
  학과정보:     {bm25: 0.5, dense: 0.5}
  교육과정:     {bm25: 0.5, dense: 0.5}
  기타:         {bm25: 0.5, dense: 0.5}
```

> 현재 코드 path 는 `bm25_dense_weights` 를 router 가 노출만 하고, `_fuse()` 는 글로벌 `hybrid_cc_weight` 만 사용. 컬렉션별 가중치 적용은 향후 `compare_pipeline_rerank.py` 등 실험에서 활성.

---

## 5. Hybrid Fusion 알고리즘 (cc)

`src/retrieval/hybrid.py:99-110`

```python
# 1. raw 점수 수집
dense_raw[doc_id]  = float(qdrant_score)
sparse_raw[doc_id] = float(bm25_score)

# 2. min-max 정규화 (mm-norm)
dense_norm[id]  = (dense_raw[id]  - min(dense_raw))  / (max - min)
sparse_norm[id] = (sparse_raw[id] - min(sparse_raw)) / (max - min)

# 3. weighted sum (w = settings.hybrid_cc_weight = 0.6)
score[id] = w * dense_norm[id] + (1 - w) * sparse_norm[id]

# 4. 컬렉션 boost
if collection in decision.boosts:
    score[id] *= decision.boosts[collection]   # 예: 2.0

# 5. top-K
ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k_final]
```

### 5.1 RRF 변형 (settings.hybrid_method="rrf")

```python
score[id] = sum(1 / (rrf_k + rank_i)) for each retriever
            # rrf_k = 60
```

---

## 6. Citation 보강 (post-processor)

`src/generation/citation.py`

```python
CITATION_PATTERN = re.compile(r"\[출처\s*[::]")

def ensure_citation(answer, candidates):
    if has_citation(answer):
        return answer                        # LLM 이 이미 추가
    if not candidates:
        return answer                        # 검색 결과 없음
    top = candidates[0]
    payload = top.get("payload") or {}
    citation = f"[출처: {top.doc_id}, {payload.category}, {payload.campus}]"
    return f"{answer}\n\n{citation}"
```

순수 함수 — LLM 호출 없음. 매 답변에 적용 (`pipeline.run` 라인 156, 195).

---

## 7. 재시도 정책 (HyDE retry)

`src/pipeline/rag_pipeline.py:170-216`

### 7.1 Relaxable 쿼리 감지
`_is_relaxable(query)` — 다음 패턴 매칭 시 `notSure` 도 통과:
- multi-hop (조항·과목·학과 결합)
- date-arithmetic ("2026년 3월 1일에 X는 Y인가요?")
- 그 외 보수적

(상세 정규식은 `src/pipeline/rag_pipeline.py:1-100`)

### 7.2 Retry 흐름

```
1차 retrieve → answer → groundedness:
  grounded                                → 통과
  notSure  AND relaxable                  → 통과 (multi-hop 등)
  notSure  AND NOT relaxable              → retry
  notGrounded                             → retry

retry (hyde_enabled=True):
  hyde_doc = await llm.hyde_expand(query, max_tokens=200)
  expanded_query = f"{query}\n\n{hyde_doc}"
  candidates = retrieve(expanded_query, top_k=top_k_dense+20, rerank=top_k_rerank_retry=10)
  answer = llm.generate(query, candidates)
  groundedness:
    grounded     → 통과 (retry=True 플래그)
    notGrounded  → fallback

retry (hyde_enabled=False):
  expanded_query = query              # HyDE 건너뜀, 원본 query 로 재검색만
  ...

fallback:
  FALLBACK_ANSWER = "제공된 자료에서 해당 정보를 찾을 수 없습니다."
  ensure_citation(FALLBACK_ANSWER, candidates)   # 검색된 contexts 인용 보존
  return {grounded: False, retry: True, ...}
```

### 7.3 Routing decision 재사용

HyDE 확장 텍스트가 캠퍼스 신호를 희석할 수 있어 **원본 query 의 RoutingDecision 을 그대로 재사용**. router 다시 호출 안 함.

---

## 8. Embedding 호출 디테일

`src/embeddings/solar_embedder.py`

| 항목 | 값 |
|---|---|
| Endpoint | `{upstage_base_url}/embeddings` |
| 모델 (passage / query) | `solar-embedding-1-large-passage` / `solar-embedding-1-large-query` |
| 차원 | 4096 |
| Batch size | `embed_batch_size=100` |
| Retry | `embed_retry_max=5` (exponential backoff base 2.0s) |

Query 임베딩은 retrieval 매번 호출. Passage 임베딩은 인덱싱 시 한 번만.

---

## 9. Qdrant 검색 디테일

`src/retrieval/qdrant_store.py`

| 항목 | 값 |
|---|---|
| Collection | `euljiu_knowledge` |
| Distance | Cosine |
| Vector size | 4096 |
| 검색 메서드 | `client.search(vec, top_k, query_filter)` |
| Filter | campus IN [질의캠퍼스, "전체"] (router 결정) |

---

## 10. BM25 (KoNLPy Okt) 디테일

`src/retrieval/bm25_okt.py`

| 항목 | 값 |
|---|---|
| 인덱스 파일 | `data/bm25_okt.pkl` (schema_v2) |
| 토크나이저 | KoNLPy Okt — `okt.morphs(text)` |
| 인덱스 크기 | 2,382 docs |
| Filter | metadata 기반 dict (campus 등) |
| 점수 정규화 | min-max in `_fuse()` |

JVM 필요 (Dockerfile 에 `default-jre-headless` 설치).

---

## 11. 평가 메트릭 정의

`src/eval/retrieval_metrics.py`

| 메트릭 | 정의 | 통과 기준 (평가명세서 §8.1) |
|---|---|---|
| recall@5 | top-5 안에 expected_doc_id 가 1개 이상 들어왔는지 (binary) → 평균 | ≥ 0.85 |
| recall@10 | top-10 동일 | ≥ 0.95 |
| MRR | 첫 정답의 역순위 평균 | ≥ 0.65 |
| nDCG@5 | DCG@5 / IDCG@5 | ≥ 0.75 |
| citation | 답변에 `[출처: ...]` 패턴 있는지 (binary) → 평균 | ≥ 0.90 |
| grounded rate | groundedness verdict ∈ {grounded, relaxable+notSure} 비율 | (참고) |
| HyDE retry rate | retry=True 인 쿼리 비율 | (참고) |
| claim faithfulness | (옵션) supported / total 비율, partial=0.5 가중 | ≥ 0.85 |

---

## 12. 현재 default 환경 (V4_cc_w_high)

```python
# src/config.py 권장 default (2026-04-28 sweep 결과 기반)
top_k_dense           = 30
top_k_sparse          = 30
top_k_rerank_final    = 5
top_k_rerank_retry    = 10
hybrid_method         = "cc"
hybrid_cc_weight      = 0.6     # ★ 이 값이 베스트
hybrid_cc_normalize   = "mm"
reranker_enabled      = False    # GPU 환경 시 True 검토
default_campus        = "성남"
hyde_enabled          = True     # off 시 -3.3pt grounded
rewrite_enabled       = False    # 단독 사용 시 회귀
```

**측정 결과 (수작업 250)**:
- recall@5 0.852 ✅
- MRR 0.678 ✅
- citation 1.000 ✅
- nDCG@5 0.716 ❌ (-0.034)
- recall@10 0.868 ❌ (-0.082, 시스템 한계)
- grounded 0.960 (참고)

---

## 13. 파일 위치 빠른 참조

| 파일 | 역할 |
|---|---|
| `src/config.py` | 모든 settings 변수 |
| `src/pipeline/rag_pipeline.py` | RagPipeline.run() — 전체 orchestration |
| `src/pipeline/query_rewriter.py` | (옵션) intent 분류기 |
| `src/embeddings/solar_embedder.py` | Solar embedding 호출 |
| `src/retrieval/hybrid.py` | dense+sparse fusion |
| `src/retrieval/qdrant_store.py` | Qdrant 클라이언트 |
| `src/retrieval/bm25_okt.py` | BM25 + Okt 토크나이저 |
| `src/retrieval/router.py` | campus / boost 결정 |
| `src/retrieval/reranker.py` | bge-reranker / Passthrough |
| `src/generation/prompts.py` | SYSTEM / USER / HyDE 프롬프트 |
| `src/generation/solar_llm.py` | LLM async wrapper + retry |
| `src/generation/groundedness.py` | Solar Pro 심사관 |
| `src/generation/citation.py` | 인용 후처리 (pure 함수) |
| `src/eval/claim_faithfulness.py` | (옵션) RAGChecker 패턴 |
| `src/eval/retrieval_metrics.py` | recall/MRR/nDCG 계산 |
| `configs/routing_rules.yaml` | 캠퍼스 키워드 + 컬렉션 boost |

---

## 14. 평가 / 측정 도구

| 스크립트 | 용도 |
|---|---|
| `scripts/eval_adversarial.py` | 수작업 250 default 측정 |
| `scripts/eval_golden.py` | Golden Set 측정 |
| `scripts/pipeline_sweep.py` | N variants 비교 (cc_w / reranker / rewriter / hybrid_method 등) |
| `scripts/ab_test_hyde.py` | HyDE on/off paired bootstrap |
| `scripts/generate_adversarial_qa.py` | Adversarial QA 자동 생성 |
| `scripts/gradio_app.py` | 운영자 데모 UI (현재 docker `gradio` 서비스) |
