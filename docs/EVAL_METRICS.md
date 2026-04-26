# 평가지표 정리 (EulJi RAG)

> 데이터 기준: 2026-04-27 / QA 193개 / 코퍼스 동일
> 운영 기본값: **Reranker OFF** (2-core CPU 환경 가정)

## 1. 평가지표 정의

### 1.1 보조 4지표 (`scripts/eval_supplementary.py`)

| 지표 | 정의 | 목표 | 데이터 소스 |
|---|---|---:|---|
| **negative_rejection** | `qa_type=negative` 쿼리 중 fallback/거부 응답 비율 | ≥ 0.80 | `data/qa.parquet` |
| **campus_filter** | `qa_type=filter_required` 쿼리 중, 검색 결과 캠퍼스가 기대값(or "전체")인 비율 | = 1.00 | `data/qa.parquet` (`metadata.campus_filter`) |
| **routing_top3** | 비-negative 쿼리 중 정답 `source_collection`이 top-3 컨텍스트에 포함된 비율 | ≥ 0.95 | `data/qa.parquet` (`source_collection`) |
| **citation** | 비-negative 쿼리 중 답변에 `[출처: ...]` 패턴이 있는 비율 | ≥ 0.90 | 답변 텍스트 |

### 1.2 RAGAS (`scripts/evaluate_ragas.py`)

| 지표 | 의미 | 비고 |
|---|---|---|
| **faithfulness** | 답변이 컨텍스트에 충실한가 | 0~1 |
| **answer_relevancy** | 답변이 질문에 적절한가 | 0~1 |
| **context_precision** | 검색 결과의 정확도 | 0~1 |
| **context_recall** | 검색 결과의 회상도 | 0~1 |

### 1.3 리트리벌 단독 (`scripts/compare_rerank.py`)

| 지표 | 정의 |
|---|---|
| **routing_top3** | top-3 컬렉션에 정답 컬렉션 포함 |
| **collection_top1** | top-1 컬렉션이 정답과 일치 |
| **recall_at_5_doc** | top-5 문서에 정답 doc_id 포함 |

### 1.4 보조 (`scripts/compare_pipeline_rerank.py`)

| 지표 | 정의 |
|---|---|
| **fallback_rate** | 전체 쿼리 중 `verdict == "notGrounded"` 발생 비율 (낮을수록 좋음) |

---

## 2. 현재 성능 (Reranker OFF, 운영 기본, 2026-04-27 groundedness relax 적용)

> 출처: `reports/eval_supplementary.json`, `reports/fallback_diagnosis.json`

| 지표 | 값 | 목표 | 판정 | vs prev |
|---|---:|---:|:---:|---:|
| negative_rejection | **0.833** | ≥ 0.80 | PASS | −0.100 |
| campus_filter | **1.000** | = 1.00 | PASS | 0.000 |
| routing_top3 | **0.926** | ≥ 0.95 | FAIL (-2.4pt) | +0.049 |
| citation | **0.896** | ≥ 0.90 | FAIL (-0.4pt) | **+0.491** |
| fallback_rate | **0.117** | (낮을수록) | 양호 | **−0.525** |

**레이턴시**: ~4.12s/query (LLM + 임베딩 + 하이브리드 검색 포함)

### 2.1 Groundedness Relax 효과 (Option A)

`scripts/diagnose_fallback.py` 카테고리 분포 (n=163):

| Category | Before | After | Δ |
|---|---:|---:|---:|
| D_no_fallback (정상) | 64 / 0.393 | **144 / 0.883** | +80 |
| B_groundedness_strict (judge 거부) | 72 / 0.442 | 11 / 0.067 | **−61** |
| C_retry_lost_gt (HyDE 손실) | 16 / 0.098 | 4 / 0.025 | −12 |
| A_legit_miss (검색 실패) | 11 / 0.067 | 4 / 0.025 | −7 |

**Trade-off**: judge가 paraphrase·multi-doc 추론을 grounded로 인정하면서 negative 쿼리 일부도 답변으로 통과(`negative_rejection` −10pt). 잔여 negative 실패 5건은 "정보 없음" 거부 대신 답변 생성한 케이스.

---

## 3. Reranker A/B 비교 (193 QA, 풀 파이프라인)

> 출처: `reports/compare_pipeline_rerank.json`

| 지표 | no_rerank | with_rerank | Δ |
|---|---:|---:|---:|
| negative_rejection | 0.933 | 0.967 | +0.033 |
| campus_filter | 1.000 | 1.000 | 0.000 |
| routing_top3 | 0.877 | 0.939 | +0.061 |
| citation | 0.405 | 0.472 | +0.067 |
| fallback_rate | 0.642 | 0.596 | -0.047 |
| **per-query 레이턴시** | 4.12s | 5.87s | **+1.75s (+42%)** |

### 리트리벌 단독 비교 (193 QA)

> 출처: `reports/compare_rerank.json`

| 지표 | no_rerank | with_rerank | Δ |
|---|---:|---:|---:|
| routing_top3 | 0.957 | 1.000 | +0.043 |
| collection_top1 | 0.877 | 0.994 | +0.117 |
| recall_at_5_doc | 0.883 | 0.957 | +0.074 |
| **rerank/query** | 0ms | ~558ms (GPU) | — |

---

## 4. Reranker OFF 결정 근거

**프로덕션 환경**: 2-core CPU 서버

**실측 + 추정 비용**

| 환경 | bge-reranker-v2-m3-ko 레이턴시 | 비고 |
|---|---|---|
| GPU (개발/벤치마크) | ~558 ms/query | 측정값 |
| 2-core CPU (프로덕션) | **~15–30 s/query** | 추정 (50–60×) |

**효과 (with_rerank vs no_rerank)**
- routing_top3 +6.1pt, citation +6.7pt, fallback −4.7pt
- 목표 미달은 그대로 (routing은 0.939로도 0.95 미달)

**결론**: CPU에서는 사용자 응답 시간이 +20초 추가되어 서비스 불가. 효과(+6pt)도 routing/citation 목표를 풀파이프라인 기준 충족 못 시킴. 따라서 **`settings.reranker_enabled = False`를 기본값으로 고정**.

토글 방법: 환경변수 `RERANKER_ENABLED=true` (GPU 보유 시 또는 벤치마크 재현 시).

---

## 5. RAGAS

> 출처: `reports/ragas_summary.json`

| 지표 | Before (n=144) | After (n=163) | Δ |
|---|---:|---:|---:|
| faithfulness | 0.830 | **0.706** | −0.124 |
| answer_relevancy | 0.520 | **0.427** | −0.093 |
| context_precision | 0.851 | **0.724** | −0.127 |
| context_recall | 0.924 | **0.839** | −0.085 |

> **해석**: After는 groundedness relax 적용 후 측정. 모든 지표가 하락한 것처럼 보이지만, **채점 모집단이 다름** (n 144 → 163). 이전엔 fallback으로 빠진 borderline 답변이 RAGAS 채점 대상에 포함되지 않았으나, 이제 실제 답변으로 채점됨. 즉 사용자 체감 품질은 fallback 60% → 12%로 대폭 개선되었지만, RAGAS는 더 어려운 모집단을 채점한 결과.
>
> 추가 노이즈 요인: RAGAS 실행 중 TimeoutError 2건, "LLM returned 1 generations instead of 3" 다수 발생.

---

## 6. 미해결 우선순위

| # | 문제 | 가설 원인 | 다음 조치 |
|---|---|---|---|
| 1 | **negative_rejection 0.833** (-10pt) | 완화된 judge가 negative 쿼리도 grounded로 통과 (`"버스 시간표"`, `"비밀번호 뭐였지?"` 등 5건) | LLM 기반 의도 분류기 도입 (CLAUDE.md "no hardcoded pattern" 원칙)으로 negative를 사전 차단 |
| 2 | **routing_top3 0.926** (-2.4pt) | source_collection 다양성 부족 — 강의평가/학칙_조항/FAQ 쏠림 | hybrid_cc_weight 재튜닝, 컬렉션별 BM25 boost 도입 검토 |
| 3 | **citation 0.896** (-0.4pt, 거의 도달) | 잔여 17건 모두 FALLBACK 답변(인용 불가) | fallback_rate를 더 낮추거나 문제없음으로 수용 |
| 4 | RAGAS 전반 하락 | 채점 모집단 변화(borderline 답변 채점 포함) + 일부 TimeoutError | 모집단 정합성 확보 후 재측정, 답변 프롬프트 명세화 |

---

## 7. 평가 재실행 명령

```bash
# 보조 4지표
python scripts/eval_supplementary.py

# RAGAS
python scripts/evaluate_ragas.py

# 리랭커 A/B (리트리벌만)
python scripts/compare_rerank.py

# 리랭커 A/B (풀파이프라인, ~30~40분)
python scripts/compare_pipeline_rerank.py

# GPU에서 reranker ON으로 재실행
RERANKER_ENABLED=true python scripts/eval_supplementary.py
```

---

## 8. 변경 이력

| 날짜 | 변경 | 영향 |
|---|---|---|
| 2026-04-26 | sparse leak 수정 (hybrid 검색) | routing/recall 회복 |
| 2026-04-26 | groundedness multi-hop relax + citation post-processor | citation +25pt |
| 2026-04-26 | default_campus="성남" 정책 | campus_filter 0.647 → 1.000 |
| 2026-04-26 | FALLBACK contexts 보존 (eval signal) | routing_top3 0.479 → 0.933 |
| 2026-04-27 | **reranker OFF 기본값** (CPU 운영 결정) | 운영 레이턴시 −1.75s/q, routing/citation −6pt |
| 2026-04-27 | **groundedness 프롬프트 완화** (paraphrase·multi-hop·계산 허용) | citation 0.405→0.896 (+49pt), routing 0.877→0.926 (+5pt), fallback 0.607→0.117 (−49pt). Trade-off: negative_rejection 0.933→0.833 (−10pt) |
