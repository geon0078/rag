# RAG 시스템 평가 리포트

평가 일시: 2026-04-26T18:23:28

## 1. AutoRAG 자동 평가

| node_line | node_type | best_module | filename | params |
|---|---|---|---|---|
| retrieve_node_line | lexical_retrieval | BM25 | 0.parquet | {'top_k': 10, 'bm25_tokenizer': 'ko_okt'} |
| retrieve_node_line | semantic_retrieval | VectorDB | 0.parquet | {'top_k': 10, 'vectordb': 'bge_m3_chroma'} |
| retrieve_node_line | hybrid_retrieval | HybridCC | 2.parquet | {'top_k': 10, 'normalize_method': 'mm', 'weight_range': (0.0, 1.0), 'test_weight_size': 21, 'weight': 0.15000000000000002} |
| retrieve_node_line | passage_reranker | FlagEmbeddingReranker | 1.parquet | {'top_k': 5, 'batch': 16, 'model_name': 'dragonkue/bge-reranker-v2-m3-ko'} |
| post_retrieve_node_line | prompt_maker | Fstring | 0.parquet | {'prompt': '다음 [참고 문서]만을 근거로 [질문]에 한국어로 정확하게 답하세요.\n문서에 없는 정보는 추측하지 말고 "정보가 부족합니다"라고 답하세요.\n\n[참고 문서]\n{retrieved_contents}\n\n[질문]\n{query}\n\n[답변]\n'} |
| post_retrieve_node_line | generator | LlamaIndexLLM | 0.parquet | {'llm': 'openailike', 'model': 'solar-pro2', 'api_base': 'https://api.upstage.ai/v1/solar', 'api_key': '***REDACTED***', 'temperature': 0.0, 'max_tokens': 800, 'batch': 4} |

## 2. RAGAS 평가

| 메트릭 | 점수 |
|---|---|
| faithfulness | 0.830 |
| answer_relevancy | 0.520 |
| context_precision | 0.851 |
| context_recall | 0.924 |
| n (샘플 수) | 144 |

## 3. 추가 검증 4종

| 메트릭 | 실제 | 목표 | 통과 |
|---|---|---|---|
| Negative 거절률 | 0.867 | 0.80 | ✅ |
| 캠퍼스 필터 정확도 | 0.647 | 1.00 | ❌ |
| 라우팅 top-3 정확도 | 0.883 | 0.95 | ❌ |
| 출처 인용 형식 | 0.877 | 0.90 | ❌ |

## 4. 통과 기준 검증

| 지표 | 실제 | 목표 | 통과 |
|---|---|---|---|
| RAGAS faithfulness | 0.830 | ≥ 0.85 | ❌ |
| Negative 거절률 | 0.867 | ≥ 0.8 | ✅ |
| 캠퍼스 필터 정확도 | 0.647 | ≥ 1.0 | ❌ |
| 카테고리 라우팅 정확도 | 0.883 | ≥ 0.95 | ❌ |

**전체 결과: ❌ FAIL — 미달 항목 확인 필요**

## 5. 미달 항목 진단

### 5.1 19건 fallback 폭포 (3개 메트릭 동시 실패)

`reports/eval_routing.json`, `eval_citation.json`, `eval_campus_filter.json`의 실패 qid 교집합 분석 결과:

| 실패 패턴 | qid 수 | 메트릭 영향 |
|---|---|---|
| `pipeline.run`이 fallback 답변 반환 (`got_top3=[]`, answer = "제공된 자료에서 해당 정보를 찾을 수 없습니다") | 19 | routing 19/19 fail · citation 19/19 fail · campus 3/19 fail |
| 모든 19건이 `qa.parquet`에 `retrieval_gt` 1건 보유 (=정답이 코퍼스에 존재) | 19 | groundedness 거짓 음성 추정 |

19건 분포: multi_hop=15, filter_required=3 (성남캠퍼스 학사일정 날짜 계산), single_hop=2 (강의평가). multi-hop 합성 답변 또는 날짜 산술이 필요한 질문에서 `GroundednessChecker.verify`가 `notGrounded` 판정 → HyDE 재시도 후에도 동일 → fallback. 이 단일 실패 모드 해소 시:

- 라우팅 top-3: 0.883 → 1.000 (모든 19건 정답 카테고리 포함 가정)
- 출처 인용: 0.877 → ~0.994
- 캠퍼스 필터: 0.647 → 0.823 (3건 회복, 잔여 3건은 §5.2 별도 원인)

### 5.2 Sparse-only BM25 후보가 캠퍼스 필터 우회

`src/retrieval/hybrid.py:64-65`의 `_sparse()`는 `decision.qdrant_filter`를 받지 않음 (Qdrant 필터는 dense 경로에만 적용). 결과:

- 실패 패턴: `["성남", "성남", null, null, null]` (3건) — 상위 2~4개는 dense 결과로 campus="성남" 정확, 하위는 sparse-only doc_id가 `payloads.get(doc_id, {})`에서 빈 dict를 반환해 `payload.get("campus")`가 None
- 영향: 캠퍼스 필터 정확도 0.647 (목표 1.00). multi-campus 질문에서 안전성 사고로 직결되는 항목 (§1.3).

조치 옵션 (구현 미반영):
1. `_fuse` 단계에서 sparse-only 후보의 payload를 Qdrant `retrieve(ids=...)`로 lazy fetch 후 campus 사후 필터
2. BM25 인덱스에 `campus` 메타데이터 동봉 후 sparse 검색 시 사전 필터
3. 캠퍼스 명시 쿼리에 한해 dense-only 모드로 강등 (단순)

### 5.3 RAGAS faithfulness 0.830 (목표 0.85, 0.020 미달)

답변에 코퍼스 외 추론이 일부 혼입. multi-hop 질문에서 LLM이 retrieved chunk에 없는 사실을 보충해 답변하는 경향 추정. answer_relevancy 0.520 또한 낮아, multi-hop 질문에서 답변이 질문의 모든 절(節)을 커버하지 못하는 경우가 많을 것으로 보임.

### 5.4 출처 인용 형식 1건 LLM 지시 미준수

`prompts.py:13` 시스템 프롬프트는 `[출처: doc_id, 카테고리, 캠퍼스]` 명시. 실패 1건 (`ed61650d`)은 `[강의평가 | 세계의문화와유산 | lec_lecture_reviews_35_c1]` 형식으로 prefix 누락. CITATION_PATTERN이 `[출처:]` 명시 검사이므로 미스매치. 프롬프트에 few-shot 예시 1~2개 추가가 효과적일 가능성.

### 5.5 RAGAS 진행 중 비치명 오류

- `Job[269/341/378] TimeoutError`: 3건 NaN
- `Job[345/393/405/425/461/477/521/561/565] BadRequestError(n must be 1)`: Upstage `solar-pro2`가 `n>1` 미지원, RAGAS의 self-consistency 샘플링 일부 실패 → 9건 NaN
- `Job[401/425] RateLimitError(429)`: 2건 NaN

총 ~14건 NaN/실패 → 144건 중 유효 평균. 동일 평가를 OpenAI/Anthropic LLM 또는 Upstage commitment tier에서 재실행 시 점수 변동 가능.

## 6. 보안 점검 항목 (미해결)

`benchmark/0/summary.csv`에 Upstage API key (`up_Pa2u…`) 평문 노출. `generate_eval_report.py`의 `_redact()`로 final_report.md에는 `***REDACTED***` 처리 완료, 원본 csv는 그대로. 조치 필요:
1. Upstage 콘솔에서 키 회전
2. 새 키로 `.env` 갱신
3. AutoRAG 재실행 후 `benchmark/0/summary.csv` 재생성

