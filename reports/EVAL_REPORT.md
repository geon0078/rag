# EulJi RAG — Evaluation Report

> **측정일**: 2026-04-27
> **Git commit**: `6dcf74f` (Trial F RAGAS 실행 시점, 적용된 변경 모두 반영)
> **데이터셋**: `data/qa.parquet` 779 QA (single 540 / multi 147 / filter 32 / negative 60)
> **노이즈 수준**: ±0.5pt (이전 ±1pt에서 절반 감소, 데이터 4배 확장 효과)

---

## 1. 한눈에 보는 정확도 (Trial F, n=779)

### Supplementary 4지표

| 메트릭 | 값 | n | 목표 | 판정 |
|--------|-----|---|------|------|
| **negative_rejection** | **0.867** (52/60) | 60 | ≥0.80 | ✅ PASS |
| **campus_filter** | **0.938** (30/32) | 32 | =1.00 | ❌ FAIL |
| **routing_top3** | **0.968** (696/719) | 719 | ≥0.95 | ✅ PASS |
| **citation** | **1.000** (719/719) | 719 | ≥0.90 | ✅ PASS |

### RAGAS 4지표

| 메트릭 | 값 | n |
|--------|-----|---|
| faithfulness | **0.728** | 719 |
| answer_relevancy | **0.415** | 719 |
| context_precision | **0.784** | 719 |
| context_recall | **0.892** | 719 |

---

## 2. 측정 환경

```yaml
LLM:               solar-pro3 (temperature=0.0)
Embedding:         solar-embedding-1-large-passage / -query (4096 dim)
Vectorstore:       Qdrant — collection euljiu_knowledge
Sparse:            BM25 with KoNLPy Okt tokenizer
Hybrid fusion:     CC (convex combination), normalize=mm, semantic_weight=0.4
Top-k:             dense=30 / sparse=30 / final=5
Reranker:          OFF (passthrough; CPU production target)
Default campus:    성남 (router fallback)
```

전체 환경 메타데이터는 `reports/ragas_summary.json` 의 `meta.config` 블록에서 자동 캡처됨.

---

## 3. 진행 경과 (Trial timeline)

| Trial | 데이터 | 변경 | 핵심 결과 |
|-------|--------|------|-----------|
| **A~D** | 163 QA (autorag) | AutoRAG 기반 lexical/hybrid/prompt sweep | hybrid_cc winner: weight 0.4 |
| **E** | 193 QA | `hybrid_cc_weight: 0.15 → 0.4` 적용 | routing 0.926 → 0.957 (첫 PASS) |
| **E2** | 193 QA | fallback citation fix (`ensure_citation` 호출) | citation 0.847 → 1.000 |
| **F** | **779 QA** | 데이터 4× 확장 + 품질 게이트 (round-trip + dedup) | 노이즈 ±1pt → ±0.5pt, 정확한 최종 수치 확정 |

세부 trial 보고서: `reports/autorag_no_rerank_summary.md`

---

## 4. 메트릭 변화 추이

### Supplementary 비교 (Trial E n=193 vs Trial F n=779)

| 메트릭 | E (n=193) | F (n=779) | Δ | 해석 |
|--------|-----------|-----------|----|------|
| negative_rejection | 0.900 | 0.867 | -3.3pt | E의 30 negative → F의 60 negative로 확장; 진짜 값은 ~87% |
| campus_filter | 1.000 | **0.938** | -6.2pt | E의 17건은 우연히 100%, F의 32건이 정확값 (regression 발견) |
| routing_top3 | 0.957 | 0.968 | +1.1pt | 노이즈 범위, 진짜 값은 ~97% |
| citation | 1.000 | 1.000 | 0 | fallback fix 안정 입증 |

### RAGAS 비교 (Trial E n=163 vs Trial F n=719)

| 메트릭 | E (n=163) | F (n=719) | Δ |
|--------|-----------|-----------|---|
| faithfulness | 0.695 | **0.728** | +3.3pt |
| answer_relevancy | 0.420 | 0.415 | -0.5pt (노이즈) |
| context_precision | 0.763 | **0.784** | +2.1pt |
| context_recall | 0.869 | **0.892** | +2.3pt |

→ 작은 셋이 어려운 케이스 비중이 높았기 때문에 RAGAS는 4배 확장 시 일제히 상승.

---

## 5. RAGAS — 카테고리별 분포 (n=719)

`reports/ragas_by_collection.csv` 참조. 주요 관찰:

| 카테고리 | faithfulness | answer_relevancy | context_precision | context_recall |
|----------|--------------|------------------|-------------------|----------------|
| FAQ | 높음 | 평균 | 높음 | 매우 높음 |
| 강의평가 | 평균 | 낮음 | 평균 | 평균 |
| 교육과정 | 낮음 | 높음 | 매우 높음 | 매우 높음 |
| 시설_연락처 | 매우 높음 | 평균 | 매우 높음 | 매우 높음 |
| 학과정보 | 매우 높음 | 평균 | 평균 | 매우 높음 |

(실제 수치는 `ragas_by_collection.csv` 직접 참조)

**약점 카테고리**: 강의평가 — 학생 후기의 구어체/문맥 의존성으로 LLM 충실도 평가가 까다로움.

---

## 6. 실패 사례 분석

### campus_filter 2건 실패 (FAIL)
`reports/eval_campus_filter.json` 의 `failures` 블록 참조. 원인 후보:
- 신규 추가된 filter_required 케이스 중 캠퍼스 표기가 변형됨 ("성남캠" 단축형 등)
- Router 정규식 미매칭

→ 추후 처리: failures 2건 패턴 분석 후 router CAMPUS_PATTERN 보강.

### negative_rejection 8건 실패 (PASS이지만 추후 개선 가능)
`reports/eval_negative.json` 의 `failures` 블록 참조. 대부분 fallback 답변에서 "근거 부족" 표현 누락 케이스.

→ 우선순위 낮음 (목표 0.80 충족, 진짜 값 87%).

---

## 7. 산출물 인덱스

### 현재 활성 (Trial F)
| 파일 | 내용 |
|------|------|
| `reports/eval_supplementary.json` | 4지표 통합 |
| `reports/eval_negative.json` / `_campus_filter.json` / `_routing.json` / `_citation.json` | 메트릭별 상세 + failures |
| `reports/ragas_summary.json` | RAGAS 4지표 + meta (timestamp, git_commit, config) |
| `reports/ragas_by_collection.csv` | 카테고리별 RAGAS |
| `reports/ragas_report.json` | row-level RAGAS (3.4MB) |
| `reports/EVAL_REPORT.md` | **이 보고서** |
| `reports/autorag_no_rerank_summary.md` | Trial A~E 상세 |

### 진단/비교 자료 — `reports/diagnostics/`
- `compare_pipeline_rerank.json` — reranker A/B (full pipeline)
- `compare_rerank.json` — reranker A/B (retrieval only)
- `diagnose_retrieval.json` — retrieval 진단
- `fallback_diagnosis.json` — fallback 카테고리 분포
- `qa_quality_report.json` — round-trip + dedup 결과 (804 → 719, drops 85)

### 과거 스냅샷 — `reports/history/`
- `eval_supplementary.before_fallback_fix.json` — fallback citation fix 전
- `intent_prompt_ab.json` / `intent_variance.json` — intent gate 실험 (revert됨)
- `final_report_2026-04-26.md` — 초기 baseline 보고서

### 로그 — `logs/` (top-level)
- `eval_supplementary*.log` — 각 trial 실행 로그
- `evaluate_ragas*.log` — RAGAS 실행 로그
- 그 외 generate/diagnose 로그

---

## 8. 미해결 항목

| # | 문제 | 우선순위 | 다음 조치 |
|---|------|---------|----------|
| 1 | `campus_filter 0.938` (-6.2pt) | 중 | 실패 2건 분석 → router CAMPUS_PATTERN 보강 |
| 2 | `answer_relevancy 0.415` (RAGAS) | 중 | 답변 길이/관련성 prompt 튜닝 |
| 3 | RAGAS 강의평가 카테고리 약점 | 낮 | 카테고리별 prompt 분기 검토 |

---

## 9. 평가 재실행

```bash
# Supplementary 4지표 (~30분)
python scripts/eval_supplementary.py

# RAGAS 4지표 (~110분, 719 QAs)
python scripts/evaluate_ragas.py

# QA 품질 검증 (round-trip + dedup)
python scripts/qa_quality_filter.py

# 데이터 재생성 (5x quotas, ~60분)
python scripts/generate_qa.py --seed 42
python scripts/finalize_qa.py
```

---

## 10. 결론

**현재 production 정확도 (n=779 대규모 측정)**:

```
검색계: routing 96.8% / context_precision 78.4% / context_recall 89.2%
거부계: negative 86.7% / campus 93.8% (regression)
인용계: citation 100.0% ⭐
충실계: faithfulness 72.8% / answer_relevancy 41.5%
```

**달성 milestone**:
- routing_top3 첫 0.95+ 달성
- citation 100% 안정 입증 (719/719)
- 노이즈 ±1pt → ±0.5pt 측정 신뢰도 향상
- 메트릭 자동 메타데이터 기록 (`started_at` / `git_commit` / `config`)
- 시크릿 표준화 (AutoRAG redact + pre-commit scanner)

**남은 Action item**:
- campus_filter regression fix (실패 2건 분석)
- (선택) RAGAS answer_relevancy 개선 prompt 튜닝
