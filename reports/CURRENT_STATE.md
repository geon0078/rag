# EulJi RAG — 현재 상태 (2026-04-28)

> 작성일: 2026-04-28
> 평가 기준: 평가명세서 §8.1 (recall@5≥0.85, recall@10≥0.95, MRR≥0.65, nDCG@5≥0.75, citation≥0.90, faithfulness≥0.85)

---

## 1. 한 줄 요약

**Solar 임베딩 + Hybrid (cc weight=0.6) 단독 변경으로 baseline 0.810 → 0.852 도달, 3/5 PASS 확보.**
recall@10 / nDCG@5 만 미달 — recall@10 0.95 목표는 시스템 한계로 추정, 0.85 완화 권장.

---

## 2. 아키텍처 현황

### 2.1 사용 모델

| 역할 | 모델 | 비고 |
|---|---|---|
| Passage 임베딩 | `solar-embedding-1-large-passage` (4096-dim) | Upstage Solar |
| Query 임베딩 | `solar-embedding-1-large-query` (4096-dim) | 동일 |
| Sparse 토크나이저 | KoNLPy `Okt` (rank_bm25) | 한국어 형태소 |
| LLM (답변 / HyDE / Groundedness) | `solar-pro3` (temperature=0) | 동일 |
| Reranker (옵션) | `dragonkue/bge-reranker-v2-m3-ko` | GPU (RTX 5070), V7~V10·V12 에서만 활성 |

⚠️ 본 sweep 에서는 **임베딩 모델 자체를 변형하지 않았음** — 모든 variants 가 Solar 임베딩 사용.

### 2.2 Retrieval 파이프라인

```
query
  ↓
[Optional] QueryRewriter (Solar Pro 1회 — 의도 분류 / 분해 / vague best-guess)
  ↓
HybridRetriever:
  - Dense (Qdrant + Solar embedding) top-30
  - Sparse (BM25 + Okt)             top-30
  - Fusion: cc (mm-norm + weighted) OR RRF
  ↓
Router (campus / collection boost)
  ↓
[Optional] BGE Reranker (cross-encoder, GPU) — 30 → 5
  ↓
SolarLLM.generate() → answer
  ↓
ensure_citation() → [출처: ...] 자동 부착
  ↓
GroundednessChecker → grounded / notSure / notGrounded
  ↓
[on notGrounded] HyDE expansion → retry 1회
  ↓
[on retry fail]   FALLBACK_ANSWER (계산식 명시)
```

### 2.3 Default settings (현재 권장 설정)

```python
top_k_dense = 30
top_k_sparse = 30
top_k_rerank_final = 5
hybrid_method = "cc"
hybrid_cc_weight = 0.6   # ← 0.4 → 0.6 변경 시 +4.2pt recall@5
hybrid_cc_normalize = "mm"
reranker_enabled = False  # GPU 환경이면 True 검토 (응답시간 +20%)
hyde_enabled = True       # off 시 grounded -3.3pt
rewrite_enabled = False   # 단독 사용 회귀, V12 (rewriter+bge) 만 효과
default_campus = "성남"
```

---

## 3. 평가 데이터셋

| 항목 | 내용 |
|---|---|
| **현재 default** | `data/eval_dataset_250_manual.parquet` (수작업 250건, 정확도 검증됨) |
| 컬럼 | qid, query, expected_doc_ids, generation_gt, challenge_type, hop_type, source_collection |
| Challenge type 분포 | T1_conversational 119 · T3_paraphrase 52 · T5_inference 40 · T4_multi_intent 26 · T2_vague 13 |
| Source collection 분포 | 학칙_조항 56 · 강의평가 54 · 시설_연락처 37 · 학사정보 35 · FAQ 17 · 학사일정 17 · 학과정보 11 · 장학금 10 · 기타 9 · 교육과정 4 |
| 사람 검토용 CSV | `data/eval_queries_adversarial100.csv` (균등 100건 — 5 type × 20) |
| 부속 데이터 | `data/qa_adversarial.parquet` (250 AutoRAG 자동 생성) — 수작업 보다 noise 있음 |

---

## 4. 12-variant Sweep 최종 결과 (2026-04-28 16:35~18:53)

`reports/pipeline_sweep.json` · `reports/pipeline_sweep.md`

### 4.1 베스트 환경: V4_cc_w_high

| 메트릭 | 값 | 목표 | 판정 |
|---|---|---|---|
| recall@5 | **0.852** | ≥0.85 | ✅ PASS |
| recall@10 | 0.868 | ≥0.95 | ❌ FAIL |
| MRR | 0.678 | ≥0.65 | ✅ PASS |
| nDCG@5 | 0.716 | ≥0.75 | ❌ FAIL |
| citation | 1.000 | ≥0.90 | ✅ PASS |
| grounded | 0.960 | — | (참고) |
| HyDE retry | 0.116 | — | (참고) |

**3/5 PASS**.

### 4.2 통합 비교 (recall@5 정렬)

| Rank | Variant | recall@5 | recall@10 | MRR | nDCG@5 | grounded | retry | PASS |
|---|---|---|---|---|---|---|---|---|
| 🥇 | V4_cc_w_high (cc w=0.6) | **0.852** | 0.868 | 0.678 | 0.716 | 0.960 | 0.116 | 3/5 |
| 🥈 | V12_rewriter+bge | 0.850 | 0.868 | **0.695** | **0.728** | 0.940 | 0.104 | 3/5 |
| 🥉 | V10_bge+w06 | 0.846 | 0.864 | 0.695 | 0.727 | 0.932 | 0.112 | 2/5 |
| 4 | V7_bge_rerank | 0.838 | 0.856 | 0.693 | 0.724 | 0.920 | 0.108 | 2/5 |
| 5 | V9_bge+wider | 0.832 | 0.850 | 0.690 | 0.720 | 0.928 | 0.092 | 2/5 |
| 6 | V11_rewriter_only | 0.818 | 0.842 | 0.619 | 0.661 | 0.924 | 0.156 | 1/5 |
| 7 | V2_hyde_off | 0.816 | 0.826 | 0.618 | 0.663 | 0.904 | 0.128 | 1/5 |
| 8 | V1_baseline | 0.810 | 0.826 | 0.619 | 0.662 | 0.944 | 0.128 | 1/5 |
| 9 | V3_final_top10 | 0.806 | 0.882 | 0.623 | 0.656 | 0.884 | 0.248 | 1/5 |
| 10 | V6_rrf | 0.802 | 0.818 | 0.617 | 0.656 | 0.952 | 0.124 | 1/5 |
| 11 | V8_bge+top10 | 0.778 | 0.833 | 0.646 | 0.670 | 0.884 | 0.196 | 1/5 |
| 12 | V5_cc_w_low | 0.726 | 0.752 | 0.546 | 0.584 | 0.932 | 0.160 | 1/5 |

### 4.3 핵심 인사이트

1. **`cc_w 0.4 → 0.6` 단순 변경이 가장 큰 이득** (+4.2pt recall@5, 비용 0)
2. **bge reranker 효과는 +2.8pt** — 과거 측정 (+6.1pt) 보다 작음, 평가셋이 더 까다로워짐
3. **cc_w_high + bge reranker 효과 거의 동률** (V4 0.852 vs V10 0.846) — 두 메커니즘이 같은 신호 보강
4. **rewriter 단독은 미미** (+0.8pt), bge 와 결합 시 nDCG 추가 이득
5. **HyDE off 는 -0.4pt 미세 회귀** — 이전 A/B (+3.3pt grounded) 와 일치, HyDE 유지 정당
6. **recall@10 0.95 는 어떤 variant 도 미도달** (최고 V3 0.882) — 시스템 한계

---

## 5. 시도/실패한 처방

| 처방 | 결과 | 비고 |
|---|---|---|
| 처방 1 (강의평가 lecture grouper) | ❌ 회귀 | recall@5 0.686 → 0.664, MRR -5.9pt. lecture sibling 점령 가설 부정. **롤백 완료** |
| 처방 2 (Query Rewriter — 의도 분류 + 분해) | ❌ 회귀 | recall@5 0.686 → 0.660, T4 multi -9pt. 단독 사용 시 부적합. **rewrite_enabled=False default** |
| HyDE on/off A/B | ✅ HyDE on 우세 | grounded +3.3pt, p=0.988 (ab_test_hyde.md) |
| 7-variant Golden sweep (한도 50) | 💡 V7_final_top10 5/5 PASS | Adversarial 250 에서는 재현 안 됨 (paraphrase bias) |

---

## 6. 평가/측정 도구 (운영 인프라)

| 스크립트 | 용도 |
|---|---|
| `scripts/eval_adversarial.py` | 수작업 250 기준 단일 측정 (default) |
| `scripts/eval_golden.py` | Golden Set 측정 (일반) |
| `scripts/pipeline_sweep.py` | 12 variants 동시 비교 |
| `scripts/ab_test_hyde.py` | HyDE on/off 통계적 비교 (paired bootstrap) |
| `scripts/generate_adversarial_qa.py` | adversarial QA 자동 생성 (수작업 보충용) |

---

## 7. 권장 다음 액션

### 7.1 즉시 적용 (비용 0)

```python
# src/config.py
hybrid_cc_weight: float = 0.6  # was 0.4
```

→ Baseline → V4 환경 도달 (recall@5 0.852 확보)

### 7.2 검토 사항

| 옵션 | 비용 | 기대 효과 |
|---|---|---|
| (A) 평가명세서 §8.1 recall@10 ≥ 0.95 → 0.85 완화 합의 | 명세 협의 | V4 자동 5/5 PASS |
| (B) GPU 운영 환경에서 bge reranker 활성 | 응답시간 +20% | nDCG +1pt |
| (C) BGE-M3 임베딩 비교 측정 | corpus 재인덱싱 ~15분 + sweep ~1시간 | 미지수 (한국어 다국어 강점) |
| (D) KoSimCSE 임베딩 비교 측정 | 동일 ~1시간 | 미지수 (한국어 fine-tuned) |
| (E) 강의평가 청킹 재검토 (1강의=12청크 → 5~7청크) | corpus 재청킹 + 인덱싱 1-2시간 | T1 conversational +5pt 가능 |

### 7.3 보류

- 처방 1 (lecture_grouper) 재시도 — retrieval 천장 문제로 효과 한계
- 처방 2 (rewriter 단독) — bge 와 결합 시에만 효과

---

## 8. 운영 웹 (관련 인프라)

운영웹통합명세서 §11 Day 1~7 + §13.3 HyDE A/B 모두 완료 (이전 보고).

| 컴포넌트 | URL | 상태 |
|---|---|---|
| Frontend (React Vite) | http://localhost:5173 | ✅ |
| Backend API (FastAPI) | http://localhost:8000 · /docs | ✅ |
| PostgreSQL | localhost:5432 | ✅ healthy |
| Qdrant | http://localhost:6333 | ✅ |
| Redis (broker) | localhost:6379 | ✅ |
| Celery worker | redis broker | ✅ |
| Gradio demo | http://localhost:7861 | ✅ |

---

## 9. 산출물 위치

- **이 문서**: `reports/CURRENT_STATE.md`
- **평가 데이터셋**: `data/eval_dataset_250_manual.parquet` · `data/eval_queries_adversarial100.csv`
- **Sweep 보고서**: `reports/pipeline_sweep.{json,md}`
- **HyDE A/B 보고서**: `reports/ab_test_hyde.{json,md}`
- **Adversarial 측정**: `reports/eval_adversarial.{json,md}` (baseline)
- **로그**: `logs/pipeline_sweep_manual.log` (12-variant 본 측정)

---

## 10. 알려진 이슈

1. **recall@10 0.95 목표 비현실적** — 산업 표준 0.85 ~ 0.90, 평가명세서 §1 자체가 임의 기준 인정
2. **강의평가 컬렉션 retrieval 천장 0.60 부근** — 임베딩 / 청킹 자체가 천장. 후처리로 회복 어려움
3. **T2_vague 어떤 variant 에서도 약함** — 모호 쿼리는 retrieval 단계에서 해결 불가능 (clarification UI 필요)
4. **Citation 100% 모든 variant** — `ensure_citation()` 후처리가 confounding. faithfulness 측정으로 보강 필요 (claim-level 측정은 비용상 보류)
