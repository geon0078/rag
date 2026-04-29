# Adversarial Evaluation

> 측정일: 2026-04-28T22:15:06+09:00  
> Adversarial Set: `D:\github\eulGPT\AI-ver-4\data\eval_dataset_250_manual.parquet` · n=250

평가명세서 §8.1 기준: recall@5≥0.85, recall@10≥0.95, MRR≥0.65, nDCG@5≥0.75, citation≥0.90, faithfulness≥0.85

---

## 1. 전체 메트릭

| 메트릭 | 값 | 목표 | 판정 |
|--------|-----|------|------|
| recall@5 | 0.856 | ≥0.85 | ✅ PASS |
| recall@10 | 0.866 | ≥0.95 | ❌ FAIL |
| MRR | 0.671 | ≥0.65 | ✅ PASS |
| nDCG@5 | 0.711 | ≥0.75 | ❌ FAIL |
| citation | 1.000 | ≥0.9 | ✅ PASS |
| grounded rate | 0.936 | — | — |
| HyDE retry rate | 0.128 | — | — |

**판정: 3/5 PASS** · 미달: recall@10, ndcg@5

## 2. Challenge type 별 breakdown

| type | n | recall@5 | recall@10 | MRR | nDCG@5 | citation | grounded | retry |
|------|---|----------|-----------|-----|--------|----------|----------|-------|
| **T1_conversational** | 119 | 0.924 | 0.924 | 0.781 | 0.817 | 1.000 | 0.975 | 0.067 |
| **T2_vague** | 13 | 0.615 | 0.615 | 0.276 | 0.358 | 1.000 | 1.000 | 0.077 |
| **T3_paraphrase** | 52 | 0.904 | 0.904 | 0.634 | 0.701 | 1.000 | 0.942 | 0.135 |
| **T4_multi_intent** | 26 | 0.712 | 0.750 | 0.629 | 0.618 | 1.000 | 0.962 | 0.154 |
| **T5_inference** | 40 | 0.762 | 0.800 | 0.546 | 0.584 | 1.000 | 0.775 | 0.300 |

## 3. Hop type 별

| type | n | recall@5 | MRR | nDCG@5 | grounded |
|------|---|----------|-----|--------|----------|
| **multi** | 24 | 0.542 | 0.514 | 0.463 | 0.917 |
| **single** | 226 | 0.889 | 0.687 | 0.737 | 0.938 |

## 4. Source collection 별 retrieval

| 컬렉션 | n | recall@5 | MRR |
|--------|---|----------|-----|
| FAQ | 17 | 0.882 | 0.725 |
| 강의평가 | 54 | 0.852 | 0.680 |
| 교육과정 | 11 | 0.955 | 1.000 |
| 기타 | 9 | 0.944 | 0.767 |
| 시설_연락처 | 37 | 0.878 | 0.731 |
| 장학금 | 17 | 0.941 | 0.752 |
| 학과정보 | 4 | 1.000 | 1.000 |
| 학사일정 | 10 | 0.800 | 0.583 |
| 학사정보 | 35 | 0.957 | 0.810 |
| 학칙_조항 | 56 | 0.714 | 0.405 |

## 5. 약점 진단

- **T2_vague**: recall@5=0.615 (목표 0.85 미달)
- **T4_multi_intent**: recall@5=0.712 (목표 0.85 미달)
- **T5_inference**: recall@5=0.762 (목표 0.85 미달)

## 6. 산출물

- `reports/eval_adversarial.json` — 모든 메트릭 + per-row
- `reports/eval_adversarial.md` — 이 보고서
