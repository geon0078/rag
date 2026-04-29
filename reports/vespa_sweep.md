# Vespa Retriever Sweep — 시나리오 A 풀 교체 측정

> 측정일: 2026-04-28T22:30:21+09:00  
> Eval set: `D:\github\eulGPT\AI-ver-4\data\eval_dataset_250_manual.parquet` · n=250

우리 Qdrant+Okt 베이스라인 (V4_cc_w_high): recall@5 **0.852** · MRR 0.678 · nDCG@5 0.716

---

## 1. Vespa 변형 비교

| Variant | rank_profile | cc_w | recall@5 | recall@10 | MRR | nDCG@5 | vs V4 (recall@5) |
|---------|--------------|------|----------|-----------|-----|--------|------------------|
| **Vespa_BM25** | bm25_only | 0.6 | 0.000 | 0.000 | 0.000 | 0.000 | -0.852 |
| **Vespa_Vector** | vector_only | 0.6 | 0.824 | 0.877 | 0.677 | 0.706 | -0.028 |
| **Vespa_Hybrid_w0.6** | hybrid_cc | 0.6 | 0.842 | 0.903 | 0.696 | 0.723 | -0.010 |
| **Vespa_Hybrid_w0.4** | hybrid_cc | 0.4 | 0.842 | 0.899 | 0.690 | 0.720 | -0.010 |
| **Vespa_Hybrid_w0.8** | hybrid_cc | 0.8 | 0.846 | 0.903 | 0.697 | 0.725 | -0.006 |
| **Vespa_RRF** | rrf_approx | 0.6 | 0.848 | 0.905 | 0.697 | 0.726 | -0.004 |

## 2. 컬렉션별 recall@5 (Vespa best variant)

Best variant: **Vespa_RRF** (recall@5 0.848)

| 컬렉션 | n | recall@5 | MRR |
|--------|---|----------|-----|
| FAQ | 17 | 0.941 | 0.784 |
| 강의평가 | 54 | 0.833 | 0.742 |
| 교육과정 | 11 | 1.000 | 0.909 |
| 기타 | 9 | 1.000 | 0.815 |
| 시설_연락처 | 37 | 0.878 | 0.784 |
| 장학금 | 17 | 0.912 | 0.809 |
| 학과정보 | 4 | 1.000 | 1.000 |
| 학사일정 | 10 | 0.800 | 0.592 |
| 학사정보 | 35 | 0.871 | 0.750 |
| 학칙_조항 | 56 | 0.723 | 0.438 |

## 3. 결론

- ⚠ Vespa best (Vespa_RRF) recall@5 0.848 < V4 0.852
- 격차: -0.004
- 한국어 형태소 분석 (Okt) 부재 영향 추정 — Vespa 의 char-gram 이 Okt 에 못 미침.

## 4. 산출물

- `reports/vespa_sweep.json`
- `reports/vespa_sweep.md` — 본 보고서
