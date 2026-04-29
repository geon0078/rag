# Pipeline Variant Sweep

> 측정일: 2026-04-28T16:35:26+09:00  
> Golden Set: `data\eval_dataset_250_manual.parquet` · n=250

평가명세서 §8.1 기준: recall@5≥0.85, recall@10≥0.95, MRR≥0.65, nDCG@5≥0.75, citation≥0.90, faithfulness≥0.85

---

## 1. 통합 비교

| Variant | recall@5 | recall@10 | MRR | nDCG@5 | citation | grounded | retry | PASS |
|---------|----------|-----------|-----|--------|----------|----------|-------|------|
| **V1_baseline** | 0.810 | 0.826 | 0.619 | 0.662 | 1.000 | 0.944 | 0.128 | 1/5 |
| **V2_hyde_off** | 0.816 | 0.826 | 0.618 | 0.663 | 1.000 | 0.904 | 0.128 | 1/5 |
| **V3_final_top10** | 0.806 | 0.882 | 0.623 | 0.656 | 1.000 | 0.884 | 0.248 | 1/5 |
| **V4_cc_w_high** | 0.852 | 0.868 | 0.678 | 0.716 | 1.000 | 0.960 | 0.116 | 3/5 |
| **V5_cc_w_low** | 0.726 | 0.752 | 0.546 | 0.584 | 1.000 | 0.932 | 0.160 | 1/5 |
| **V6_rrf** | 0.802 | 0.818 | 0.617 | 0.656 | 1.000 | 0.952 | 0.124 | 1/5 |
| **V7_bge_rerank** | 0.838 | 0.856 | 0.693 | 0.724 | 1.000 | 0.920 | 0.108 | 2/5 |
| **V8_bge_rerank_top10** | 0.778 | 0.833 | 0.646 | 0.670 | 1.000 | 0.884 | 0.196 | 1/5 |
| **V9_bge_rerank_wider** | 0.832 | 0.850 | 0.690 | 0.720 | 1.000 | 0.928 | 0.092 | 2/5 |
| **V10_bge_rerank_w06** | 0.846 | 0.864 | 0.695 | 0.727 | 1.000 | 0.932 | 0.112 | 2/5 |
| **V11_rewriter_only** | 0.818 | 0.842 | 0.619 | 0.661 | 1.000 | 0.924 | 0.156 | 1/5 |
| **V12_rewriter_plus_bge** | 0.850 | 0.868 | 0.695 | 0.728 | 1.000 | 0.940 | 0.104 | 3/5 |

## 2. 베스트 시나리오

**🏆 V4_cc_w_high** — cc w=0.6 (semantic-heavy)

| 메트릭 | 값 | 목표 | 판정 |
|--------|-----|------|------|
| recall@5 | 0.852 | ≥0.85 | ✅ PASS |
| recall@10 | 0.868 | ≥0.95 | ❌ FAIL |
| MRR | 0.678 | ≥0.65 | ✅ PASS |
| nDCG@5 | 0.716 | ≥0.75 | ❌ FAIL |
| citation | 1.000 | ≥0.9 | ✅ PASS |

## 3. 변형별 설정

| Variant | description |
|---------|-------------|
| V1_baseline | default — hyde_on, cc w=0.4, top_k=30, final=5 |
| V2_hyde_off | hyde_off |
| V3_final_top10 | final=10 (이전 sweep 5/5 PASS 후보) |
| V4_cc_w_high | cc w=0.6 (semantic-heavy) |
| V5_cc_w_low | cc w=0.2 (BM25-heavy) |
| V6_rrf | RRF rank fusion |
| V7_bge_rerank | bge-reranker-v2-m3-ko ON |
| V8_bge_rerank_top10 | bge-reranker ON + final=10 |
| V9_bge_rerank_wider | bge-reranker ON + top_k=50 |
| V10_bge_rerank_w06 | bge-reranker ON + cc w=0.6 |
| V11_rewriter_only | rewriter ON (no rerank) |
| V12_rewriter_plus_bge | rewriter + bge-reranker (combo) |

## 4. 통과 기준 미달 항목

- **V1_baseline** (1/5 PASS): recall@5, recall@10, mrr, ndcg@5
- **V2_hyde_off** (1/5 PASS): recall@5, recall@10, mrr, ndcg@5
- **V3_final_top10** (1/5 PASS): recall@5, recall@10, mrr, ndcg@5
- **V4_cc_w_high** (3/5 PASS): recall@10, ndcg@5
- **V5_cc_w_low** (1/5 PASS): recall@5, recall@10, mrr, ndcg@5
- **V6_rrf** (1/5 PASS): recall@5, recall@10, mrr, ndcg@5
- **V7_bge_rerank** (2/5 PASS): recall@5, recall@10, ndcg@5
- **V8_bge_rerank_top10** (1/5 PASS): recall@5, recall@10, mrr, ndcg@5
- **V9_bge_rerank_wider** (2/5 PASS): recall@5, recall@10, ndcg@5
- **V10_bge_rerank_w06** (2/5 PASS): recall@5, recall@10, ndcg@5
- **V11_rewriter_only** (1/5 PASS): recall@5, recall@10, mrr, ndcg@5
- **V12_rewriter_plus_bge** (3/5 PASS): recall@10, ndcg@5

## 5. 산출물

- `reports/pipeline_sweep.json` — 모든 변형 raw 메트릭
- `reports/pipeline_sweep.md` — 이 보고서
