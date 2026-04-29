# HyDE on/off A/B Test

> 측정일: 2026-04-28T00:48:37+09:00  
> Golden Set: `D:\github\eulGPT\AI-ver-4\data\golden_candidates_v1.parquet`  
> Paired n: **150**  
> Bootstrap iters: 1000

---

## 1. Arm 요약 (paired bootstrap)

| 메트릭 | A (HyDE on) | B (HyDE off) | Δ (B−A) | 95% CI | p(Δ≤0) |
|--------|-------------|---------------|---------|--------|--------|
| retrieval recall@5 | 0.873 | 0.893 | +0.020 | [+0.000, +0.047] | 0.057 |
| grounded rate | 0.960 | 0.927 | -0.033 | [-0.073, +0.000] | 0.988 |
| citation accuracy | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| retry rate | 0.100 | 0.113 | +0.013 | [-0.013, +0.047] | 0.250 |

## 2. Retrieval 분리 메트릭 (Arm 별)

| 메트릭 | A (on) | B (off) | Δ |
|--------|--------|---------|---|
| recall@5 | 0.873 | 0.893 | +0.020 |
| recall@10 | 0.880 | 0.893 | +0.013 |
| mrr | 0.747 | 0.747 | -0.000 |
| ndcg@5 | 0.778 | 0.784 | +0.006 |

## 3. Generation 메트릭 (Arm 별)

| 메트릭 | A (on) | B (off) | Δ |
|--------|--------|---------|---|
| grounded_rate | 0.960 | 0.927 | -0.033 |
| citation | 1.000 | 1.000 | +0.000 |
| retry_rate | 0.100 | 0.113 | +0.013 |

## 4. 판정 가이드

- `p(Δ≤0)` 가 작을수록 (≤ 0.05) HyDE off 가 통계적으로 더 나음.
- `p(Δ≤0)` 가 클수록 (≥ 0.95) HyDE on 이 더 나음.
- 그 사이는 차이 미미 — 비용·지연 측면에서 HyDE 비활성화 검토 가능.

## 5. 산출물

- `reports/ab_test_hyde.json` — 모든 메트릭 + per-row
- `reports/ab_test_hyde.md` — 이 보고서
