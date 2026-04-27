# AutoRAG Reranker-OFF Optimization Report

> **실행일**: 2026-04-27
> **목적**: `reranker_enabled=False` 환경(2-core CPU 운영)에서 retrieval/generation 플로우 최적 조합 탐색
> **데이터**: `data/qa_autorag.parquet` (193 → 163 non-negative QA), `data/corpus_autorag.parquet` (2,382 docs)
> **임베딩**: `solar-embedding-1-large-passage` (Solar API, 4096 dim) — bge-m3 미사용
> **LLM**: `solar-pro3` (temperature=0.0, max_tokens=800)

---

## 1. 실험 개요

| Trial | 변수 | 결과 디렉터리 |
|-------|------|---------------|
| **A** | Lexical 토크나이저 (okt/kkma/kiwi) | `benchmark/no_rerank_A/` |
| **B** | Hybrid fusion (cc 4 norms × 21 weights + rrf) | `benchmark/no_rerank_B/` |
| **C** | 생성 baseline (단일 prompt + Solar) | `benchmark/no_rerank_C/` |
| **D** | Prompt 변형 4종 (current/citation/CoT/few-shot) | `benchmark/no_rerank_D/` |
| **E** | 우승 조합 검증 (supplementary + RAGAS) | `reports/eval_supplementary.json`, `reports/ragas_summary.json` |

설정 파일: `configs/autorag_no_rerank_{A,B,C,D}.yaml`

---

## 2. Trial A — Lexical Tokenizer

### 결과 (top_k=10, 163 QAs)

| Tokenizer | F1 | Recall | Precision | Latency |
|-----------|----|----|-----------|---------|
| **ko_okt** ⭐ | **0.1854** | **0.9018** | **0.1043** | **8.7ms** |
| ko_kkma | 0.1725 | 0.8466 | 0.0969 | 33.3ms |
| ko_kiwi | 0.1671 | 0.8221 | 0.0939 | 1.47s |

**우승: ko_okt** — 정확도 최고이면서도 kiwi 대비 170배 빠름. 현재 production과 일치 → **변경 없음**.

---

## 3. Trial B — Hybrid Fusion Sweep

### 상위 후보 (모두 F1=0.1985, Recall=0.9693에서 tie)

| Module | Normalize | Weight/k | F1 | Recall | Precision | Exec |
|--------|-----------|----------|-----|--------|-----------|------|
| **HybridCC** ⭐ | **dbsf** | **0.5** | 0.1985 | 0.9693 | 0.1117 | 40.6ms |
| HybridCC | mm | 0.4 | 0.1985 | 0.9693 | 0.1117 | 40.2ms |
| HybridCC | tmm | 0.5 | 0.1963 | 0.9571 | 0.1104 | 39.7ms |
| HybridCC | z | 0.35 | 0.1967 | 0.9693 | 0.1104 | 39.7ms |
| HybridRRF | — | k=4 | 0.1985 | 0.9693 | 0.1117 | 153.6ms |

**우승: HybridCC (dbsf, w=0.5)** — AutoRAG가 tie-break으로 선택. 사실상 5개 조합 동등 성능.

**핵심 관찰**: recall=0.969가 이 코퍼스의 **사실상 ceiling**. 5/163건은 어떤 fusion으로도 회복 불가능한 systematic miss.

**Production 변경 결정**: 최소 변경 원칙으로 **`hybrid_cc_normalize='mm'` 유지 + `hybrid_cc_weight: 0.15 → 0.4`**

---

## 4. Trial C — Generator Baseline

### 결과 (HybridCC dbsf w=0.5 + 단일 prompt)

| 메트릭 | 값 |
|--------|-----|
| **BLEU** | 24.30 |
| **METEOR** | 0.5037 |
| **ROUGE** | 0.3489 |
| 평균 출력 토큰 | 301.8 |
| 평균 prompt 토큰 | 696.1 |
| 생성 latency | 0.86s/query |

**해석**: 한국어 RAG 응답 기준 양호한 baseline 확립.

---

## 5. Trial D — Prompt Variants ⚠️

### 결과 (4 prompts 모두 동일 점수)

| Prompt | BLEU | METEOR | ROUGE |
|--------|------|--------|-------|
| 0. CoT | 14.509 | 0.4839 | 0.2029 |
| 1. Current (baseline) | 14.509 | 0.4839 | 0.2029 |
| 2. Few-shot | 14.509 | 0.4839 | 0.2029 |
| 3. Citation-forcing | 14.509 | 0.4839 | 0.2029 |

**관찰**: 4개 prompt 모두 점수가 **소수점 15자리까지 완벽히 동일**.

**해석 (둘 다 가능):**
- Solar @ temp=0이 prompt 변형에 robust → 동일 출력 생성
- AutoRAG의 알려진 버그 — 첫 prompt 결과를 4번 재활용

**결론**: AutoRAG 기반 prompt 비교는 **이번 세팅에서 신뢰할 수 없음**. 변경 없음 (현재 prompt 유지).

**참고**: Trial C와 Trial D BLEU가 다름 (24.3 vs 14.5)은 trial 간 generator 호출 경로 차이로 추정 (prompt_maker 평가용 generator vs 생성기 노드 generator).

---

## 6. Trial E — 우승 조합 실측 검증

### 변경: `src/config.py` `hybrid_cc_weight: 0.15 → 0.4`

### Supplementary 4지표 (193 QAs)

| 메트릭 | Baseline (w=0.15) | Trial E (w=0.4) | Δ | 목표 | 판정 |
|--------|-------------------|-----------------|----|------|------|
| **negative_rejection** | 0.833 | **0.867** | **+3.4pt** ✅ | ≥0.80 | PASS |
| **campus_filter** | 1.000 | 1.000 | 0 | =1.00 | PASS |
| **routing_top3** | 0.926 | **0.969** | **+4.3pt** ⭐ | ≥0.95 | **PASS (처음)** |
| **citation** | 0.896 | 0.847 | **−5.0pt** ⚠️ | ≥0.90 | FAIL |

### RAGAS 4지표 (163 QAs)

| 메트릭 | Baseline | Trial E | Δ |
|--------|----------|---------|---|
| faithfulness | 0.716 | 0.695 | **−2.2pt** ⚠️ |
| answer_relevancy | 0.437 | 0.420 | **−1.7pt** ⚠️ |
| **context_precision** | 0.733 | 0.763 | **+3.0pt** ✅ |
| **context_recall** | 0.832 | 0.869 | **+3.7pt** ✅ |

### 패턴 관찰

| 측면 | 변화 |
|------|------|
| **검색 메트릭 (4종)** | routing↑, negative↑, ctx_precision↑, ctx_recall↑ — 일관 개선 |
| **생성 메트릭 (3종)** | citation↓, faithfulness↓, answer_relevancy↓ — 일관 후퇴 |

**해석**: 가중치 0.4는 더 좋은 후보 문서를 가져오지만 컨텍스트가 더 다양해져 생성기가 인용/충실성에서 흔들림. **검색-생성 trade-off 명확히 관측**.

---

## 7. 메타데이터 캡처 검증

`reports/ragas_summary.json`에 새 메타데이터 자동 기록 확인 (오늘 추가한 기능):

```json
{
  "meta": {
    "started_at": "2026-04-27T15:09:32+09:00",
    "started_at_utc": "2026-04-27T06:09:32+00:00",
    "git_commit": "31fa8da",
    "platform": "Windows-11-10.0.26200-SP0",
    "python": "3.12.7",
    "config": {
      "hybrid_cc_weight": 0.4,
      "hybrid_cc_normalize": "mm",
      "reranker_enabled": false,
      "default_campus": "성남",
      "...": "..."
    }
  }
}
```

→ 향후 모든 RAGAS 실행이 정확한 시점/환경/설정을 자동 기록.

---

## 8. 보안 사후 조치

AutoRAG는 expanded yaml(API 키 평문)을 그대로 trial 산출물에 저장 → 매 실행 후 redact 필수.

**조치 완료**: `benchmark/no_rerank_{A,B,C,D}/0/{summary.csv,config.yaml}` 모두 `up_*` 패턴을 `REDACTED_API_KEY`로 치환. `benchmark/`는 `.gitignore` 등재됨.

**향후 개선**: `scripts/run_autorag.py` 종료 시 자동 sanitize 후크 추가 검토 (Task #20과 연관).

---

## 9. 의사결정 옵션

| 옵션 | 행동 | 장점 | 단점 |
|------|------|------|------|
| **A. 채택 (w=0.4 유지)** | 현재 변경 그대로 commit | routing 0.95 첫 달성, ctx_recall ↑ | citation/faithfulness ↓ |
| B. 절충 (w=0.25-0.30) | 0.15~0.4 사이 재탐색 | trade-off 균형 시도 | 추가 trial ~30분 |
| C. 거부 (revert to 0.15) | `src/config.py` 되돌리기 | citation/RAGAS 보존 | routing 0.95 미달 유지 |
| **D. 채택 + 후속 fix** | w=0.4 + fallback 답변 인용 패턴 추가 | 양쪽 회복 | 후속 개발 1~2일 |

**권장**: **옵션 D** — citation 후퇴 5pt는 모두 "정보없음" fallback 답변에서 발생 (잘못된 인용이 아닌 인용 부재). fallback 응답에 일반 안내 인용을 추가하는 후속 작업으로 회복 가능. routing 0.95 달성은 의미있는 milestone.

---

## 10. 산출물 인덱스

### Configs (커밋 대상)
- `configs/autorag_no_rerank_A.yaml`
- `configs/autorag_no_rerank_B.yaml`
- `configs/autorag_no_rerank_C.yaml`
- `configs/autorag_no_rerank_D.yaml`

### 코드 변경 (커밋 대상)
- `scripts/run_autorag.py` — Solar 임베딩(`solar_passage`, `solar_query`) 등록 추가
- `scripts/evaluate_ragas.py` — 메타데이터(`started_at`, `git_commit`, `config`) 자동 기록
- `src/config.py` — `hybrid_cc_weight: 0.15 → 0.4`

### 실측 결과 (커밋 대상)
- `reports/eval_supplementary.json` (Trial E 갱신)
- `reports/ragas_summary.json` (메타데이터 포함)
- `reports/ragas_by_collection.csv`
- `reports/ragas_report.json`
- `reports/autorag_no_rerank_summary.md` (이 보고서)

### 데이터 (커밋 대상)
- `data/qa_autorag.parquet` (negative 30건 제외, AutoRAG 호환)

### Benchmark 산출물 (git ignored)
- `benchmark/no_rerank_{A,B,C,D}/` — sanitize 완료, gitignore 유지

---

## 11. 결론

1. **Lexical**: ko_okt 유지 (production 일치) — 변경 없음
2. **Embedding**: solar-embedding-1-large-passage 단일 사용 (Trial 일관성 확보)
3. **Hybrid weight**: **0.15 → 0.4** (검색 4지표 +3~4pt, 생성 3지표 −2~5pt trade-off)
4. **Prompt**: AutoRAG에서 변별력 미관측 → 변경 없음
5. **Reranker**: passthrough 고정 (CPU 운영 결정 유지)

**핵심 milestone**: routing_top3 **0.926 → 0.969로 처음 목표(0.95) 달성**.

다음 단계는 사용자 결정 대기 (옵션 A~D).
