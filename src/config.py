"""Centralized config loaded from environment + .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    corpus_path: Path = PROJECT_ROOT / "data" / "corpus.parquet"
    bm25_index_path: Path = PROJECT_ROOT / "data" / "bm25_outline.pkl"
    log_dir: Path = PROJECT_ROOT / "logs"

    upstage_api_key: str = Field(default_factory=lambda: os.getenv("UPSTAGE_API_KEY", ""))
    upstage_base_url: str = "https://api.upstage.ai/v1/solar"
    embedding_model_passage: str = "solar-embedding-1-large-passage"
    embedding_model_query: str = "solar-embedding-1-large-query"
    embedding_dim: int = 4096
    llm_model_pro: str = "solar-pro3"
    llm_model_mini: str = "solar-mini"
    llm_temperature: float = 0.0

    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_collection: str = "euljiu_outline"

    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    cache_ttl_default_sec: int = 60 * 60 * 24
    cache_ttl_calendar_sec: int = 60 * 60 * 6
    cache_similarity_threshold: float = 0.95

    # Tuned via AutoRAG benchmark (benchmark/0/summary.csv, 2026-04-26):
    # - bm25 ko_okt won lexical (Recall@10=0.902)
    # - hybrid_cc with mm-norm, semantic_weight=0.15 won fusion
    #   (F1=0.193, Recall@10=0.951) over RRF (F1=0.152)
    # - bge-reranker-v2-m3-ko top_k=5 lifted Recall@5 to 0.939
    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_rerank_final: int = 5
    top_k_rerank_retry: int = 10

    hybrid_method: str = "cc"  # "cc" (convex combination) or "rrf"
    # AutoRAG no-rerank sweep (2026-04-27) over 4 norms × 21 weights tied at
    # F1=0.1985 / Recall=0.9693 across {mm:0.4, dbsf:0.5, tmm:0.5, z:0.35,
    # rrf:k=4}. Picked mm normalize (current) + 0.4 weight as the minimal-
    # change winner. Previous 0.15 came from the original AutoRAG run that
    # included reranker; 0.4 is the rerank-OFF optimum.
    hybrid_cc_weight: float = 0.6  # semantic weight; (1 - w) goes to BM25
    # 2026-04-28 sweep (12 variants × manual 250) 기준 V4_cc_w_high 가 베스트:
    # recall@5 0.852 (목표 0.85 PASS), MRR 0.678, nDCG 0.716, grounded 0.960.
    # 0.4 baseline 대비 +4.2pt recall@5, 비용 0.
    hybrid_cc_normalize: str = "mm"  # mm | tmm | z | dbsf
    rrf_k: int = 60  # used only when hybrid_method == "rrf"

    reranker_model: str = "dragonkue/bge-reranker-v2-m3-ko"
    # Reranker is disabled by default for the 2-core CPU production target.
    # Full-pipeline A/B (193 QA, reports/compare_pipeline_rerank.json) showed
    # bge-reranker lifts routing_top3 +6.1pt and citation +6.7pt at +1.75s/q
    # on GPU; on a 2-core CPU the same model is estimated 50-60x slower
    # (~15-30s/q), which is unusable in production. Toggle on (e.g. for GPU
    # eval runs) via RERANKER_ENABLED=true.
    reranker_enabled: bool = Field(
        default_factory=lambda: os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    )

    bm25_tokenizer: str = "okt"

    # When the router cannot extract an explicit campus signal from the query,
    # fall back to this campus instead of issuing an unfiltered search. The
    # main campus for student-facing 학사 queries is 성남; defaulting to it
    # closes the campus_filter eval failures (qids that expect=성남 but
    # router previously returned None → no filter → 의정부 docs leaked in).
    default_campus: str = "성남"

    embed_batch_size: int = 100
    embed_retry_max: int = 5

    # LLM request timeout. Without this, a single hung Solar API call can
    # stall the eval pipeline indefinitely (observed 24-min hang on a
    # multi-hop HyDE retry, 2026-04-26). 60s covers normal completions
    # comfortably while bounding worst-case latency.
    llm_timeout_sec: float = 60.0

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
