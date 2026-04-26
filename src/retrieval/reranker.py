"""Cross-encoder reranking with `dragonkue/bge-reranker-v2-m3-ko`.

Includes a `PassthroughReranker` no-op alternative for CPU-only production
where the cross-encoder is too slow (see settings.reranker_enabled).
"""

from __future__ import annotations

from typing import Any, Sequence

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class PassthroughReranker:
    """No-op reranker: keeps hybrid score order, slices to top_k.

    Used when `settings.reranker_enabled` is False (default on 2-core CPU
    deployments where bge-reranker latency is prohibitive).
    """

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int | None = None,
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        k = top_k or settings.top_k_rerank_final
        out: list[dict[str, Any]] = []
        for cand in list(candidates)[:k]:
            item = dict(cand)
            item.setdefault("rerank_score", item.get("rrf_score", 0.0))
            out.append(item)
        log.info(f"passthrough: {len(candidates)} -> {len(out)}")
        return out


class KoReranker:
    _model = None  # class-level cache to avoid reloading 1.1GB weights

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.device = device or self._auto_device()
        self._ensure_model()

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model(self) -> None:
        if KoReranker._model is not None:
            return
        from sentence_transformers import CrossEncoder
        import torch

        log.info(f"loading reranker {self.model_name!r} on {self.device}")
        KoReranker._model = CrossEncoder(
            self.model_name,
            device=self.device,
            default_activation_function=torch.nn.Sigmoid(),
        )
        log.info("reranker ready")

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int | None = None,
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        k = top_k or settings.top_k_rerank_final
        pairs = [(query, c.get("contents", "")) for c in candidates]
        scores = KoReranker._model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )

        scored = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        log.info(f"rerank: {len(candidates)} -> {min(k, len(scored))}")
        return scored[:k]
