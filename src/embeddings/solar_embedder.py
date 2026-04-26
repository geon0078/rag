"""Solar embedding client (passage/query split, OpenAI-compatible)."""

from __future__ import annotations

import time
from typing import Iterable, Sequence

from openai import OpenAI, APIError, RateLimitError

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class SolarEmbedder:
    def __init__(self, mode: str = "passage") -> None:
        if mode not in {"passage", "query"}:
            raise ValueError(f"mode must be 'passage' or 'query', got {mode!r}")
        self.mode = mode
        self.model = (
            settings.embedding_model_passage
            if mode == "passage"
            else settings.embedding_model_query
        )
        self.client = OpenAI(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        attempt = 0
        delay = 1.0
        while True:
            attempt += 1
            try:
                resp = self.client.embeddings.create(model=self.model, input=list(texts))
                return [d.embedding for d in resp.data]
            except (RateLimitError, APIError) as exc:
                if attempt >= settings.embed_retry_max:
                    log.error(f"embed failed after {attempt} attempts: {exc}")
                    raise
                log.warning(f"embed retry {attempt}/{settings.embed_retry_max} after {delay:.1f}s: {exc}")
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)

    def embed_batched(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        progress: bool = True,
    ) -> list[list[float]]:
        bs = batch_size or settings.embed_batch_size
        out: list[list[float]] = []
        total = len(texts)
        if progress:
            from tqdm import tqdm
            iterator: Iterable[int] = tqdm(range(0, total, bs), desc=f"embed[{self.mode}]")
        else:
            iterator = range(0, total, bs)
        for start in iterator:
            chunk = texts[start : start + bs]
            out.extend(self.embed(chunk))
        if len(out) != total:
            raise RuntimeError(f"embedding count mismatch: got {len(out)}, expected {total}")
        return out
