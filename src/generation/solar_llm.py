"""Solar LLM wrapper (Upstage, OpenAI-compatible). Async + streaming."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from src.config import settings
from src.generation.prompts import (
    SYSTEM_PROMPT,
    render_hyde_prompt,
    render_user_prompt,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class SolarLLM:
    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        self.model = model or settings.llm_model_pro
        self.temperature = settings.llm_temperature if temperature is None else temperature
        # Per-request timeout bounds worst-case latency. Without it, a single
        # hung HyDE retry stalled the eval pipeline for 24+ min (2026-04-26).
        self.client = AsyncOpenAI(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
            timeout=settings.llm_timeout_sec,
        )

    async def _chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        max_tokens: int | None = None,
    ):
        attempt = 0
        delay = 1.0
        while True:
            attempt += 1
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    stream=stream,
                )
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                return await self.client.chat.completions.create(**kwargs)
            except (RateLimitError, APITimeoutError, APIError) as exc:
                if attempt >= settings.embed_retry_max:
                    log.error(f"chat failed after {attempt} attempts: {exc}")
                    raise
                log.warning(f"chat retry {attempt} after {delay:.1f}s: {exc}")
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)

    async def generate(
        self,
        query: str,
        candidates: list[dict],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": render_user_prompt(query, candidates)},
        ]
        resp = await self._chat(messages, stream=False)
        content = resp.choices[0].message.content or ""
        return content.strip()

    async def stream(
        self,
        query: str,
        candidates: list[dict],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": render_user_prompt(query, candidates)},
        ]
        stream_resp = await self._chat(messages, stream=True)
        async for chunk in stream_resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def hyde_expand(self, query: str, max_tokens: int = 200) -> str:
        messages = [
            {"role": "system", "content": "당신은 한국어 문서 작성 도우미입니다."},
            {"role": "user", "content": render_hyde_prompt(query)},
        ]
        resp = await self._chat(messages, stream=False, max_tokens=max_tokens)
        return (resp.choices[0].message.content or "").strip()
