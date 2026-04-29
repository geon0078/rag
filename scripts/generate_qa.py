"""Generate AutoRAG-format QA dataset from corpus.parquet using Solar LLM.

Quotas (from 명세서 §Phase 5 Task 5.1):
  학칙_조항: 30 / FAQ: 25 / 학사정보: 20 / 강의평가: 30
  시설_연락처: 15 / 장학금: 15 / 학사일정: 15
  학과정보: 5 / 교육과정: 5 / 기타: 5
Single-hop 80% + Multi-hop 20%.

Output: data/qa.parquet
Schema (AutoRAG):
  qid: str, query: str,
  retrieval_gt: list[list[str]]  (multi-hop has >1 inner lists),
  generation_gt: list[str]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI, RateLimitError, APIError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


COLLECTION_QUOTAS: dict[str, int] = {
    # 5x scale-up (2026-04-27) so a single QA worth ≈0.1pt instead of ≈0.6pt,
    # bringing the metric noise floor below the typical Solar API drift
    # (±1pt before, ±0.4pt expected after). Some categories have fewer corpus
    # docs than the quota — the sampler reuses docs with replacement once
    # exhausted, which is acceptable for query-side variation testing.
    "학칙_조항": 150,
    "FAQ": 125,
    "학사정보": 100,
    "강의평가": 150,
    "시설_연락처": 75,
    "장학금": 75,
    "학사일정": 75,
    "학과정보": 25,
    "교육과정": 25,
    "기타": 25,
}

MULTI_HOP_RATIO = 0.20
MAX_DOC_CHARS = 1500
QA_LLM_MODEL = "solar-pro3"
QA_LLM_TEMPERATURE = 0.6
# Concurrency lowered from 4 to 2 for the 5x scale-up: at 4 the burst hits
# Solar's per-second rate ceiling and ~30% of calls return 429. Pair this
# with the exponential-backoff retry loop in QAGenerator._chat below.
CONCURRENCY = 2
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 2.0


SINGLE_HOP_SYSTEM = """당신은 한국어 RAG 평가 데이터셋을 만드는 전문가입니다.
주어진 [문서] 하나만으로 답변 가능한 질문(single-hop)을 만들어야 합니다.

규칙:
1. 질문은 자연스러운 한국어 구어체 또는 학생이 실제로 물어볼 법한 표현으로 작성하세요.
2. 답변은 반드시 [문서] 안에 명시된 정보만 사용하세요. 추측 금지.
3. 출처 표기는 답변에 포함하지 마세요(평가 도구가 별도 처리).
4. 답변은 1~3 문장으로 간결하게.
5. 강의평가 도큐먼트라면 "학생 의견에 따르면" 등의 표현을 자연스럽게 사용하세요.

반드시 다음 JSON 한 줄만 출력하세요. 다른 텍스트, 마크다운, 코드펜스 금지:
{"query": "...", "generation_gt": "..."}"""


MULTI_HOP_SYSTEM = """당신은 한국어 RAG 평가 데이터셋을 만드는 전문가입니다.
서로 다른 두 [문서]를 모두 참고해야만 답변할 수 있는 질문(multi-hop)을 만들어야 합니다.

규칙:
1. 두 문서의 정보를 결합·비교·종합해야 답이 나오는 질문을 작성하세요.
2. 한 문서만으로 답이 나오는 질문은 만들지 마세요.
3. 답변은 두 문서 정보를 모두 활용하여 1~3 문장으로 작성하세요.
4. 출처 표기는 포함하지 마세요.

반드시 다음 JSON 한 줄만 출력하세요. 다른 텍스트, 마크다운, 코드펜스 금지:
{"query": "...", "generation_gt": "..."}"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _truncate(text: str, n: int = MAX_DOC_CHARS) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "..."


def _extract_json(raw: str) -> dict[str, str] | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "query" not in obj or "generation_gt" not in obj:
        return None
    return {"query": str(obj["query"]).strip(), "generation_gt": str(obj["generation_gt"]).strip()}


def _build_single_user(content: str) -> str:
    return f"[문서]\n{_truncate(content)}\n\n위 문서로 답할 수 있는 질문 1개와 정답을 JSON으로 만들어 주세요."


def _build_multi_user(content_a: str, content_b: str) -> str:
    return (
        f"[문서 A]\n{_truncate(content_a)}\n\n"
        f"[문서 B]\n{_truncate(content_b)}\n\n"
        "두 문서를 모두 참조해야 답이 나오는 질문 1개와 정답을 JSON으로 만들어 주세요."
    )


class QAGenerator:
    def __init__(self, model: str = QA_LLM_MODEL) -> None:
        self.model = model
        self.client = AsyncOpenAI(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
        )
        self._sem = asyncio.Semaphore(CONCURRENCY)

    async def _chat(self, system: str, user: str) -> str:
        async with self._sem:
            attempt = 0
            delay = RATE_LIMIT_BASE_DELAY
            while True:
                attempt += 1
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=QA_LLM_TEMPERATURE,
                        max_tokens=400,
                    )
                    return resp.choices[0].message.content or ""
                except RateLimitError as exc:
                    if attempt >= RATE_LIMIT_RETRIES:
                        log.error(f"chat: 429 retry exhausted after {attempt} attempts")
                        raise
                    log.warning(f"chat: 429 retry {attempt}/{RATE_LIMIT_RETRIES} sleeping {delay:.1f}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, 30.0)
                except APIError as exc:
                    # Other transient errors — retry once with the same backoff.
                    if attempt >= 2:
                        raise
                    log.warning(f"chat: transient APIError, retrying once: {exc}")
                    await asyncio.sleep(delay)

    async def single_hop(self, doc_id: str, content: str) -> dict[str, Any] | None:
        try:
            raw = await self._chat(SINGLE_HOP_SYSTEM, _build_single_user(content))
        except Exception as exc:  # noqa: BLE001
            log.warning(f"single-hop LLM error for {doc_id}: {exc}")
            return None
        parsed = _extract_json(raw)
        if not parsed:
            log.warning(f"single-hop JSON parse failed for {doc_id}: {raw[:120]!r}")
            return None
        return {
            "qid": str(uuid.uuid4()),
            "query": parsed["query"],
            "retrieval_gt": [[doc_id]],
            "generation_gt": [parsed["generation_gt"]],
        }

    async def multi_hop(
        self, doc_a_id: str, doc_a_content: str, doc_b_id: str, doc_b_content: str
    ) -> dict[str, Any] | None:
        try:
            raw = await self._chat(
                MULTI_HOP_SYSTEM, _build_multi_user(doc_a_content, doc_b_content)
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"multi-hop LLM error for {doc_a_id}+{doc_b_id}: {exc}")
            return None
        parsed = _extract_json(raw)
        if not parsed:
            log.warning(f"multi-hop JSON parse failed for {doc_a_id}+{doc_b_id}: {raw[:120]!r}")
            return None
        return {
            "qid": str(uuid.uuid4()),
            "query": parsed["query"],
            "retrieval_gt": [[doc_a_id, doc_b_id]],
            "generation_gt": [parsed["generation_gt"]],
        }


def _load_corpus(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["source_collection"] = df["metadata"].apply(
        lambda m: m.get("source_collection") if isinstance(m, dict) else None
    )
    return df


def _sample_for_collection(df: pd.DataFrame, collection: str, n: int, rng: random.Random) -> pd.DataFrame:
    pool = df[df["source_collection"] == collection]
    if len(pool) == 0:
        log.warning(f"empty pool for collection={collection!r}")
        return pool
    take = min(n, len(pool))
    indices = rng.sample(range(len(pool)), take)
    return pool.iloc[indices].reset_index(drop=True)


def _split_quotas(total: int, multi_ratio: float = MULTI_HOP_RATIO) -> tuple[int, int]:
    multi = max(0, round(total * multi_ratio))
    single = max(0, total - multi)
    return single, multi


async def _gather_collection(
    gen: QAGenerator,
    collection: str,
    quota: int,
    sample: pd.DataFrame,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(sample) == 0 or quota == 0:
        return []
    single_n, multi_n = _split_quotas(quota)
    rows: list[dict[str, Any]] = []

    single_pool = sample.iloc[:single_n]
    single_tasks = [
        gen.single_hop(row["doc_id"], row["contents"]) for _, row in single_pool.iterrows()
    ]
    multi_tasks: list[Any] = []
    available_for_multi = sample.iloc[single_n:].to_dict(orient="records")
    if len(available_for_multi) < multi_n * 2:
        available_for_multi = sample.to_dict(orient="records")
    rng.shuffle(available_for_multi)
    for i in range(multi_n):
        if 2 * i + 1 >= len(available_for_multi):
            break
        a = available_for_multi[2 * i]
        b = available_for_multi[2 * i + 1]
        multi_tasks.append(
            gen.multi_hop(a["doc_id"], a["contents"], b["doc_id"], b["contents"])
        )

    log.info(
        f"[{collection}] generating single={len(single_tasks)} multi={len(multi_tasks)}"
    )
    single_results = await asyncio.gather(*single_tasks, return_exceptions=False)
    multi_results = await asyncio.gather(*multi_tasks, return_exceptions=False)

    for r in single_results:
        if r:
            r["source_collection"] = collection
            r["hop_type"] = "single"
            rows.append(r)
    for r in multi_results:
        if r:
            r["source_collection"] = collection
            r["hop_type"] = "multi"
            rows.append(r)
    log.info(
        f"[{collection}] kept single={sum(1 for r in single_results if r)} multi={sum(1 for r in multi_results if r)}"
    )
    return rows


async def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    df = _load_corpus(Path(args.corpus))
    log.info(f"corpus loaded: {len(df)} rows from {args.corpus}")

    gen = QAGenerator()
    all_rows: list[dict[str, Any]] = []

    for collection, quota in COLLECTION_QUOTAS.items():
        sample = _sample_for_collection(df, collection, quota, rng)
        rows = await _gather_collection(gen, collection, quota, sample, rng)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("no QA generated; aborting")

    qa_df = pd.DataFrame(all_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qa_df.to_parquet(out_path, index=False)
    log.info(f"wrote {len(qa_df)} QA rows to {out_path}")

    summary = qa_df.groupby(["source_collection", "hop_type"]).size().unstack(fill_value=0)
    log.info(f"summary by collection:\n{summary}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(settings.corpus_path))
    p.add_argument("--output", default=str(settings.data_dir / "qa.parquet"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
