"""Adversarial QA generator — 5 corpus-grounded challenge types.

Existing data/qa.parquet is dominated by ``paraphrase-of-doc`` queries because
generate_qa.py asks the LLM to write a question for a single doc. The eval
scores look great on that distribution but the live Gradio answers feel weak
— the eval-reality gap.

This script asks Solar to produce queries the corpus *can* answer but in five
distinctly harder shapes:

  T1 conversational  — short colloquial student question
  T2 vague           — under-specified but answerable from corpus
  T3 paraphrase      — corpus terms swapped for synonyms / round-about wording
  T4 multi_intent    — single query bundling two corpus facts (multi-hop)
  T5 inference       — answer requires a small inference over corpus content

The retrieval_gt is the source doc(s), so downstream eval / judge can verify
whether the pipeline actually reaches the same source.

Run:
    python scripts/generate_adversarial_qa.py
        [--per-type 50]
        [--seed 42]
        [--output data/qa_adversarial.parquet]
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
from openai import APIError, AsyncOpenAI, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

QA_LLM_MODEL = "solar-pro3"
TEMPERATURE = 0.7
CONCURRENCY = 2
RETRY_MAX = 5
RETRY_BASE = 2.0
MAX_DOC_CHARS = 1500


CHALLENGE_PROMPTS: dict[str, str] = {
    "T1_conversational": """당신은 한국 대학생이 학교 챗봇에 질문하는 짧고 구어체적인 표현을 만드는 전문가입니다.

규칙:
1. 주어진 [문서]가 답할 수 있는 정보를 바탕으로, 학생이 실제로 던질 법한 매우 짧은 (10~25자) 구어체 질문을 작성하세요.
2. "휴학 어떻게 해?" "수강신청 언제부터야?" 같이 자연스러운 한국어로.
3. 답변(generation_gt)은 [문서] 정보를 그대로 사용해 1~3 문장.
4. JSON 한 줄만 출력: {"query": "...", "generation_gt": "..."}""",

    "T2_vague": """당신은 학생이 챗봇에 던지는 "모호하지만 corpus가 답할 수 있는" 질문을 만드는 전문가입니다.

규칙:
1. 주어진 [문서] 내용을 답으로 가지지만, 질문에는 핵심 키워드를 부분적으로만 노출하세요.
2. 예: 문서가 "수강신청 절차"를 다루면 "그거 어떻게 해야 돼?" 가 아니라 "수강 신청 절차 좀" 정도로 모호하게.
3. 너무 광범위하지 않게 — 문서 주제의 한 측면을 모호하게 묻기.
4. 답변(generation_gt)은 [문서] 정보를 그대로 사용해 1~3 문장.
5. JSON 한 줄만 출력: {"query": "...", "generation_gt": "..."}""",

    "T3_paraphrase": """당신은 corpus의 표현을 동의어/우회 표현으로 바꿔 묻는 질문을 만드는 전문가입니다.

규칙:
1. [문서]의 핵심 어휘(예: "휴학", "장학금", "교수회") 중 1~2개를 같은 의미의 다른 표현으로 바꿔 질문하세요.
2. 단, 의미가 흐려지면 안 됨 — corpus를 검색하면 같은 doc이 나와야 함.
3. 예: "교수회" → "교수 회의", "위탁생" → "외부 위탁 학생"
4. 답변(generation_gt)은 [문서] 정보를 그대로 사용해 1~3 문장.
5. JSON 한 줄만 출력: {"query": "...", "generation_gt": "..."}""",

    "T4_multi_intent": """당신은 두 [문서]에서 답을 찾아야 하는 복합 질문을 만드는 전문가입니다.

규칙:
1. 한 질문 안에 [문서 A]와 [문서 B] 양쪽 정보가 모두 필요한 형태로.
2. "X 하면서 Y도 하려면?", "A와 B 둘 다 알려줘" 같은 자연스러운 다중 의도.
3. 답변은 두 doc 정보를 모두 결합해 1~3 문장.
4. JSON 한 줄만 출력: {"query": "...", "generation_gt": "..."}""",

    "T5_inference": """당신은 corpus 정보로 추론·재구성해야 답할 수 있는 질문을 만드는 전문가입니다.

규칙:
1. [문서]에 정보는 있지만 "그대로 직접 발췌"하는 답이 아니라, 약간의 재해석/적용이 필요한 질문.
2. 예: 문서가 "휴학 신청 기간 4주" 라면 "내가 개강 5주차인데 휴학 가능해?" 같은 적용 질문.
3. 단, 답은 반드시 [문서] 정보로 도출 가능해야 함.
4. JSON 한 줄만 출력: {"query": "...", "generation_gt": "..."}""",
}


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
    if not isinstance(obj, dict) or "query" not in obj or "generation_gt" not in obj:
        return None
    return {
        "query": str(obj["query"]).strip(),
        "generation_gt": str(obj["generation_gt"]).strip(),
    }


class AdversarialGenerator:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.upstage_api_key, base_url=settings.upstage_base_url
        )
        self._sem = asyncio.Semaphore(CONCURRENCY)

    async def _chat(self, system: str, user: str) -> str:
        async with self._sem:
            attempt = 0
            delay = RETRY_BASE
            while True:
                attempt += 1
                try:
                    resp = await self.client.chat.completions.create(
                        model=QA_LLM_MODEL,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=400,
                    )
                    return resp.choices[0].message.content or ""
                except RateLimitError:
                    if attempt >= RETRY_MAX:
                        raise
                    log.warning(f"chat: 429 retry {attempt}/{RETRY_MAX} sleep {delay:.1f}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, 30.0)
                except APIError:
                    if attempt >= 2:
                        raise
                    await asyncio.sleep(delay)

    async def generate_single(self, ctype: str, doc_id: str, contents: str) -> dict[str, Any] | None:
        system = CHALLENGE_PROMPTS[ctype]
        user = f"[문서]\n{_truncate(contents)}\n\n위 문서로 답할 수 있는 질문 하나와 정답을 JSON으로 만들어 주세요."
        try:
            raw = await self._chat(system, user)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"{ctype} error for {doc_id}: {exc}")
            return None
        parsed = _extract_json(raw)
        if not parsed:
            log.warning(f"{ctype} JSON parse failed for {doc_id}: {raw[:120]!r}")
            return None
        return {
            "qid": str(uuid.uuid4()),
            "query": parsed["query"],
            "retrieval_gt": [[doc_id]],
            "generation_gt": [parsed["generation_gt"]],
            "challenge_type": ctype,
            "hop_type": "single",
        }

    async def generate_multi(self, ctype: str, a_id: str, a_content: str, b_id: str, b_content: str) -> dict[str, Any] | None:
        system = CHALLENGE_PROMPTS[ctype]
        user = (
            f"[문서 A]\n{_truncate(a_content)}\n\n"
            f"[문서 B]\n{_truncate(b_content)}\n\n"
            "두 문서 모두 참조해야 답이 나오는 질문 하나와 정답을 JSON으로 만들어 주세요."
        )
        try:
            raw = await self._chat(system, user)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"{ctype} error for {a_id}+{b_id}: {exc}")
            return None
        parsed = _extract_json(raw)
        if not parsed:
            return None
        return {
            "qid": str(uuid.uuid4()),
            "query": parsed["query"],
            "retrieval_gt": [[a_id, b_id]],
            "generation_gt": [parsed["generation_gt"]],
            "challenge_type": ctype,
            "hop_type": "multi",
        }


async def main_async(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    df = pd.read_parquet(args.corpus)
    df["sc"] = df["metadata"].apply(
        lambda m: m.get("source_collection") if isinstance(m, dict) else None
    )
    df = df[df["sc"].notna()].reset_index(drop=True)
    log.info(f"corpus: {len(df)} docs across {df['sc'].nunique()} collections")

    gen = AdversarialGenerator()
    rows: list[dict[str, Any]] = []
    n_per = args.per_type

    # Single-doc types: T1, T2, T3, T5
    for ctype in ["T1_conversational", "T2_vague", "T3_paraphrase", "T5_inference"]:
        sample_idx = rng.sample(range(len(df)), min(n_per, len(df)))
        log.info(f"{ctype}: generating {len(sample_idx)} ...")
        tasks = [
            gen.generate_single(ctype, df.iloc[i]["doc_id"], df.iloc[i]["contents"])
            for i in sample_idx
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        kept = 0
        for r, i in zip(results, sample_idx):
            if r:
                r["source_collection"] = df.iloc[i]["sc"]
                rows.append(r)
                kept += 1
        log.info(f"{ctype}: kept {kept}/{len(sample_idx)}")

    # Multi-doc type: T4 — sample pairs within each collection
    pairs: list[tuple[Any, Any, str]] = []
    per_collection = max(1, n_per // df["sc"].nunique())
    for sc in df["sc"].unique():
        pool = df[df["sc"] == sc]
        if len(pool) < 2:
            continue
        for _ in range(per_collection):
            i, j = rng.sample(range(len(pool)), 2)
            pairs.append((pool.iloc[i], pool.iloc[j], sc))
    pairs = pairs[:n_per]
    log.info(f"T4_multi_intent: generating {len(pairs)} ...")
    tasks = [
        gen.generate_multi("T4_multi_intent", a["doc_id"], a["contents"], b["doc_id"], b["contents"])
        for a, b, _ in pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    kept = 0
    for r, (_, _, sc) in zip(results, pairs):
        if r:
            r["source_collection"] = sc
            rows.append(r)
            kept += 1
    log.info(f"T4_multi_intent: kept {kept}/{len(pairs)}")

    if not rows:
        log.error("no QA generated; aborting")
        return 1

    out_df = pd.DataFrame(rows)
    out_df["qa_type"] = out_df["hop_type"].map({"single": "single_hop", "multi": "multi_hop"})
    out_df["metadata"] = [{"campus_filter": None}] * len(out_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    log.info(f"wrote {len(out_df)} adversarial QA to {out_path}")
    log.info(f"summary by challenge_type:\n{out_df['challenge_type'].value_counts()}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(settings.corpus_path))
    p.add_argument("--output", default=str(settings.data_dir / "qa_adversarial.parquet"))
    p.add_argument("--per-type", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
