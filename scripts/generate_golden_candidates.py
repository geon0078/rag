"""Golden Set 후보 자동 생성 (평가명세서 §10.4).

corpus 청크 100개를 카테고리 비중에 맞게 샘플링하고, 각 청크에서 Solar에게
**서로 다른 표현 스타일** 5개의 학생 자연 질문을 생성하도록 지시한다. 결과
500건 후보를 1차 후처리(중복/너무 짧음 제거)한 뒤 150건으로 압축, 사람이
100건을 큐레이션할 수 있는 baseline 셋을 만든다.

평가명세서 §3.1 / §10 의 7개 패턴 균형을 prompt에서 명시:
  P1 일반 (반말·구어), P2 정식, P3 축약, P4 오타·비정형, P5 multi-hop,
  P6 캠퍼스 명시, P7 negative

P7(정답 없는 질문)은 corpus와 무관한 외부 질문이므로 별도 큐레이션 권장
(scripts/finalize_qa.py 의 NEGATIVE_QUERIES 참고).

Run:
    python scripts/generate_golden_candidates.py
        [--per-chunk 5]
        [--output data/golden_candidates_v1.parquet]
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
TEMPERATURE = 0.85
CONCURRENCY = 2
RETRY_MAX = 5
RETRY_BASE = 2.0
MAX_DOC_CHARS = 1500


# 평가명세서 §10.3 — 컬렉션별 100건 기준 분포 (사람 큐레이션 시 참고).
# 청크 샘플링은 이 비율 × 1.5배로 (150 청크 → 450 candidate → 150 stratified).
COLLECTION_QUOTA: dict[str, int] = {
    "학칙_조항": 15,
    "FAQ": 20,
    "학사정보": 15,
    "학사일정": 10,
    "강의평가": 15,
    "시설_연락처": 10,
    "장학금": 5,
    "학과정보": 3,
    "교육과정": 3,
    "기타": 4,
}
SAMPLE_MULTIPLIER = 1.5


GOLDEN_SYSTEM = """당신은 한국 대학생이 학교 챗봇에 던지는 **자연스러운 질문**을 만드는 전문가입니다.

핵심 원칙: **모든 질문은 반드시 [청크]의 구체 정보를 정확히 묻는 형태여야 한다.**
일반론·추상적·corpus 무관 질문 금지. 각 query에 대응하는 generation_gt는 정확히 그 [청크]에서 추출한 답이어야 한다.

작성 규칙:
1. 주어진 [청크]가 답할 수 있는 핵심 정보(절차·요건·기간·기관·금액·조건 등) 중 하나를 골라 질문 1개 생성.
2. **paraphrase 금지** — [청크] 본문의 단어를 그대로 쓰지 말고 학생의 자연 표현으로 다시 표현.
3. 3개 질문은 **서로 다른 스타일**:
   - 1개 polite: 정중·정식 ("...을 알려주세요" / "...은 어떻게 되나요?")
   - 1개 casual: 반말·구어 ("...뭐야?" / "...어떻게 해?")
   - 1개 short: 매우 짧은 축약 ("졸업학점?", "휴학 신청?")
4. 3개 모두 [청크]에서 답을 찾을 수 있는 동일 주제. 단지 표현 스타일만 다름.
5. 답변(generation_gt)은 [청크] 정보를 그대로 사용해 1~3 문장.
6. 출처 표기 금지.

검증: 생성한 query를 보고, 답을 [청크] 외부에서 찾아야 한다면 그 query는 만들지 마세요.

반드시 다음 JSON 한 줄만 출력하세요. 다른 텍스트, 마크다운, 코드펜스 금지:
{"queries": [{"query": "...", "style": "polite|casual|short", "generation_gt": "..."}, ...]}"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _truncate(text: str, n: int = MAX_DOC_CHARS) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "..."


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "queries" not in obj:
        return None
    qs = obj["queries"]
    if not isinstance(qs, list):
        return None
    out = []
    for q in qs:
        if not isinstance(q, dict):
            continue
        query = str(q.get("query", "")).strip()
        gt = str(q.get("generation_gt", "")).strip()
        style = str(q.get("style", "")).strip()
        if not query or not gt:
            continue
        out.append({"query": query, "style": style, "generation_gt": gt})
    return {"queries": out}


class GoldenGenerator:
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
                        max_tokens=1000,
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

    async def generate(self, doc_id: str, contents: str, sc: str) -> list[dict[str, Any]]:
        user = f"[청크]\n{_truncate(contents)}\n\n위 청크로 답할 수 있는 학생 자연 질문 5개를 JSON으로 만들어 주세요."
        try:
            raw = await self._chat(GOLDEN_SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"generate error {doc_id}: {exc}")
            return []
        parsed = _extract_json(raw)
        if not parsed or not parsed.get("queries"):
            log.warning(f"JSON parse failed for {doc_id}: {raw[:120]!r}")
            return []
        out = []
        for q in parsed["queries"]:
            out.append({
                "qid": str(uuid.uuid4()),
                "query": q["query"],
                "style": q["style"],
                "expected_doc_ids": [doc_id],
                "expected_answer": q["generation_gt"],
                "expected_grounded": True,
                "source_collection": sc,
                "intent_category": sc,
                "difficulty": "medium",
                "source": "solar_auto_v1",
            })
        return out


async def main_async(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    df = pd.read_parquet(args.corpus)
    df["sc"] = df["metadata"].apply(
        lambda m: m.get("source_collection") if isinstance(m, dict) else None
    )
    df = df[df["sc"].notna()].reset_index(drop=True)
    log.info(f"corpus: {len(df)} docs across {df['sc'].nunique()} collections")

    chunks_to_use: list[pd.Series] = []
    for sc, n in COLLECTION_QUOTA.items():
        pool = df[df["sc"] == sc]
        if len(pool) == 0:
            log.warning(f"empty pool for {sc}")
            continue
        sample_n = int(n * SAMPLE_MULTIPLIER)
        take = min(sample_n, len(pool))
        idx = rng.sample(range(len(pool)), take)
        for i in idx:
            chunks_to_use.append(pool.iloc[i])
    log.info(f"sampled {len(chunks_to_use)} chunks from corpus")

    gen = GoldenGenerator()
    tasks = [
        gen.generate(c["doc_id"], c["contents"], c["sc"])
        for c in chunks_to_use
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    rows: list[dict[str, Any]] = []
    for batch in results:
        rows.extend(batch)

    seen_q: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for r in rows:
        q = r["query"].strip()
        if len(q) < 5 or len(q) > 120:
            continue
        if q in seen_q:
            continue
        seen_q.add(q)
        cleaned.append(r)
    log.info(f"after dedup/length filter: {len(cleaned)} candidates (from {len(rows)})")

    if len(cleaned) > args.target:
        # Stratified sampling: 카테고리별 quota 비율을 유지하면서 target에 맞춤
        total_q = sum(COLLECTION_QUOTA.values())
        by_sc: dict[str, list[dict[str, Any]]] = {}
        for r in cleaned:
            by_sc.setdefault(r["source_collection"], []).append(r)
        out: list[dict[str, Any]] = []
        for sc, group in by_sc.items():
            cat_quota = COLLECTION_QUOTA.get(sc, 1)
            target_n = max(1, round(args.target * cat_quota / total_q))
            target_n = min(target_n, len(group))
            rng.shuffle(group)
            out.extend(group[:target_n])
        # 부족하면 모든 카테고리에서 균등하게 보충
        if len(out) < args.target:
            out_qids = {r["qid"] for r in out}
            remain = [r for r in cleaned if r["qid"] not in out_qids]
            rng.shuffle(remain)
            out.extend(remain[: args.target - len(out)])
        # 초과면 절단
        cleaned = out[: args.target]

    log.info(f"final candidate count: {len(cleaned)}")

    out_df = pd.DataFrame(cleaned)
    out_df["curated"] = False
    out_df["notes"] = ""

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    log.info(f"wrote {len(out_df)} candidates -> {args.output}")
    log.info(f"by collection:\n{out_df['source_collection'].value_counts()}")
    log.info(f"by style:\n{out_df['style'].value_counts()}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(settings.corpus_path))
    p.add_argument("--output", default=str(settings.data_dir / "golden_candidates_v1.parquet"))
    p.add_argument("--per-chunk", type=int, default=5)
    p.add_argument("--target", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
