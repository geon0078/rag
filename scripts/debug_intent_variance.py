"""Variance check: run the production intent prompt N times against the
over-rejected set to confirm classifier non-determinism on borderline cases.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings  # noqa: E402
from src.generation.intent_classifier import _INTENT_SYSTEM_PROMPT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT / "data" / "qa.parquet"
EVAL = ROOT / "reports" / "eval_supplementary.json"
OUT = ROOT / "reports" / "intent_variance.json"
N_RUNS = 5


async def classify(client: AsyncOpenAI, query: str) -> str:
    resp = await client.chat.completions.create(
        model="solar-pro3",
        messages=[
            {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"[질문]\n{query}\n\n[판정]"},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    if "unanswerable" in text:
        return "unanswerable"
    if "answerable" in text:
        return "answerable"
    return "answerable"


async def main() -> None:
    df = pd.read_parquet(QA)
    report = json.load(open(EVAL, encoding="utf-8"))
    qids = [f["qid"] for f in report["routing"]["failures"] if not f["got_top3"]]
    rows = df[df["qid"].isin(qids)][["qid", "query"]].to_dict(orient="records")

    client = AsyncOpenAI(
        api_key=settings.upstage_api_key,
        base_url=settings.upstage_base_url,
    )
    sem = asyncio.Semaphore(8)

    async def one(query: str) -> str:
        async with sem:
            return await classify(client, query)

    per_query: dict[str, list[str]] = {r["qid"]: [] for r in rows}
    for run in range(N_RUNS):
        verdicts = await asyncio.gather(*[one(r["query"]) for r in rows])
        for r, v in zip(rows, verdicts):
            per_query[r["qid"]].append(v)
        print(f"run {run+1}/{N_RUNS} done")

    lookup = {r["qid"]: r["query"] for r in rows}
    summary = []
    for qid, verdicts in per_query.items():
        c = Counter(verdicts)
        ans = c.get("answerable", 0)
        unans = c.get("unanswerable", 0)
        flap = "FLAP" if (ans > 0 and unans > 0) else "stable"
        summary.append(
            {
                "qid": qid,
                "query": lookup[qid][:80],
                "ans": ans,
                "unans": unans,
                "flap": flap,
            }
        )
    summary.sort(key=lambda x: x["unans"], reverse=True)

    OUT.write_text(
        json.dumps(
            {"n_runs": N_RUNS, "per_query": summary}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    flap_count = sum(1 for s in summary if s["flap"] == "FLAP")
    always_unans = sum(1 for s in summary if s["unans"] == N_RUNS)
    always_ans = sum(1 for s in summary if s["ans"] == N_RUNS)
    print(f"\n=== variance over {N_RUNS} runs ({len(summary)} queries) ===")
    print(f"always answerable:   {always_ans}")
    print(f"always unanswerable: {always_unans}")
    print(f"flapping:            {flap_count}")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
