"""RAGAS supplementary evaluation for the EulJi RAG pipeline.

Loads data/qa.parquet, runs RagPipeline.run() on every non-negative QA,
then evaluates with RAGAS (faithfulness, answer_relevancy, context_precision,
context_recall) using Solar (ChatUpstage / UpstageEmbeddings).

Outputs:
    reports/ragas_report.json         - per-row scores
    reports/ragas_summary.json        - aggregate metric averages
    reports/ragas_by_collection.csv   - per-collection metric breakdown

Run:
    python scripts/evaluate_ragas.py [--limit N] [--out-dir reports]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("evaluate_ragas")

QA_PATH = PROJECT_ROOT / "data" / "qa.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports"


async def _collect_samples(qa_df: pd.DataFrame, limit: int | None) -> list[dict]:
    pipeline = RagPipeline()
    samples: list[dict] = []
    rows = qa_df.head(limit) if limit else qa_df
    total = len(rows)
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        query = str(row["query"])
        try:
            result = await pipeline.run(query)
        except Exception as exc:
            log.error(f"[{idx}/{total}] pipeline error for {query!r}: {exc}")
            continue

        gen_gt = row["generation_gt"]
        if hasattr(gen_gt, "tolist"):
            gen_gt = gen_gt.tolist()
        ground_truth = gen_gt[0] if gen_gt else ""

        contexts = [c["contents"] for c in result.get("contexts", []) if c.get("contents")]
        if not contexts:
            log.warning(f"[{idx}/{total}] empty contexts for {query!r}; skipping")
            continue

        samples.append(
            {
                "qid": row["qid"],
                "question": query,
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": ground_truth,
                "collection": row.get("source_collection"),
                "qa_type": row.get("qa_type"),
            }
        )
        if idx % 10 == 0:
            log.info(f"  collected {idx}/{total}")
    return samples


def _run_ragas(samples: list[dict]) -> pd.DataFrame:
    from datasets import Dataset
    from langchain_upstage import ChatUpstage, UpstageEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    api_key = os.environ.get("UPSTAGE_API_KEY") or os.environ.get("SOLAR_API_KEY")
    if not api_key:
        raise RuntimeError("UPSTAGE_API_KEY or SOLAR_API_KEY must be set")

    llm = ChatUpstage(model="solar-pro3", api_key=api_key)
    embeddings = UpstageEmbeddings(
        model="solar-embedding-1-large-query", api_key=api_key
    )

    eval_rows = [
        {
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in samples
    ]
    dataset = Dataset.from_list(eval_rows)

    log.info(f"Running RAGAS on {len(eval_rows)} samples (this may take a while)...")
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    df = score.to_pandas()
    df["qid"] = [s["qid"] for s in samples]
    df["collection"] = [s["collection"] for s in samples]
    df["qa_type"] = [s["qa_type"] for s in samples]
    return df


def _write_outputs(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        c
        for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if c in df.columns
    ]
    summary = {col: float(df[col].mean()) for col in metric_cols}
    summary["n"] = int(len(df))

    df.to_json(out_dir / "ragas_report.json", orient="records", force_ascii=False)
    (out_dir / "ragas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "collection" in df.columns and metric_cols:
        per_col = df.groupby("collection")[metric_cols].mean().reset_index()
        per_col.to_csv(out_dir / "ragas_by_collection.csv", index=False, encoding="utf-8")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Cap samples for smoke test")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not QA_PATH.exists():
        log.error(f"qa.parquet missing: {QA_PATH}")
        return 1

    qa_df = pd.read_parquet(QA_PATH)
    if "qa_type" not in qa_df.columns:
        log.error("qa.parquet has no qa_type column. Run scripts/finalize_qa.py first.")
        return 2

    qa_df = qa_df[qa_df["qa_type"] != "negative"].reset_index(drop=True)
    log.info(f"Evaluating {len(qa_df)} non-negative QAs")

    samples = asyncio.run(_collect_samples(qa_df, args.limit))
    if not samples:
        log.error("No samples collected; aborting")
        return 3
    log.info(f"Collected {len(samples)} samples for RAGAS")

    df = _run_ragas(samples)
    summary = _write_outputs(df, args.out_dir)

    log.info("RAGAS aggregate scores:")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
