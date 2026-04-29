"""Gradio demo for the EulJi RAG pipeline.

Wraps `src.pipeline.RagPipeline.run()` in a chat-style UI exposing the answer,
groundedness verdict, retry flag, resolved campus, retrieval sources, and
latency. Useful for ad-hoc QA, judging fallback behaviour, and showing the
current pipeline configuration alongside the response.

Run:
    python scripts/gradio_app.py [--host 0.0.0.0] [--port 7860]

Requires UPSTAGE_API_KEY in .env and a running Qdrant on settings.qdrant_url.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

from src.config import settings  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("gradio_app")

_pipeline: RagPipeline | None = None


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        log.info("initializing RagPipeline (first request)")
        _pipeline = RagPipeline()
    return _pipeline


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_검색된 소스 없음_"
    lines = ["| # | doc_id | category | campus | score |", "|---|---|---|---|---|"]
    for i, s in enumerate(sources, 1):
        score = s.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        lines.append(
            f"| {i} | `{s.get('doc_id', '?')}` | {s.get('category', '—')} "
            f"| {s.get('campus', '—')} | {score_str} |"
        )
    return "\n".join(lines)


def _format_status(result: dict) -> str:
    verdict = result.get("verdict", "—")
    grounded = result.get("grounded", False)
    retry = result.get("retry", False)
    campus = result.get("resolved_campus") or "—"
    inferred = result.get("campus_was_inferred", False)
    elapsed = result.get("elapsed_ms", 0)

    grounded_badge = "✅ grounded" if grounded else "⚠️ notGrounded"
    retry_badge = " 🔁 HyDE retry" if retry else ""
    campus_badge = f"{campus}" + (" (inferred)" if inferred else "")
    return (
        f"**상태**: {grounded_badge} · verdict=`{verdict}`{retry_badge}  \n"
        f"**캠퍼스**: {campus_badge} · **응답시간**: {elapsed} ms"
    )


def _config_md() -> str:
    return (
        f"- LLM: `{settings.llm_model_pro}` (temp={settings.llm_temperature}) · "
        f"임베딩: `{settings.embedding_model_query}`  \n"
        f"- Hybrid: `{settings.hybrid_method}` "
        f"({settings.hybrid_cc_normalize}, w={settings.hybrid_cc_weight}) · "
        f"reranker: {'on' if settings.reranker_enabled else 'off'}  \n"
        f"- top_k: dense={settings.top_k_dense} sparse={settings.top_k_sparse} "
        f"final={settings.top_k_rerank_final} · default_campus=`{settings.default_campus}`"
    )


async def ask(query: str) -> tuple[str, str, str]:
    """Gradio 6 supports async event handlers natively — no asyncio.run wrap."""
    if not query or not query.strip():
        return "_질문을 입력하세요._", "", ""
    pipeline = _get_pipeline()
    try:
        result = await pipeline.run(query.strip())
    except Exception as exc:
        log.exception("pipeline error")
        return f"⚠️ 파이프라인 오류: `{type(exc).__name__}: {exc}`", "", ""
    return result["answer"], _format_status(result), _format_sources(result["sources"])


def build_app() -> gr.Blocks:
    # Gradio 6: theme moves to launch(); gr.Markdown drops `label=`.
    with gr.Blocks(title="EulJi RAG Demo") as app:
        gr.Markdown("# EulJi RAG Demo\n을지대학교 학사 안내 RAG 파이프라인 데모")
        gr.Markdown(_config_md())

        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(
                    label="질문",
                    placeholder="예: 휴학 신청 기간은 언제인가요?",
                    lines=2,
                )
                with gr.Row():
                    submit = gr.Button("질문하기", variant="primary")
                    clear = gr.Button("지우기")
                gr.Examples(
                    examples=[
                        "위탁생이 되려면 어떤 조건이 필요한가요?",
                        "학사학위취득유예는 언제 신청할 수 있고, 어떤 방법으로 신청하나요?",
                        "학칙 제29조 제1항에 따르면 학생이 한 학기에 최대 몇 학점까지 수강할 수 있나요?",
                        "을지대학교 성남캠퍼스에 셔틀버스가 있나요?",
                        "기숙사 입사 시 휠체어 이용 가능한 건물은 어디인가요?",
                        "교직과목은 어떻게 이수하나요?",
                        "전체 교수회와 캠퍼스 교수회를 소집하는 사람은 누구인가요?",
                        "학점은행제로 학사학위를 받으려면 본교에서 취득해야 하는 최소 학점은 얼마인가요?",
                    ],
                    inputs=query,
                )
            with gr.Column(scale=3):
                gr.Markdown("### 답변")
                answer = gr.Markdown(value="_아직 질문이 없습니다._")
                gr.Markdown("### 상태")
                status = gr.Markdown(value="")

        gr.Markdown("### 검색된 소스")
        sources = gr.Markdown(value="_아직 질문이 없습니다._")

        submit.click(ask, inputs=query, outputs=[answer, status, sources])
        query.submit(ask, inputs=query, outputs=[answer, status, sources])
        clear.click(
            lambda: ("", "", "_아직 질문이 없습니다._", ""),
            outputs=[answer, status, sources, query],
        )

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = p.parse_args()

    if not settings.upstage_api_key:
        raise SystemExit("UPSTAGE_API_KEY missing — set it in .env")

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
