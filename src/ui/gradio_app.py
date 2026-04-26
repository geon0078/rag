"""Gradio web UI for live testing of the EulJi RAG pipeline.

Talks to the FastAPI server (default http://localhost:8000) over SSE so the
streaming behavior, semantic cache hit/miss, groundedness verdict, and source
attribution all reflect what production users will see.

Run with:
  python -m src.ui.gradio_app
  # or
  GRADIO_API_BASE=http://localhost:8000 python -m src.ui.gradio_app
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import gradio as gr
import httpx

DEFAULT_API_BASE = os.getenv("GRADIO_API_BASE", "http://localhost:8000")
EXAMPLES = [
    "을지대학교 2026학년도 1학기 개강일은 언제인가요?",
    "성남캠퍼스 학생식당 운영시간 알려주세요",
    "휴학 신청은 언제까지 해야 하나요?",
    "대전캠퍼스 도서관 위치가 어디인가요?",
    "전과 신청 자격 요건이 어떻게 되나요?",
]

VERDICT_BADGE = {
    "grounded": ("✅ grounded", "#16a34a"),
    "notSure": ("⚠️ notSure", "#ca8a04"),
    "notGrounded": ("❌ notGrounded", "#dc2626"),
}


def _format_meta(meta: dict) -> str:
    parts = []
    cached = meta.get("cached")
    if cached:
        sim = meta.get("similarity")
        sim_str = f" (cos={sim:.3f})" if sim is not None else ""
        parts.append(f"💾 캐시 히트{sim_str}")
    elapsed = meta.get("elapsed_ms")
    if elapsed is not None:
        parts.append(f"⏱ {elapsed} ms")
    verdict = meta.get("verdict")
    if verdict:
        label, color = VERDICT_BADGE.get(verdict, (verdict, "#475569"))
        parts.append(f'<span style="color:{color};font-weight:600">{label}</span>')
    return " · ".join(parts)


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_출처 없음_"
    rows = ["| # | 카테고리 | 캠퍼스 | 제목 | rerank |", "|---|---|---|---|---|"]
    for i, s in enumerate(sources, 1):
        title = (s.get("title") or s.get("doc_id") or "—")[:60]
        cat = s.get("category") or "—"
        campus = s.get("campus") or "—"
        score = s.get("rerank_score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        rows.append(f"| {i} | {cat} | {campus} | {title} | {score_str} |")
    return "\n".join(rows)


async def _stream_query(
    api_base: str, query: str, use_cache: bool
) -> AsyncIterator[tuple[str, str, str]]:
    """Yield (answer_so_far, status_html, sources_md) tuples from SSE stream."""
    url = f"{api_base.rstrip('/')}/api/v1/query/stream"
    payload = {"query": query, "use_cache": use_cache}
    answer = ""
    status = "🔍 검색 중..."
    sources_md = ""

    timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield (
                        "",
                        f'<span style="color:#dc2626">서버 오류 {resp.status_code}: {body.decode("utf-8", "replace")[:200]}</span>',
                        "",
                    )
                    return

                event_type = "message"
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        event_type = "message"
                        continue
                    if raw_line.startswith("event:"):
                        event_type = raw_line[6:].strip()
                        continue
                    if not raw_line.startswith("data:"):
                        continue
                    data = raw_line[5:].lstrip()

                    if event_type == "meta":
                        try:
                            meta = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if meta.get("cached"):
                            status = _format_meta({**meta, "verdict": "grounded"})
                        else:
                            status = (
                                f"🧠 후보 {meta.get('candidates', '?')}개로 답변 생성 중..."
                            )
                        yield answer, status, sources_md
                    elif event_type == "token":
                        answer += data
                        yield answer, status, sources_md
                    elif event_type == "done":
                        try:
                            done = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        status = _format_meta(done)
                        sources_md = _format_sources(done.get("sources", []))
                        yield answer, status, sources_md
                        return
    except httpx.ConnectError:
        yield (
            "",
            f'<span style="color:#dc2626">서버 연결 실패: {api_base} 에 FastAPI가 떠있나요? (`uvicorn src.api.main:app --port 8000`)</span>',
            "",
        )
    except httpx.ReadTimeout:
        yield answer, '<span style="color:#dc2626">응답 타임아웃 (120s)</span>', sources_md


async def _check_health(api_base: str) -> str:
    url = f"{api_base.rstrip('/')}/api/v1/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            data = resp.json()
        status = data.get("status", "unknown")
        components = data.get("components", {})
        redis = data.get("redis", False)
        bullets = [
            f"- redis: {'🟢' if redis else '🔴'}",
            *[
                f"- {name}: {'🟢' if ok else '🔴'}"
                for name, ok in components.items()
            ],
        ]
        return f"**status: `{status}`**\n" + "\n".join(bullets)
    except Exception as exc:
        return f'<span style="color:#dc2626">health 호출 실패: {exc}</span>'


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="을지대 RAG 테스트") as demo:
        gr.Markdown(
            "# 을지대학교 RAG 테스트\n"
            "Solar API + Qdrant + BM25(Okt) + bge-reranker · Self-Corrective + 시맨틱 캐시"
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="질문",
                    placeholder="예: 2026학년도 학사일정이 어떻게 되나요?",
                    lines=2,
                    autofocus=True,
                )
                with gr.Row():
                    submit_btn = gr.Button("질의", variant="primary")
                    clear_btn = gr.Button("초기화")
                use_cache = gr.Checkbox(label="시맨틱 캐시 사용", value=True)
                gr.Examples(examples=EXAMPLES, inputs=query_box, label="예시 질문")

            with gr.Column(scale=2):
                api_base = gr.Textbox(
                    label="API 서버",
                    value=DEFAULT_API_BASE,
                    info="FastAPI uvicorn 주소",
                )
                health_btn = gr.Button("헬스체크", size="sm")
                health_md = gr.Markdown("_헬스체크 미실행_")

        status_html = gr.HTML(label="상태")
        answer_md = gr.Markdown(label="답변", value="_질의를 입력하세요._")
        sources_md = gr.Markdown(label="출처")

        submit_btn.click(
            fn=_stream_query,
            inputs=[api_base, query_box, use_cache],
            outputs=[answer_md, status_html, sources_md],
        )
        query_box.submit(
            fn=_stream_query,
            inputs=[api_base, query_box, use_cache],
            outputs=[answer_md, status_html, sources_md],
        )
        clear_btn.click(
            fn=lambda: ("", "", "_초기화됨_", ""),
            inputs=[],
            outputs=[query_box, status_html, answer_md, sources_md],
        )
        health_btn.click(fn=_check_health, inputs=[api_base], outputs=[health_md])

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "0") == "1",
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
