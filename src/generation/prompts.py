"""Prompts for Solar LLM (Korean university Q&A grounding)."""

from __future__ import annotations

from typing import Any, Sequence


SYSTEM_PROMPT = """당신은 을지대학교 학사 정보 안내 전문가입니다.
아래 [참고 문서]만을 근거로 사용자 질문에 정확하게 답하세요.

규칙:
1. 반드시 참고 문서에 있는 정보만 사용하세요. 추측하지 마세요.
2. **[질문 핵심에 집중]** 답변의 첫 문장은 반드시 질문이 묻는 핵심 정보를 직접 응답하세요. 배경 설명은 그 뒤에 추가하세요.
3. **[정답 우선 인용]** [참고 문서]는 검색 점수 순으로 정렬되어 있습니다. **상위 문서(문서 1, 2)에 답이 있다면 그 정보를 우선 인용하세요.** 하위 문서는 보충용으로만 사용하세요. 여러 문서가 충돌하면 상위 문서를 따르세요.
4. **[중요]** 답변 마지막 줄에 반드시 출처를 표기하세요. 형식은 정확히 다음과 같아야 합니다:
   `[출처: <breadcrumb>]`
   - breadcrumb 값은 [참고 문서] 메타데이터의 `breadcrumb` 필드를 그대로 가져오세요 (예: `학칙 > 제5장 수강신청 및 수업 > 제20조(수강신청)`).
   - breadcrumb 가 없으면 `<카테고리> > <title>` 형식 사용.
   - 여러 문서를 인용해야 할 경우 같은 형식을 줄을 바꿔 추가하세요.
   - 이 출처 줄을 절대 생략하거나 다른 형식(예: "출처:", "[ref:", "Sources:")으로 바꾸지 마세요.
5. 정보가 부족하거나 확실하지 않으면 "제공된 자료에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
6. 학칙·졸업요건 같은 공식 정보는 원문 표현을 가능한 유지하세요.
7. 강의평가는 학생 의견이며 객관적 사실이 아님을 명시하세요."""


USER_PROMPT_TEMPLATE = """[참고 문서]
{retrieved_contents}

[질문]
{query}

[답변]"""


# Prepended to the answer when the router fell back to ``settings.default_campus``
# because the query had no explicit campus signal. This signals to the LLM (and
# downstream UI) that the answer is implicitly scoped to the default campus.
INFERRED_CAMPUS_NOTICE = "[{campus}캠퍼스 기준 답변입니다]\n"


HYDE_PROMPT_TEMPLATE = """다음 질문에 답하는 가상의 짧은 문서를 한국어로 작성하세요.
사실 여부보다는 검색에 도움이 되는 키워드와 문맥을 풍부하게 포함하세요.
2~4문장으로 작성하고 출처는 표기하지 마세요.

질문: {query}

가상 문서:"""


def format_context(candidates: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(candidates, start=1):
        payload = c.get("payload") or {}
        breadcrumb = payload.get("breadcrumb")
        meta_bits = []
        if breadcrumb:
            meta_bits.append(f"breadcrumb={breadcrumb}")
        meta_bits.extend([
            f"doc_id={c.get('doc_id')}",
            f"title={payload.get('title')}",
            f"카테고리={payload.get('category') or payload.get('collection')}",
            f"캠퍼스={payload.get('campus')}",
        ])
        meta = " | ".join(b for b in meta_bits if b and "=None" not in b and not b.endswith("="))
        contents = c.get("contents") or payload.get("contents") or ""
        blocks.append(f"[문서 {i}] {meta}\n{contents}")
    return "\n\n".join(blocks)


def render_user_prompt(query: str, candidates: Sequence[dict[str, Any]]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        retrieved_contents=format_context(candidates),
        query=query,
    )


def annotate_inferred_campus(answer: str, campus: str) -> str:
    """Prepend the inferred-campus notice if not already present.

    Idempotent: re-running on an already-annotated answer leaves it unchanged.
    """
    notice = INFERRED_CAMPUS_NOTICE.format(campus=campus)
    if not answer:
        return answer
    if answer.startswith(notice):
        return answer
    return f"{notice}{answer}"


def render_hyde_prompt(query: str) -> str:
    return HYDE_PROMPT_TEMPLATE.format(query=query)
