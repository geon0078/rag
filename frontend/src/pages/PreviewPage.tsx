import { useState } from "react";
import { Link } from "react-router-dom";
import {
  usePreviewSearch,
  usePreviewAnswer,
  type PreviewSearchResp,
  type PreviewAnswerResp,
} from "@/api/preview";

// 운영웹통합명세서 §6.6 / §10.2 — RAG 미리보기 페이지.
// HyDE doc + retrieval candidates + 답변 + Groundedness 한 화면.
export default function PreviewPage() {
  const [query, setQuery] = useState("");
  const [searchResult, setSearchResult] = useState<PreviewSearchResp | null>(null);
  const [answerResult, setAnswerResult] = useState<PreviewAnswerResp | null>(null);

  const search = usePreviewSearch();
  const answer = usePreviewAnswer();

  const onSearchOnly = async () => {
    const r = await search.mutateAsync({ query, top_k: 5 });
    setSearchResult(r);
  };
  const onAnswer = async () => {
    const r = await answer.mutateAsync({ query, top_k: 5 });
    setAnswerResult(r);
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold text-zinc-900">RAG 미리보기</h1>
      <p className="text-sm text-zinc-500">
        운영자 검수용 — HyDE 가상 답변 + retrieval + 답변 + Groundedness 결과를 한 번에.
      </p>

      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="졸업학점은 몇 학점이에요?"
          className="flex-1 rounded border border-zinc-300 bg-white px-3 py-2 text-sm"
          onKeyDown={(e) => e.key === "Enter" && query.trim() && onAnswer()}
        />
        <button
          type="button"
          onClick={onSearchOnly}
          disabled={!query.trim() || search.isPending}
          className="rounded border border-zinc-300 bg-white px-3 py-2 text-sm hover:bg-zinc-50 disabled:opacity-50"
        >
          🔍 검색만
        </button>
        <button
          type="button"
          onClick={onAnswer}
          disabled={!query.trim() || answer.isPending}
          className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
        >
          💬 답변까지
        </button>
      </div>

      {(search.isPending || answer.isPending) && (
        <p className="text-sm text-zinc-500">실행 중…</p>
      )}

      {searchResult && (
        <section className="rounded border border-zinc-200 bg-white p-4">
          <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">
            HyDE 가상 답변
          </h2>
          <p className="mt-1 text-sm text-zinc-700">
            {searchResult.hyde_doc ?? "(생성 실패)"}
          </p>
          <h2 className="mt-4 text-sm font-medium uppercase tracking-wide text-zinc-500">
            검색된 청크 (top {searchResult.candidates.length})
          </h2>
          <ul className="mt-2 space-y-2">
            {searchResult.candidates.map((c, i) => (
              <li key={c.doc_id} className="rounded border border-zinc-200 p-2">
                <div className="flex justify-between text-xs text-zinc-500">
                  <Link
                    to={`/chunks/${encodeURIComponent(c.doc_id)}`}
                    className="text-blue-700 hover:underline"
                  >
                    {i + 1}. {c.doc_id}
                  </Link>
                  <span>
                    score{" "}
                    {typeof c.score === "number" ? c.score.toFixed(3) : "—"}
                  </span>
                </div>
                <div className="mt-1 text-sm font-medium text-zinc-900">{c.title ?? "(제목 없음)"}</div>
                <div className="text-xs text-zinc-500">
                  {c.category} · {c.campus}
                </div>
                <p className="mt-1 text-sm text-zinc-700">{c.snippet}</p>
              </li>
            ))}
          </ul>
        </section>
      )}

      {answerResult && (
        <section className="rounded border border-zinc-200 bg-white p-4">
          <div className="flex items-center gap-2">
            <span
              className={
                "rounded px-2 py-0.5 text-xs " +
                (answerResult.grounded
                  ? "bg-emerald-100 text-emerald-700"
                  : "bg-red-100 text-red-700")
              }
            >
              {answerResult.grounded ? "✅ grounded" : "⚠️ notGrounded"}
            </span>
            <span className="text-xs text-zinc-500">verdict={answerResult.verdict}</span>
            {answerResult.retry && (
              <span className="text-xs text-amber-700">🔁 HyDE retry</span>
            )}
            <span className="ml-auto text-xs text-zinc-500">
              {answerResult.elapsed_ms} ms
            </span>
          </div>
          <h2 className="mt-3 text-sm font-medium uppercase tracking-wide text-zinc-500">
            답변
          </h2>
          <pre className="mt-1 whitespace-pre-wrap rounded bg-zinc-50 p-2 text-sm text-zinc-900">
            {answerResult.answer}
          </pre>
          <h2 className="mt-3 text-sm font-medium uppercase tracking-wide text-zinc-500">
            Sources
          </h2>
          <ul className="mt-1 space-y-1">
            {answerResult.sources.map((s) => (
              <li key={s.doc_id} className="text-sm">
                <Link
                  to={`/chunks/${encodeURIComponent(s.doc_id)}`}
                  className="text-blue-700 hover:underline"
                >
                  {s.doc_id}
                </Link>
                <span className="ml-2 text-xs text-zinc-500">
                  {s.category} · {s.campus} · score{" "}
                  {typeof s.score === "number" ? s.score.toFixed(3) : "—"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {(search.isError || answer.isError) && (
        <p className="text-sm text-red-600">
          오류: {(search.error || answer.error)?.message}
        </p>
      )}
    </div>
  );
}
