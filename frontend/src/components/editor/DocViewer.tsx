import ReactMarkdown from "react-markdown";
import type { ChunkRow } from "@/api/chunks";

// 운영웹통합명세서 §6.4 — 중앙 콘텐츠 보기 모드.
// Day 3: 마크다운 렌더링 + breadcrumb. Day 4 에 편집 모드(TipTap) 추가 예정.
export default function DocViewer({ chunk }: { chunk: ChunkRow }) {
  const meta = chunk.metadata as Record<string, unknown>;
  const breadcrumb: string[] = Array.isArray(meta?.breadcrumb)
    ? (meta.breadcrumb as string[])
    : [];

  return (
    <article className="prose max-w-none">
      <nav className="mb-3 text-sm text-zinc-500">
        {breadcrumb.length > 0 ? breadcrumb.join("  >  ") : chunk.path}
      </nav>
      <h1 className="text-2xl font-semibold text-zinc-900">
        {String(meta?.title ?? chunk.doc_id)}
      </h1>
      <div className="mt-2 flex gap-2 text-xs text-zinc-500">
        <span>{chunk.source_collection}</span>
        {meta?.campus !== undefined && <span>· 캠퍼스: {String(meta.campus)}</span>}
        <span>· status: {chunk.status}</span>
        <span>· version: {String(meta?.version ?? 1)}</span>
      </div>
      <hr className="my-4 border-zinc-200" />
      <ReactMarkdown>{chunk.contents}</ReactMarkdown>
    </article>
  );
}
