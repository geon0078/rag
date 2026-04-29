import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  useChunk,
  useChunkHistory,
  useChunkRelated,
  usePatchChunk,
} from "@/api/chunks";
import DocViewer from "@/components/editor/DocViewer";
import DocEditor from "@/components/editor/DocEditor";
import MetaPanel from "@/components/meta/MetaPanel";
import { chunkTitle, chunkPath } from "@/lib/chunkLabel";

// 운영웹통합명세서 §10.2 + docs.github.com 스타일 — /chunks/:doc_id.
// "아카이브" = PostgreSQL chunks 테이블 (영구 저장소).
// 사용자에게 "어디에 저장되어 있는지" 가 명확히 보이도록 path/title/version/저장시각 노출.

export default function ChunkPage() {
  const { doc_id } = useParams<{ doc_id: string }>();
  const chunkQ = useChunk(doc_id);
  const relQ = useChunkRelated(doc_id);
  const histQ = useChunkHistory(doc_id);
  const patch = usePatchChunk(doc_id ?? "");
  const [editing, setEditing] = useState(false);

  if (chunkQ.isLoading) return <p className="text-sm text-zinc-500">로딩 중…</p>;
  if (chunkQ.error)
    return (
      <p className="text-sm text-red-600">
        로드 실패: {(chunkQ.error as Error).message}
      </p>
    );
  if (!chunkQ.data) return <p>청크를 찾을 수 없습니다.</p>;

  const chunk = chunkQ.data;
  const onSaveContents = (next: string) => patch.mutate({ contents: next });
  const onChangeMeta = (nextMeta: Record<string, unknown>) =>
    patch.mutate({ metadata: nextMeta });

  const title = chunkTitle(chunk);
  const path = chunkPath(chunk);
  const segments = (chunk.path ?? "").split("/").filter(Boolean);
  const version = (chunk.metadata as Record<string, unknown>)?.version ?? 1;
  const updated = chunk.updated_at?.replace("T", " ").slice(0, 19) ?? "—";
  const lastEditedAt =
    histQ.data?.items?.[0]?.changed_at?.replace("T", " ").slice(0, 19) ?? null;

  return (
    <div className="space-y-6">
      <header className="space-y-3 border-b border-zinc-200 pb-4">
        <nav className="flex flex-wrap items-center gap-1 text-xs text-zinc-500">
          <Link to="/" className="hover:text-zinc-900">
            EulJi RAG
          </Link>
          <span className="text-zinc-300">/</span>
          <Link
            to={`/collections/${encodeURIComponent(chunk.source_collection)}`}
            className="rounded bg-zinc-100 px-1.5 py-0.5 text-[11px] hover:bg-zinc-200"
          >
            {chunk.source_collection}
          </Link>
          {segments.slice(1).map((seg, i) => (
            <span key={i} className="flex items-center gap-1">
              <span className="text-zinc-300">/</span>
              <span className="truncate">{seg}</span>
            </span>
          ))}
        </nav>

        <div className="flex flex-wrap items-start gap-2">
          <h1 className="flex-1 text-2xl font-semibold text-zinc-900">{title}</h1>
          <button
            type="button"
            onClick={() => setEditing(!editing)}
            className="rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-sm hover:bg-zinc-50"
          >
            {editing ? "👁️ 보기" : "✏️ 편집"}
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span
            className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-medium text-emerald-700"
            title="PostgreSQL chunks 테이블에 영구 저장된 상태"
          >
            ✓ 아카이브 저장됨
          </span>
          <span className="rounded bg-zinc-100 px-2 py-0.5 text-zinc-700">
            상태: {chunk.status}
          </span>
          <span className="rounded bg-zinc-100 px-2 py-0.5 text-zinc-700">
            버전 v{String(version)}
          </span>
          <span className="text-zinc-500">최종 저장: {updated}</span>
          {lastEditedAt && (
            <span className="text-zinc-400">· 최근 편집: {lastEditedAt}</span>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
        <div className="space-y-6">
          <section>
            <h2
              id="storage"
              className="scroll-mt-4 text-sm font-medium uppercase tracking-wide text-zinc-500"
            >
              저장 위치
            </h2>
            <div className="mt-2 rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
              <dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
                <div>
                  <dt className="text-[11px] uppercase tracking-wide text-zinc-500">
                    컬렉션
                  </dt>
                  <dd className="mt-0.5">
                    <Link
                      to={`/collections/${encodeURIComponent(chunk.source_collection)}`}
                      className="text-blue-700 hover:underline"
                    >
                      {chunk.source_collection}
                    </Link>
                  </dd>
                </div>
                <div>
                  <dt className="text-[11px] uppercase tracking-wide text-zinc-500">
                    경로 (Breadcrumb)
                  </dt>
                  <dd className="mt-0.5 break-words text-zinc-800">📁 {path}</dd>
                </div>
                <div>
                  <dt className="text-[11px] uppercase tracking-wide text-zinc-500">
                    내부 식별자
                  </dt>
                  <dd className="mt-0.5 font-mono text-[12px] text-zinc-600">
                    {chunk.doc_id}
                  </dd>
                </div>
                <div>
                  <dt className="text-[11px] uppercase tracking-wide text-zinc-500">
                    스키마
                  </dt>
                  <dd className="mt-0.5 text-zinc-700">{chunk.schema_version}</dd>
                </div>
              </dl>
            </div>
          </section>

          <section>
            <h2
              id="content"
              className="scroll-mt-4 text-sm font-medium uppercase tracking-wide text-zinc-500"
            >
              본문
            </h2>
            <div className="mt-2">
              {editing ? (
                <DocEditor initialContent={chunk.contents} onSave={onSaveContents} />
              ) : (
                <DocViewer chunk={chunk} />
              )}
              {patch.isPending && (
                <p className="mt-2 text-xs text-zinc-500">아카이브에 저장 중…</p>
              )}
              {patch.isSuccess && (
                <p className="mt-2 text-xs text-emerald-700">✓ 저장됨</p>
              )}
              {patch.isError && (
                <p className="mt-2 text-xs text-red-600">
                  저장 실패: {(patch.error as Error).message}
                </p>
              )}
            </div>
          </section>

          {relQ.data &&
            (relQ.data.siblings.length > 0 || relQ.data.children.length > 0) && (
              <section>
                <h2
                  id="related"
                  className="scroll-mt-4 text-sm font-medium uppercase tracking-wide text-zinc-500"
                >
                  관련 청크
                </h2>
                <div className="mt-2 grid grid-cols-1 gap-4 md:grid-cols-2">
                  {relQ.data.siblings.length > 0 && (
                    <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
                      <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                        같은 부모 (siblings)
                      </div>
                      <ul className="mt-2 space-y-1 text-sm">
                        {relQ.data.siblings.map((s) => (
                          <li key={s.doc_id}>
                            <Link
                              to={`/chunks/${encodeURIComponent(s.doc_id)}`}
                              className="text-blue-700 hover:underline"
                            >
                              {chunkTitle(s)}
                            </Link>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {relQ.data.children.length > 0 && (
                    <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
                      <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                        자식 청크 (children)
                      </div>
                      <ul className="mt-2 space-y-1 text-sm">
                        {relQ.data.children.map((s) => (
                          <li key={s.doc_id}>
                            <Link
                              to={`/chunks/${encodeURIComponent(s.doc_id)}`}
                              className="text-blue-700 hover:underline"
                            >
                              {chunkTitle(s)}
                            </Link>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </section>
            )}

          <section>
            <h2
              id="cross-links"
              className="scroll-mt-4 text-sm font-medium uppercase tracking-wide text-zinc-500"
            >
              빠른 작업
            </h2>
            <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Link
                to={`/preview?q=${encodeURIComponent(title)}`}
                className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm hover:shadow-md"
              >
                <div className="text-sm font-medium text-zinc-900">🔍 RAG 미리보기</div>
                <div className="mt-1 text-xs text-zinc-500">
                  이 제목으로 답변 시뮬레이션
                </div>
              </Link>
              <Link
                to="/history"
                className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm hover:shadow-md"
              >
                <div className="text-sm font-medium text-zinc-900">🕒 변경 이력</div>
                <div className="mt-1 text-xs text-zinc-500">전체 활동 피드 열기</div>
              </Link>
              <Link
                to="/indexing"
                className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm hover:shadow-md"
              >
                <div className="text-sm font-medium text-zinc-900">
                  ⚙️ 인덱싱 트리거
                </div>
                <div className="mt-1 text-xs text-zinc-500">
                  Qdrant 재인덱싱 시작
                </div>
              </Link>
            </div>
          </section>
        </div>

        <aside className="lg:sticky lg:top-4">
          <MetaPanel chunk={chunk} onChange={onChangeMeta} />
        </aside>
      </div>
    </div>
  );
}
