import { Link } from "react-router-dom";
import { useGlobalHistory } from "@/api/history";
import { chunkTitle, chunkPath } from "@/lib/chunkLabel";

// 운영웹통합명세서 §11 Day 7 — 전역 변경 이력 화면.
export default function HistoryPage() {
  const { data, isLoading, error } = useGlobalHistory({ limit: 100 });

  return (
    <div className="space-y-3">
      <h1 className="text-2xl font-semibold text-zinc-900">변경 이력</h1>
      <p className="text-sm text-zinc-500">
        모든 청크의 최근 편집 활동. 10초마다 자동 새로고침.
      </p>

      {isLoading && <p className="text-sm text-zinc-500">불러오는 중…</p>}
      {error && (
        <p className="text-sm text-red-600">로드 실패: {(error as Error).message}</p>
      )}

      <ul className="space-y-2">
        {(data?.items ?? []).map((h) => {
          const diffKeys = Object.keys(h.diff);
          return (
            <li key={h.id} className="rounded-lg border border-zinc-200 bg-white p-3 shadow-sm">
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <span className="font-mono">#{h.id}</span>
                <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[11px]">
                  {h.source_collection}
                </span>
                <span className="ml-auto">
                  {h.changed_at?.replace("T", " ").slice(0, 19) ?? "—"}
                </span>
              </div>
              <Link
                to={`/chunks/${encodeURIComponent(h.doc_id)}`}
                className="mt-1 block text-sm font-medium text-zinc-900 hover:text-blue-700 hover:underline"
              >
                {chunkTitle(h)}
              </Link>
              <div className="mt-0.5 text-[11px] text-zinc-500">
                📁 {chunkPath(h)}
              </div>
              <div className="mt-1 text-xs text-zinc-600">
                v{h.version} · {diffKeys.length}개 필드 변경
                <span className="ml-2 font-mono text-[11px] text-zinc-400">
                  ({h.doc_id})
                </span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {diffKeys.slice(0, 8).map((k) => (
                  <span
                    key={k}
                    className="rounded bg-blue-50 px-1.5 py-0.5 text-[11px] text-blue-700"
                  >
                    {k}
                  </span>
                ))}
                {diffKeys.length > 8 && (
                  <span className="text-[11px] text-zinc-400">
                    +{diffKeys.length - 8}
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ul>

      {data && data.items.length === 0 && (
        <p className="text-sm text-zinc-500">변경 이력이 없습니다.</p>
      )}
    </div>
  );
}
