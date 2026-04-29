import { useTriggerFull, useTriggerIncremental } from "@/api/indexing";
import JobQueue from "@/components/indexing/JobQueue";

// 운영웹통합명세서 §10.2 / §11 Day 6 — 인덱싱 트리거 + 작업 큐 페이지.
export default function IndexingPage() {
  const incremental = useTriggerIncremental();
  const full = useTriggerFull();

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold text-zinc-900">인덱싱</h1>
      <p className="text-sm text-zinc-500">
        Draft/Published 청크를 Qdrant 에 재인덱싱합니다. 진행률은 SSE 로 실시간 표시.
      </p>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => incremental.mutate()}
          disabled={incremental.isPending}
          className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
        >
          ⚡ 증분 인덱싱
        </button>
        <button
          type="button"
          onClick={() => {
            if (window.confirm("전체 재인덱싱을 시작합니다. 시간이 오래 걸립니다.")) {
              full.mutate();
            }
          }}
          disabled={full.isPending}
          className="rounded border border-zinc-300 bg-white px-3 py-2 text-sm hover:bg-zinc-50 disabled:opacity-50"
        >
          🔄 전체 재인덱싱
        </button>
      </div>

      {(incremental.isError || full.isError) && (
        <p className="text-sm text-red-600">
          오류: {(incremental.error || full.error)?.message}
        </p>
      )}

      <JobQueue />
    </div>
  );
}
