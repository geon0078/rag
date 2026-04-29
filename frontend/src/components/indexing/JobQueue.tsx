import { useState } from "react";
import {
  useCancelJob,
  useJobList,
  useJobStream,
  type IndexingJob,
} from "@/api/indexing";

// 운영웹통합명세서 §8.5 / §11 Day 6 — 인덱싱 작업 큐 + SSE 진행률.

const STATUS_CLASS: Record<IndexingJob["status"], string> = {
  queued: "bg-zinc-100 text-zinc-700",
  running: "bg-blue-100 text-blue-700",
  success: "bg-emerald-100 text-emerald-700",
  failed: "bg-red-100 text-red-700",
  cancelled: "bg-zinc-200 text-zinc-500",
};

function ProgressBar({ job }: { job: IndexingJob }) {
  const total = job.chunks_total ?? 0;
  const processed = job.chunks_processed ?? 0;
  const pct = total > 0 ? Math.min(100, (processed / total) * 100) : 0;
  return (
    <div className="mt-1">
      <div className="h-1.5 w-full overflow-hidden rounded bg-zinc-200">
        <div
          className="h-full bg-blue-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="mt-0.5 flex justify-between text-[11px] text-zinc-500">
        <span>
          {processed} / {total} chunks
        </span>
        <span>{pct.toFixed(0)}%</span>
      </div>
    </div>
  );
}

function JobRow({
  job,
  selected,
  onSelect,
  onCancel,
}: {
  job: IndexingJob;
  selected: boolean;
  onSelect: () => void;
  onCancel: () => void;
}) {
  const isTerminal =
    job.status === "success" ||
    job.status === "failed" ||
    job.status === "cancelled";
  return (
    <li
      className={
        "rounded border p-2 " +
        (selected ? "border-blue-400 bg-blue-50" : "border-zinc-200 bg-white")
      }
    >
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onSelect}
          className="text-xs text-blue-700 hover:underline"
        >
          #{job.id}
        </button>
        <span
          className={"rounded px-2 py-0.5 text-[11px] " + STATUS_CLASS[job.status]}
        >
          {job.status}
        </span>
        <span className="text-xs text-zinc-500">{job.job_type}</span>
        <span className="ml-auto text-[11px] text-zinc-400">
          {job.started_at?.replace("T", " ").slice(0, 19) ?? "—"}
        </span>
        {!isTerminal && (
          <button
            type="button"
            onClick={onCancel}
            className="ml-2 rounded border border-zinc-300 bg-white px-2 py-0.5 text-[11px] hover:bg-zinc-50"
          >
            취소
          </button>
        )}
      </div>
      {job.status === "running" && <ProgressBar job={job} />}
      {job.error_message && (
        <div className="mt-1 text-xs text-red-600">{job.error_message}</div>
      )}
    </li>
  );
}

export default function JobQueue() {
  const list = useJobList(30);
  const cancel = useCancelJob();
  const [streamId, setStreamId] = useState<number | null>(null);
  const { job: liveJob, error: streamError } = useJobStream(streamId);

  const items = list.data?.items ?? [];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">
          작업 큐
        </h2>
        <button
          type="button"
          onClick={() => list.refetch()}
          className="rounded border border-zinc-300 bg-white px-2 py-0.5 text-xs hover:bg-zinc-50"
        >
          새로고침
        </button>
      </div>

      {list.isLoading && <p className="text-sm text-zinc-500">불러오는 중…</p>}
      {list.isError && (
        <p className="text-sm text-red-600">
          목록 로드 실패: {(list.error as Error).message}
        </p>
      )}

      {items.length === 0 && !list.isLoading && (
        <p className="text-sm text-zinc-500">작업이 없습니다.</p>
      )}

      <ul className="space-y-2">
        {items.map((j) => (
          <JobRow
            key={j.id}
            job={j}
            selected={streamId === j.id}
            onSelect={() => setStreamId(j.id)}
            onCancel={() => cancel.mutate(j.id)}
          />
        ))}
      </ul>

      {streamId !== null && (
        <section className="rounded border border-zinc-200 bg-white p-3">
          <h3 className="text-sm font-medium text-zinc-700">
            실시간 스트림 — Job #{streamId}
          </h3>
          {streamError && (
            <p className="mt-1 text-xs text-red-600">{streamError}</p>
          )}
          {liveJob ? (
            <>
              <div className="mt-1 flex items-center gap-2">
                <span
                  className={
                    "rounded px-2 py-0.5 text-[11px] " +
                    STATUS_CLASS[liveJob.status]
                  }
                >
                  {liveJob.status}
                </span>
                <span className="text-xs text-zinc-500">{liveJob.job_type}</span>
                <span className="ml-auto text-[11px] text-zinc-400">
                  started{" "}
                  {liveJob.started_at?.replace("T", " ").slice(0, 19) ?? "—"}
                </span>
              </div>
              <ProgressBar job={liveJob} />
              {liveJob.error_message && (
                <p className="mt-1 text-xs text-red-600">{liveJob.error_message}</p>
              )}
            </>
          ) : (
            <p className="mt-1 text-xs text-zinc-500">연결 중…</p>
          )}
        </section>
      )}
    </div>
  );
}
