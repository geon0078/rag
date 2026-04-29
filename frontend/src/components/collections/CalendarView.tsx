import { Link } from "react-router-dom";
import type { ChunkRow } from "@/api/chunks";

// 운영웹통합명세서 §7.3 + §11 Day 7 — 학사일정 캘린더 뷰.
// 외부 라이브러리 없이 학기별 그룹핑 + 시간순 정렬로 가벼운 캘린더 구현.

type Props = {
  items: ChunkRow[];
};

type Event = {
  doc_id: string;
  title: string;
  semester: string;
  start: string | null;
  end: string | null;
  expired: boolean;
};

function readEvent(c: ChunkRow): Event {
  const m = (c.metadata ?? {}) as Record<string, unknown>;
  const start = (m.start_date as string | undefined) ?? null;
  const end = (m.end_date as string | undefined) ?? start;
  const today = new Date().toISOString().slice(0, 10);
  return {
    doc_id: c.doc_id,
    title: String(m.title ?? c.doc_id),
    semester: String(m.semester ?? "기타"),
    start,
    end,
    expired: !!end && end < today,
  };
}

function fmtRange(start: string | null, end: string | null): string {
  if (!start) return "—";
  if (!end || end === start) return start;
  return `${start} ~ ${end}`;
}

export default function CalendarView({ items }: Props) {
  const events = items.map(readEvent);
  const groups = new Map<string, Event[]>();
  for (const e of events) {
    const arr = groups.get(e.semester) ?? [];
    arr.push(e);
    groups.set(e.semester, arr);
  }
  for (const arr of groups.values()) {
    arr.sort((a, b) => (a.start ?? "").localeCompare(b.start ?? ""));
  }
  const semesters = Array.from(groups.keys()).sort();

  return (
    <div className="space-y-4">
      <p className="text-sm text-zinc-500">
        총 {events.length}개 이벤트 · 학기별 그룹.
      </p>
      {semesters.map((sem) => (
        <section key={sem} className="rounded border border-zinc-200 bg-white">
          <h2 className="border-b border-zinc-100 px-3 py-2 text-sm font-medium text-zinc-700">
            📅 {sem}{" "}
            <span className="text-xs text-zinc-400">
              ({groups.get(sem)?.length ?? 0})
            </span>
          </h2>
          <ul className="divide-y divide-zinc-100">
            {groups.get(sem)?.map((e) => (
              <li
                key={e.doc_id}
                className={
                  "flex items-baseline gap-3 px-3 py-2 text-sm " +
                  (e.expired ? "opacity-40" : "")
                }
              >
                <span className="w-44 shrink-0 font-mono text-xs text-zinc-500">
                  {fmtRange(e.start, e.end)}
                </span>
                <Link
                  to={`/chunks/${encodeURIComponent(e.doc_id)}`}
                  className="text-zinc-900 hover:underline"
                >
                  {e.title}
                </Link>
                {e.expired && (
                  <span className="ml-auto rounded bg-zinc-200 px-1.5 py-0.5 text-[11px] text-zinc-600">
                    만료
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
