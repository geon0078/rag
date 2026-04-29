import { Link, useParams } from "react-router-dom";
import { useChunkList } from "@/api/chunks";
import CalendarView from "@/components/collections/CalendarView";
import RegulationView from "@/components/collections/RegulationView";
import LectureView from "@/components/collections/LectureView";

// 운영웹통합명세서 §6.3 / §7 / §11 Day 7 — /collections/:name.
// 컬렉션별 특수 뷰 분기 (학사일정·학칙_조항·강의평가) + 기본 청크 목록.
export default function CollectionPage() {
  const { name } = useParams<{ name: string }>();
  const { data, isLoading, error } = useChunkList({ collection: name, limit: 500 });

  if (isLoading) return <p className="text-sm text-zinc-500">로딩 중…</p>;
  if (error) return <p className="text-sm text-red-600">로드 실패: {(error as Error).message}</p>;
  if (!data) return null;

  const isCalendar = name === "학사일정";
  const isRegulation = name === "학칙_조항" || name === "학칙";
  const isLecture = name === "강의평가";

  return (
    <div className="space-y-3">
      <h1 className="text-2xl font-semibold text-zinc-900">{name}</h1>
      <p className="text-sm text-zinc-500">
        {data.count}개 청크 (limit {data.limit})
      </p>

      {isCalendar && <CalendarView items={data.items} />}
      {isRegulation && <RegulationView items={data.items} />}
      {isLecture && <LectureView items={data.items} />}

      {!isCalendar && !isRegulation && !isLecture && (
        <ul className="space-y-1">
          {data.items.map((c) => {
            const m = c.metadata as Record<string, unknown>;
            return (
              <li
                key={c.doc_id}
                className="rounded border border-zinc-200 bg-white p-2 hover:bg-zinc-50"
              >
                <Link to={`/chunks/${encodeURIComponent(c.doc_id)}`} className="block">
                  <div className="text-sm font-medium text-zinc-900">
                    {String(m?.title ?? c.doc_id)}
                  </div>
                  <div className="mt-0.5 text-xs text-zinc-500">
                    {c.path} · {c.status}
                  </div>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
