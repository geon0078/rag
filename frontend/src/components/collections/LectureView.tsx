import { Link } from "react-router-dom";
import type { ChunkRow } from "@/api/chunks";

// 운영웹통합명세서 §7.2 + §11 Day 7 — 강의평가 area > lecture 그룹 뷰.

type Props = {
  items: ChunkRow[];
};

type Row = {
  doc_id: string;
  title: string;
  area: string;
  lecture: string;
  section: string;
};

function readRow(c: ChunkRow): Row {
  const m = (c.metadata ?? {}) as Record<string, unknown>;
  return {
    doc_id: c.doc_id,
    title: String(m.title ?? c.doc_id),
    area: String(m.area ?? m.category ?? "기타"),
    lecture: String(m.lecture_title ?? m.lecture_id ?? "(강의 미지정)"),
    section: String(m.section_header ?? ""),
  };
}

export default function LectureView({ items }: Props) {
  const rows = items.map(readRow);

  const byArea = new Map<string, Map<string, Row[]>>();
  for (const r of rows) {
    let lectureMap = byArea.get(r.area);
    if (!lectureMap) {
      lectureMap = new Map();
      byArea.set(r.area, lectureMap);
    }
    const arr = lectureMap.get(r.lecture) ?? [];
    arr.push(r);
    lectureMap.set(r.lecture, arr);
  }

  const areas = Array.from(byArea.keys()).sort();

  return (
    <div className="space-y-4">
      <p className="text-sm text-zinc-500">
        총 {rows.length}개 청크 · {areas.length}개 영역.
      </p>
      {areas.map((area) => {
        const lectureMap = byArea.get(area)!;
        const lectures = Array.from(lectureMap.keys()).sort();
        const totalChunks = Array.from(lectureMap.values()).reduce(
          (acc, arr) => acc + arr.length,
          0
        );
        return (
          <section key={area} className="rounded border border-zinc-200 bg-white">
            <h2 className="border-b border-zinc-100 px-3 py-2 text-sm font-medium text-zinc-700">
              📚 {area}{" "}
              <span className="text-xs text-zinc-400">
                ({lectures.length}개 강의 · {totalChunks}개 청크)
              </span>
            </h2>
            <div className="divide-y divide-zinc-100">
              {lectures.map((lec) => {
                const chunks = lectureMap.get(lec) ?? [];
                return (
                  <details key={lec} className="px-3 py-2">
                    <summary className="cursor-pointer text-sm text-zinc-800">
                      📘 {lec}{" "}
                      <span className="text-xs text-zinc-400">
                        ({chunks.length})
                      </span>
                    </summary>
                    <ul className="mt-1 space-y-1 pl-4">
                      {chunks.map((c) => (
                        <li key={c.doc_id} className="text-sm">
                          <Link
                            to={`/chunks/${encodeURIComponent(c.doc_id)}`}
                            className="text-zinc-700 hover:underline"
                          >
                            📋 {c.section || c.title}
                          </Link>
                        </li>
                      ))}
                    </ul>
                  </details>
                );
              })}
            </div>
          </section>
        );
      })}
    </div>
  );
}
