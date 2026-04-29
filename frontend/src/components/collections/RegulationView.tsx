import { Link } from "react-router-dom";
import type { ChunkRow } from "@/api/chunks";

// 운영웹통합명세서 §7.1 + §11 Day 7 — 학칙_조항 chapter/article 뷰.

type Props = {
  items: ChunkRow[];
};

function articleSortKey(article: string): number {
  const m = article.match(/(\d+)/);
  return m ? parseInt(m[1], 10) : 9999;
}

export default function RegulationView({ items }: Props) {
  type Row = {
    doc_id: string;
    title: string;
    chapter: string;
    article: string;
    section: string;
  };
  const rows: Row[] = items.map((c) => {
    const m = (c.metadata ?? {}) as Record<string, unknown>;
    return {
      doc_id: c.doc_id,
      title: String(m.title ?? c.doc_id),
      chapter: String(m.chapter ?? "(장 미지정)"),
      article: String(m.article_number ?? ""),
      section: String(m.section ?? ""),
    };
  });

  const groups = new Map<string, Row[]>();
  for (const r of rows) {
    const arr = groups.get(r.chapter) ?? [];
    arr.push(r);
    groups.set(r.chapter, arr);
  }
  for (const arr of groups.values()) {
    arr.sort((a, b) => articleSortKey(a.article) - articleSortKey(b.article));
  }
  const chapters = Array.from(groups.keys()).sort(
    (a, b) => articleSortKey(a) - articleSortKey(b)
  );

  return (
    <div className="space-y-4">
      <p className="text-sm text-zinc-500">
        총 {rows.length}개 조항 · 장 별 그룹.
      </p>
      {chapters.map((ch) => (
        <section key={ch} className="rounded border border-zinc-200 bg-white">
          <h2 className="border-b border-zinc-100 px-3 py-2 text-sm font-medium text-zinc-700">
            📖 {ch}{" "}
            <span className="text-xs text-zinc-400">
              ({groups.get(ch)?.length ?? 0})
            </span>
          </h2>
          <ul className="divide-y divide-zinc-100">
            {groups.get(ch)?.map((r) => (
              <li
                key={r.doc_id}
                className="flex items-baseline gap-3 px-3 py-2 text-sm"
              >
                <span className="w-20 shrink-0 font-mono text-xs text-zinc-500">
                  {r.article || "—"}
                </span>
                <Link
                  to={`/chunks/${encodeURIComponent(r.doc_id)}`}
                  className="text-zinc-900 hover:underline"
                >
                  {r.title}
                </Link>
                {r.section && (
                  <span className="ml-auto text-[11px] text-zinc-500">
                    {r.section}
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
