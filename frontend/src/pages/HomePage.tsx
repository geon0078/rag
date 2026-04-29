import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useTree } from "@/api/tree";
import { useGlobalHistory } from "@/api/history";
import { useJobList } from "@/api/indexing";
import { chunkTitle, chunkPath } from "@/lib/chunkLabel";

// docs.github.com/ko 스타일의 docs landing 페이지.
//   - 상단 hero (소개 + 빠른 액세스 카드)
//   - 작업 영역별 섹션 카드 (콘텐츠 / 검색·검수 / 운영)
//   - 데이터 분포 + 활동 피드 + 인덱싱 작업

type Health = {
  ok: boolean;
  components: Record<string, { ok: boolean; error?: string }>;
};

const COLLECTION_META: Record<string, { icon: string; gradient: string }> = {
  FAQ: { icon: "💬", gradient: "from-rose-500 to-orange-500" },
  강의평가: { icon: "🎓", gradient: "from-violet-500 to-purple-500" },
  학칙_조항: { icon: "📖", gradient: "from-blue-500 to-indigo-500" },
  학사일정: { icon: "📅", gradient: "from-amber-500 to-yellow-500" },
  시설_연락처: { icon: "🏢", gradient: "from-cyan-500 to-blue-500" },
  장학금: { icon: "💰", gradient: "from-emerald-500 to-teal-500" },
  학사정보: { icon: "📚", gradient: "from-sky-500 to-blue-500" },
  학과정보: { icon: "🏛️", gradient: "from-indigo-500 to-violet-500" },
  교육과정: { icon: "🎯", gradient: "from-pink-500 to-rose-500" },
  기타: { icon: "📦", gradient: "from-zinc-500 to-zinc-700" },
};

const SECTIONS: Array<{
  id: string;
  title: string;
  description: string;
  icon: string;
  links: Array<{ to: string; label: string; hint: string }>;
}> = [
  {
    id: "content",
    title: "콘텐츠 관리",
    description: "청크 메타·답변·키워드를 운영자가 직접 편집하고 검수합니다.",
    icon: "📝",
    links: [
      { to: "/faq/new", label: "FAQ 신규 작성", hint: "Solar 보조 3-step 마법사" },
      { to: "/upload", label: "CSV 일괄 업로드", hint: "doc_id, path, contents 필수" },
      { to: "/history", label: "변경 이력", hint: "전역 활동 피드" },
    ],
  },
  {
    id: "search",
    title: "검색·검수",
    description: "RAG 파이프라인을 미리 실행하고 답변/근거를 검증합니다.",
    icon: "🔍",
    links: [
      { to: "/preview", label: "RAG 미리보기", hint: "HyDE + retrieval + 답변" },
      { to: "/collections/FAQ", label: "FAQ 컬렉션", hint: "117 청크" },
      { to: "/collections/학사일정", label: "학사일정 캘린더", hint: "학기별 그룹" },
    ],
  },
  {
    id: "ops",
    title: "운영",
    description: "Qdrant 인덱스 갱신과 작업 큐를 SSE 진행률로 모니터링합니다.",
    icon: "⚙️",
    links: [
      { to: "/indexing", label: "인덱싱 트리거", hint: "증분 / 전체 + SSE" },
      { to: "/collections/학칙_조항", label: "학칙 조항 뷰", hint: "장 별 그룹" },
      { to: "/collections/강의평가", label: "강의평가 뷰", hint: "영역 > 강의" },
    ],
  },
];

async function fetchHealth(): Promise<Health> {
  const r = await fetch("/api/healthz");
  if (!r.ok) throw new Error(`healthz ${r.status}`);
  return r.json();
}

export default function HomePage() {
  const tree = useTree();
  const history = useGlobalHistory({ limit: 8 });
  const jobs = useJobList(10);
  const { data: health, error: healthErr } = useQuery({
    queryKey: ["healthz"],
    queryFn: fetchHealth,
    refetchInterval: 5_000,
  });

  const collections = (tree.data?.tree ?? [])
    .slice()
    .sort((a, b) => (b.count ?? 0) - (a.count ?? 0));
  const totalChunks = collections.reduce((acc, n) => acc + (n.count ?? 0), 0);
  const runningJobs = (jobs.data?.items ?? []).filter(
    (j) => j.status === "running" || j.status === "queued"
  ).length;

  return (
    <div className="space-y-10">
      {/* Hero */}
      <header className="space-y-3 border-b border-zinc-200 pb-6">
        <div className="text-xs font-medium uppercase tracking-wider text-blue-700">
          EulJi RAG · Admin Docs
        </div>
        <h1 className="text-3xl font-semibold text-zinc-900">
          운영 웹 어드민 핸드북
        </h1>
        <p className="max-w-3xl text-base text-zinc-600">
          청크 메타와 답변을 직접 관리하면서 RAG 파이프라인을 한 화면에서 검수합니다.
          아래 작업 영역에서 시작할 페이지를 고르세요.
        </p>
        <div className="flex flex-wrap gap-2 pt-2">
          <Link
            to="/preview"
            className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700"
          >
            🔍 RAG 미리보기 시작
          </Link>
          <Link
            to="/faq/new"
            className="inline-flex items-center gap-1.5 rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-sm font-medium text-zinc-800 hover:bg-zinc-50"
          >
            ✨ FAQ 신규 작성
          </Link>
          <Link
            to="/upload"
            className="inline-flex items-center gap-1.5 rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-sm font-medium text-zinc-800 hover:bg-zinc-50"
          >
            ⬆ CSV 업로드
          </Link>
        </div>
      </header>

      {/* KPI strip */}
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <KpiCard label="총 청크" value={totalChunks.toLocaleString()} hint={`${collections.length}개 컬렉션`} />
        <KpiCard label="진행 중 작업" value={runningJobs} hint="indexing queue" />
        <KpiCard label="최근 변경" value={history.data?.items.length ?? 0} hint="10초 자동 새로고침" />
        <KpiCard label="HyDE A/B 결과" value="grounded +3.3pt" hint="reports/ab_test_hyde.md" />
      </section>

      {/* Section cards */}
      <section className="space-y-3">
        <h2 id="topics" className="scroll-mt-4 text-lg font-semibold text-zinc-900">
          작업 영역
        </h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {SECTIONS.map((s) => (
            <article
              key={s.id}
              className="group flex flex-col rounded-xl border border-zinc-200 bg-white p-5 shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="flex items-center gap-2">
                <span className="text-2xl leading-none">{s.icon}</span>
                <h3 className="text-base font-semibold text-zinc-900">{s.title}</h3>
              </div>
              <p className="mt-2 text-sm text-zinc-600">{s.description}</p>
              <ul className="mt-3 space-y-1.5">
                {s.links.map((l) => (
                  <li key={l.to}>
                    <Link
                      to={l.to}
                      className="group/li flex items-baseline justify-between gap-2 rounded-md border border-transparent px-2 py-1.5 text-sm hover:border-zinc-200 hover:bg-zinc-50"
                    >
                      <span className="font-medium text-blue-700 group-hover/li:underline">
                        {l.label} →
                      </span>
                      <span className="truncate text-xs text-zinc-500">{l.hint}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>

      {/* Collections grid */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 id="collections" className="scroll-mt-4 text-lg font-semibold text-zinc-900">
            컬렉션 살펴보기
          </h2>
          <Link to="/upload" className="text-xs text-blue-700 hover:underline">
            + CSV 추가
          </Link>
        </div>
        {tree.error && (
          <p className="text-sm text-red-600">
            트리 로드 실패: {(tree.error as Error).message}
          </p>
        )}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {collections.map((c) => {
            const meta = COLLECTION_META[c.name] ?? COLLECTION_META["기타"];
            return (
              <Link
                key={c.id}
                to={`/collections/${encodeURIComponent(c.name)}`}
                className="group rounded-xl border border-zinc-200 bg-white p-4 shadow-sm transition-shadow hover:shadow-md"
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-gradient-to-br ${meta.gradient} text-lg text-white shadow-sm`}
                  >
                    {meta.icon}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-zinc-900 group-hover:text-blue-700">
                      {c.name}
                    </div>
                    <div className="mt-0.5 text-xs text-zinc-500">
                      {(c.count ?? 0).toLocaleString()}개 청크 · 보기 →
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Activity + Jobs */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 id="activity" className="scroll-mt-4 text-lg font-semibold text-zinc-900">
              최근 활동
            </h2>
            <Link to="/history" className="text-xs text-blue-700 hover:underline">
              전체 보기 →
            </Link>
          </div>
          <div className="rounded-xl border border-zinc-200 bg-white shadow-sm">
            {(history.data?.items ?? []).slice(0, 8).map((h, i, arr) => (
              <Link
                key={h.id}
                to={`/chunks/${encodeURIComponent(h.doc_id)}`}
                className={
                  "block px-4 py-2.5 hover:bg-zinc-50 " +
                  (i < arr.length - 1 ? "border-b border-zinc-100" : "")
                }
              >
                <div className="flex items-center gap-2 text-sm">
                  <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[11px] text-zinc-600">
                    {h.source_collection}
                  </span>
                  <span className="flex-1 truncate font-medium text-zinc-900">
                    {chunkTitle(h)}
                  </span>
                  <span className="text-[11px] text-zinc-400">
                    v{h.version} ·{" "}
                    {h.changed_at?.replace("T", " ").slice(11, 19) ?? "—"}
                  </span>
                </div>
                <div className="mt-0.5 truncate text-[11px] text-zinc-500">
                  📁 {chunkPath(h)}
                </div>
              </Link>
            ))}
            {history.data && history.data.items.length === 0 && (
              <div className="p-4 text-sm text-zinc-500">변경 이력이 없습니다.</div>
            )}
          </div>
        </section>

        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 id="jobs" className="scroll-mt-4 text-lg font-semibold text-zinc-900">
              인덱싱 작업
            </h2>
            <Link to="/indexing" className="text-xs text-blue-700 hover:underline">
              전체 보기 →
            </Link>
          </div>
          <div className="rounded-xl border border-zinc-200 bg-white shadow-sm">
            {(jobs.data?.items ?? []).slice(0, 8).map((j, i, arr) => {
              const dot =
                j.status === "success"
                  ? "bg-emerald-500"
                  : j.status === "running"
                  ? "bg-blue-500"
                  : j.status === "failed"
                  ? "bg-red-500"
                  : j.status === "cancelled"
                  ? "bg-zinc-400"
                  : "bg-amber-500";
              return (
                <div
                  key={j.id}
                  className={
                    "flex items-center gap-3 px-4 py-2.5 text-sm " +
                    (i < arr.length - 1 ? "border-b border-zinc-100" : "")
                  }
                >
                  <span className={`h-2 w-2 rounded-full ${dot}`} />
                  <span className="text-zinc-700">#{j.id}</span>
                  <span className="text-xs text-zinc-500">{j.job_type}</span>
                  <span className="ml-auto text-[11px] text-zinc-400">
                    {j.chunks_processed ?? 0}/{j.chunks_total ?? "—"}
                  </span>
                  <span className="text-[11px] uppercase text-zinc-500">{j.status}</span>
                </div>
              );
            })}
            {jobs.data && jobs.data.items.length === 0 && (
              <div className="p-4 text-sm text-zinc-500">작업 기록이 없습니다.</div>
            )}
          </div>
        </section>
      </div>

      {/* System status */}
      <section className="space-y-3">
        <h2 id="system" className="scroll-mt-4 text-lg font-semibold text-zinc-900">
          인프라 상태
        </h2>
        {healthErr && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            Backend 연결 실패: {(healthErr as Error).message}
          </div>
        )}
        {health && (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {Object.entries(health.components).map(([name, c]) => (
              <div
                key={name}
                className={
                  "flex items-center justify-between rounded-xl border p-4 shadow-sm " +
                  (c.ok
                    ? "border-emerald-200 bg-emerald-50"
                    : "border-red-200 bg-red-50")
                }
              >
                <div>
                  <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                    {name}
                  </div>
                  <div
                    className={
                      "mt-1 text-sm font-medium " +
                      (c.ok ? "text-emerald-700" : "text-red-700")
                    }
                  >
                    {c.ok ? "Operational" : "Down"}
                  </div>
                  {c.error && (
                    <div className="mt-1 text-xs text-red-600">{c.error}</div>
                  )}
                </div>
                <div
                  className={
                    "h-2 w-2 rounded-full " + (c.ok ? "bg-emerald-500" : "bg-red-500")
                  }
                />
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function KpiCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | number;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-zinc-900">{value}</div>
      {hint && <div className="mt-1 text-xs text-zinc-400">{hint}</div>}
    </div>
  );
}
