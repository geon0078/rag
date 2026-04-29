import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, Link, useLocation, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import TreeView from "@/components/tree/TreeView";
import RightRail from "@/components/layout/RightRail";
import { useChunkList } from "@/api/chunks";
import { chunkTitle, chunkPath } from "@/lib/chunkLabel";

// 운영웹통합명세서 §6.2 — 사이드바 네비 + 토픽 컨텐츠 + 메타 우측의 3-pane 어드민.

type Health = {
  ok: boolean;
  components: Record<string, { ok: boolean }>;
};

async function fetchHealth(): Promise<Health> {
  const r = await fetch("/api/healthz");
  if (!r.ok) throw new Error(`healthz ${r.status}`);
  return r.json();
}

const NAV_GROUPS: Array<{
  label: string;
  items: Array<{ to: string; icon: string; label: string }>;
}> = [
  {
    label: "운영",
    items: [
      { to: "/", icon: "📊", label: "대시보드" },
      { to: "/preview", icon: "🔍", label: "RAG 미리보기" },
      { to: "/indexing", icon: "⚙️", label: "인덱싱" },
    ],
  },
  {
    label: "콘텐츠",
    items: [
      { to: "/faq/new", icon: "✨", label: "FAQ 신규 작성" },
      { to: "/upload", icon: "⬆", label: "CSV 일괄 업로드" },
      { to: "/history", icon: "🕒", label: "변경 이력" },
    ],
  },
];

function HeaderSearch() {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [debounced, setDebounced] = useState("");
  const navigate = useNavigate();
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(q.trim()), 200);
    return () => clearTimeout(t);
  }, [q]);

  const { data } = useChunkList({ q: debounced || undefined, limit: 8 });

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const items = debounced ? data?.items ?? [] : [];

  return (
    <div ref={boxRef} className="relative w-72">
      <input
        type="search"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        placeholder="청크 검색…"
        className="w-full rounded-md border border-zinc-200 bg-zinc-50 px-3 py-1.5 text-sm placeholder-zinc-400 focus:border-blue-400 focus:bg-white focus:outline-none focus:ring-1 focus:ring-blue-200"
      />
      {open && debounced && (
        <div className="absolute right-0 top-full z-30 mt-1 w-[28rem] overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-lg">
          {items.length === 0 ? (
            <div className="px-3 py-2 text-sm text-zinc-500">결과 없음</div>
          ) : (
            <ul>
              {items.map((c) => (
                <li key={c.doc_id}>
                  <button
                    type="button"
                    onClick={() => {
                      setOpen(false);
                      setQ("");
                      navigate(`/chunks/${encodeURIComponent(c.doc_id)}`);
                    }}
                    className="block w-full px-3 py-2 text-left hover:bg-zinc-50"
                  >
                    <div className="truncate text-sm font-medium text-zinc-900">
                      {chunkTitle(c)}
                    </div>
                    <div className="mt-0.5 flex items-center gap-2 text-[11px] text-zinc-500">
                      <span className="rounded bg-zinc-100 px-1.5 py-0.5">
                        {c.source_collection}
                      </span>
                      <span className="truncate">📁 {chunkPath(c)}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function StatusPill() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["healthz", "pill"],
    queryFn: fetchHealth,
    refetchInterval: 10_000,
  });
  let cls = "bg-zinc-100 text-zinc-500 border-zinc-200";
  let label = "확인 중…";
  let dot = "bg-zinc-400";
  if (error) {
    cls = "bg-red-50 text-red-700 border-red-200";
    label = "API 오프라인";
    dot = "bg-red-500";
  } else if (data && !isLoading) {
    const allOk = data.ok && Object.values(data.components).every((c) => c.ok);
    if (allOk) {
      cls = "bg-emerald-50 text-emerald-700 border-emerald-200";
      label = "All systems normal";
      dot = "bg-emerald-500";
    } else {
      cls = "bg-amber-50 text-amber-700 border-amber-200";
      label = "Degraded";
      dot = "bg-amber-500";
    }
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${cls}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  );
}

function Breadcrumbs() {
  const { pathname } = useLocation();
  const parts = pathname.split("/").filter(Boolean);
  const crumbs: Array<{ label: string; to: string }> = [
    { label: "EulJi RAG", to: "/" },
  ];
  let path = "";
  for (const seg of parts) {
    path += `/${seg}`;
    crumbs.push({ label: decodeURIComponent(seg), to: path });
  }
  return (
    <nav className="flex items-center gap-1 text-xs text-zinc-500">
      {crumbs.map((c, i) => (
        <span key={c.to} className="flex items-center gap-1">
          {i > 0 && <span className="text-zinc-300">/</span>}
          {i === crumbs.length - 1 ? (
            <span className="font-medium text-zinc-700">{c.label}</span>
          ) : (
            <Link to={c.to} className="hover:text-zinc-900">
              {c.label}
            </Link>
          )}
        </span>
      ))}
    </nav>
  );
}

export default function AppLayout() {
  return (
    <div className="flex h-full flex-col bg-zinc-50">
      <header className="flex items-center gap-4 border-b border-zinc-200 bg-white px-5 py-2.5 shadow-sm">
        <Link to="/" className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-md bg-gradient-to-br from-blue-600 to-indigo-600 text-xs font-bold text-white shadow-sm">
            E
          </div>
          <span className="font-semibold text-zinc-900">EulJi RAG Admin</span>
          <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            dev
          </span>
        </Link>
        <div className="ml-auto flex items-center gap-3">
          <HeaderSearch />
          <a
            href="/chat"
            target="_blank"
            rel="noreferrer"
            className="rounded-md border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700 hover:bg-blue-100"
            title="학생용 챗봇 (Onyx 호환 어댑터)"
          >
            🎓 학생 챗봇
          </a>
          <a
            href="http://localhost:3001"
            target="_blank"
            rel="noreferrer"
            className="rounded-md border border-violet-200 bg-violet-50 px-2.5 py-1 text-xs font-medium text-violet-700 hover:bg-violet-100"
            title="Docmost — 운영자 docs"
          >
            📝 Docmost
          </a>
          <StatusPill />
          <span className="text-xs text-zinc-400">v0.1.0</span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <aside className="flex w-64 shrink-0 flex-col border-r border-zinc-200 bg-white">
          <nav className="space-y-4 px-3 py-4">
            {NAV_GROUPS.map((g) => (
              <div key={g.label}>
                <div className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  {g.label}
                </div>
                <ul className="space-y-0.5">
                  {g.items.map((it) => (
                    <li key={it.to}>
                      <NavLink
                        to={it.to}
                        end={it.to === "/"}
                        className={({ isActive }) =>
                          "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors " +
                          (isActive
                            ? "bg-blue-50 font-medium text-blue-700"
                            : "text-zinc-700 hover:bg-zinc-100")
                        }
                      >
                        <span className="text-base leading-none">{it.icon}</span>
                        <span>{it.label}</span>
                      </NavLink>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </nav>

          <div className="mt-2 flex flex-1 flex-col overflow-hidden border-t border-zinc-100">
            <div className="flex items-center justify-between px-3 pb-1 pt-3">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                Collections
              </span>
            </div>
            <div className="flex-1 overflow-y-auto px-2 pb-4">
              <TreeView />
            </div>
          </div>
        </aside>

        <main className="flex flex-1 flex-col overflow-hidden">
          <div className="flex items-center justify-between border-b border-zinc-200 bg-white px-6 py-2">
            <Breadcrumbs />
          </div>
          <div className="flex flex-1 overflow-hidden">
            <div
              id="main-scroll"
              className="flex-1 overflow-y-auto bg-zinc-50 px-6 py-6"
            >
              <div className="mx-auto max-w-5xl">
                <Outlet />
              </div>
            </div>
            <RightRail />
          </div>
        </main>
      </div>
    </div>
  );
}
