import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

// docs.github.com 의 "On this page" 앵커 네비.
// 메인 컨텐츠 영역(`#main-scroll`)의 h2[id], h3[id] 를 스캔하여 자동 TOC.

type Heading = { id: string; text: string; level: 2 | 3 };

function collectHeadings(): Heading[] {
  const root = document.getElementById("main-scroll");
  if (!root) return [];
  const nodes = Array.from(root.querySelectorAll("h2[id], h3[id]"));
  return nodes.map((el) => ({
    id: el.id,
    text: (el.textContent ?? "").trim(),
    level: el.tagName === "H2" ? 2 : 3,
  }));
}

export default function RightRail() {
  const { pathname } = useLocation();
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    const update = () => setHeadings(collectHeadings());
    update();
    const root = document.getElementById("main-scroll");
    if (!root) return;
    const obs = new MutationObserver(() => update());
    obs.observe(root, { childList: true, subtree: true });
    return () => obs.disconnect();
  }, [pathname]);

  useEffect(() => {
    if (!headings.length) return;
    const root = document.getElementById("main-scroll");
    if (!root) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (visible) setActiveId(visible.target.id);
      },
      { root, rootMargin: "0px 0px -70% 0px", threshold: 0 }
    );
    headings.forEach((h) => {
      const el = document.getElementById(h.id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [headings]);

  if (headings.length < 2) return null;

  return (
    <aside className="hidden w-56 shrink-0 border-l border-zinc-200 bg-white px-4 py-6 xl:block">
      <div className="sticky top-2 space-y-2">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
          On this page
        </div>
        <ul className="space-y-1 border-l border-zinc-200">
          {headings.map((h) => (
            <li key={h.id}>
              <a
                href={`#${h.id}`}
                onClick={(e) => {
                  e.preventDefault();
                  document
                    .getElementById(h.id)
                    ?.scrollIntoView({ behavior: "smooth", block: "start" });
                  history.replaceState(null, "", `#${h.id}`);
                }}
                className={
                  "block border-l-2 py-0.5 pl-3 text-xs transition-colors " +
                  (h.level === 3 ? "pl-6 " : "") +
                  (activeId === h.id
                    ? "-ml-px border-blue-600 font-medium text-blue-700"
                    : "-ml-px border-transparent text-zinc-500 hover:text-zinc-900")
                }
              >
                {h.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
