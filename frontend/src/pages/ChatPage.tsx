import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ocrFile, ALLOWED_OCR_EXTS } from "../api/ocr";

// onyx+docmost 개발.md §5 — 학생용 챗봇 페이지.
// AppLayout 외부에 mount → 학생에게 admin sidebar 노출 안 함.
// /api/onyx/chat 호출 (Onyx-호환 어댑터, 우리 RagPipeline 위에 변환 레이어).
// 파일 업로드: Solar OCR 로 텍스트 추출 → 입력창에 자동 채워넣어 후속 질문 가능.

type Citation = {
  document_id: string;
  link: string | null;
  source_type: string | null;
  semantic_identifier: string | null;
  blurb: string | null;
  score: number | null;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  grounded?: boolean;
  verdict?: string;
  ts: string;
};

const SUGGESTIONS = [
  "졸업학점은 몇 학점인가요?",
  "수강신청 기간이 언제인가요?",
  "기숙사 입사 신청 방법 알려주세요",
  "학사 경고 기준이 어떻게 되나요?",
  "성남캠퍼스 학식당 운영시간은?",
  "장학금 종류 알려주세요",
];

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ocrPending, setOcrPending] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setOcrPending(true);
    try {
      const result = await ocrFile(file);
      const extracted = (result.content?.text || result.content?.markdown || "").trim();
      if (!extracted) {
        setError("OCR 결과 비어있음 — 다른 파일을 시도해보세요.");
        return;
      }
      // 추출 텍스트를 입력창에 자동 채워서 사용자가 후속 질문 작성 가능
      const summary = extracted.slice(0, 1500);
      const more = extracted.length > 1500 ? `\n... (총 ${extracted.length}자)` : "";
      setInput(
        `📎 ${file.name} 에서 추출된 내용:\n\n${summary}${more}\n\n위 내용에 대해 알려주세요.`,
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "OCR 오류");
    } finally {
      setOcrPending(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  useEffect(() => {
    fetch("/api/onyx/sessions/create", { method: "POST" })
      .then((r) => r.json())
      .then((d) => setSessionId(d.chat_session_id))
      .catch(() => setSessionId(null));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, pending]);

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || pending) return;
    setError(null);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed, ts: new Date().toISOString() },
    ]);
    setInput("");
    setPending(true);

    // assistant placeholder 추가 — SSE 토큰이 들어올수록 content 누적.
    const assistantIdx = (() => {
      let target = -1;
      setMessages((prev) => {
        target = prev.length;
        return [
          ...prev,
          { role: "assistant", content: "", ts: new Date().toISOString() },
        ];
      });
      return target;
    })();

    try {
      const resp = await fetch("/api/onyx/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, chat_session_id: sessionId }),
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      const handleEvent = (event: string, data: string) => {
        try {
          if (event === "message") {
            const j = JSON.parse(data);
            if (j.chat_session_id && !sessionId) setSessionId(j.chat_session_id);
          } else if (event === "citations") {
            const arr = JSON.parse(data) as Citation[];
            setMessages((prev) => {
              const next = [...prev];
              if (next[assistantIdx]) {
                next[assistantIdx] = { ...next[assistantIdx], citations: arr };
              }
              return next;
            });
          } else if (event === "token") {
            const piece = JSON.parse(data) as string;
            setMessages((prev) => {
              const next = [...prev];
              if (next[assistantIdx]) {
                next[assistantIdx] = {
                  ...next[assistantIdx],
                  content: (next[assistantIdx].content ?? "") + piece,
                };
              }
              return next;
            });
          } else if (event === "done") {
            const j = JSON.parse(data);
            setMessages((prev) => {
              const next = [...prev];
              if (next[assistantIdx]) {
                next[assistantIdx] = {
                  ...next[assistantIdx],
                  grounded: j.grounded,
                  verdict: j.verdict,
                };
              }
              return next;
            });
          } else if (event === "error") {
            const j = JSON.parse(data);
            setError(j.error ?? "스트림 오류");
          }
        } catch {
          // ignore malformed event
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE 이벤트는 \n\n 으로 구분.
        let sep: number;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const block = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          let event = "message";
          let data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (data) handleEvent(event, data);
        }
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "알 수 없는 오류");
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex h-full flex-col bg-zinc-50">
      <header className="flex items-center gap-3 border-b border-zinc-200 bg-white px-6 py-3 shadow-sm">
        <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-base font-bold text-white shadow-sm">
          E
        </div>
        <div className="flex-1">
          <div className="font-semibold text-zinc-900">을지대 학사 도우미</div>
          <div className="text-xs text-zinc-500">
            Solar Pro + HyDE · 출처 인용 자동 제공
          </div>
        </div>
        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">
          ✓ 검증된 RAG
        </span>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.length === 0 && !pending && (
            <div className="rounded-2xl border border-dashed border-zinc-300 bg-white p-8 text-center">
              <div className="text-4xl">🎓</div>
              <h1 className="mt-3 text-xl font-semibold text-zinc-900">
                무엇이 궁금하신가요?
              </h1>
              <p className="mt-1 text-sm text-zinc-500">
                을지대학교 학사 정보를 바로 검색해 드립니다.
              </p>
              <div className="mt-5 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => send(s)}
                    className="rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-left text-sm text-zinc-700 transition hover:border-blue-300 hover:bg-white hover:shadow-sm"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <MessageBubble key={i} m={m} />
          ))}

          {pending && (
            <div className="flex gap-3">
              <Avatar role="assistant" />
              <div className="rounded-2xl border border-zinc-200 bg-white px-4 py-3 shadow-sm">
                <span className="inline-flex gap-1">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.3s]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.15s]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400" />
                </span>
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              오류: {error}
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-zinc-200 bg-white px-6 py-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept={ALLOWED_OCR_EXTS.join(",")}
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={pending || ocrPending}
            title="이미지·PDF 업로드 (Solar OCR)"
            className="rounded-xl border border-zinc-300 bg-white px-3 py-3 text-zinc-600 shadow-sm hover:border-blue-300 hover:text-blue-600 disabled:opacity-50"
          >
            {ocrPending ? (
              <span className="inline-flex gap-1">
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:-0.3s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:-0.15s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400" />
              </span>
            ) : (
              <span aria-hidden>📎</span>
            )}
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            rows={1}
            placeholder={ocrPending ? "파일 OCR 처리 중..." : "궁금한 점을 자유롭게 물어보세요"}
            disabled={pending || ocrPending}
            className="flex-1 resize-none rounded-2xl border border-zinc-300 bg-white px-4 py-3 text-sm placeholder-zinc-400 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-200 disabled:opacity-60"
          />
          <button
            type="button"
            onClick={() => send(input)}
            disabled={!input.trim() || pending || ocrPending}
            className="rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 px-4 py-3 text-sm font-medium text-white shadow-sm hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50"
          >
            전송
          </button>
        </div>
        <div className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-zinc-400">
          Enter 전송 · Shift+Enter 줄바꿈 · 📎 클릭으로 사진·PDF 업로드 (Solar OCR)
        </div>
      </div>
    </div>
  );
}

function Avatar({ role }: { role: "user" | "assistant" }) {
  if (role === "user") {
    return (
      <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-zinc-200 text-sm text-zinc-700">
        나
      </div>
    );
  }
  return (
    <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 text-sm font-bold text-white">
      E
    </div>
  );
}

function MessageBubble({ m }: { m: Message }) {
  const isUser = m.role === "user";
  return (
    <div className={"flex gap-3 " + (isUser ? "justify-end" : "")}>
      {!isUser && <Avatar role="assistant" />}
      <div className={"max-w-[80%] " + (isUser ? "order-first" : "")}>
        <div
          className={
            "rounded-2xl px-4 py-3 shadow-sm " +
            (isUser
              ? "bg-blue-600 text-white"
              : "border border-zinc-200 bg-white text-zinc-900")
          }
        >
          {isUser ? (
            <div className="whitespace-pre-wrap text-sm leading-relaxed">
              {m.content}
            </div>
          ) : (
            <div className="prose prose-sm max-w-none text-sm leading-relaxed prose-a:text-blue-600 prose-a:underline prose-a:underline-offset-2 hover:prose-a:text-blue-700">
              <ReactMarkdown
                components={{
                  a: ({ href, children, ...props }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 rounded bg-blue-50 px-1.5 py-0.5 text-blue-700 no-underline hover:bg-blue-100"
                      {...props}
                    >
                      🔗 {children}
                    </a>
                  ),
                }}
              >
                {m.content}
              </ReactMarkdown>
            </div>
          )}
          {!isUser && m.grounded === false && (
            <div className="mt-2 inline-flex items-center gap-1 rounded bg-amber-50 px-2 py-0.5 text-[11px] text-amber-700">
              ⚠ 근거가 부족할 수 있습니다 (verdict: {m.verdict})
            </div>
          )}
        </div>
        {!isUser && m.citations && m.citations.length > 0 && (
          <div className="mt-2 space-y-1">
            <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
              출처 ({m.citations.length})
            </div>
            <ul className="space-y-1">
              {m.citations.map((c, i) => (
                <li key={c.document_id + i}>
                  <a
                    href={c.link ?? "#"}
                    target="_blank"
                    rel="noreferrer"
                    className="block rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs hover:border-blue-300 hover:shadow-sm"
                  >
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] text-zinc-600">
                        {c.source_type ?? "출처"}
                      </span>
                      <span className="truncate font-medium text-zinc-900">
                        {c.semantic_identifier ?? c.document_id}
                      </span>
                      {typeof c.score === "number" && (
                        <span className="ml-auto text-[10px] text-zinc-400">
                          {c.score.toFixed(2)}
                        </span>
                      )}
                    </div>
                    {c.blurb && (
                      <div className="mt-1 line-clamp-2 text-[11px] text-zinc-500">
                        {c.blurb}
                      </div>
                    )}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      {isUser && <Avatar role="user" />}
    </div>
  );
}
