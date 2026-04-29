import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useCreateChunk, usePatchChunk } from "@/api/chunks";
import {
  useGenerateKeywords,
  useGenerateNegatives,
  useGenerateVariants,
} from "@/api/assist";

// 운영웹통합명세서 §6.5 / §7.4 / §11 Day 6 — FAQ 신규 작성 3-step 마법사.
//   1. 표준 질문 + 답변 입력 (chunk 생성)
//   2. Solar 가 variants 5개 생성 → 사용자 검수 → patch
//   3. Solar 가 keywords·negatives 추출 → 사용자 검수 → patch + status=Published
// 모두 사람 검수 후 저장.

type Props = {
  onDone?: (docId: string) => void;
};

const CATEGORIES = [
  "학사",
  "학교생활",
  "장학금/근로/비용",
  "수강신청",
  "기숙사",
  "도서관",
  "기타",
];

const CAMPUSES = ["성남", "대전", "공통"];

function genDocId(category: string): string {
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).slice(2, 6);
  const cat = category.replace(/[^A-Za-z0-9가-힣]/g, "");
  return `FAQ:${cat}:${ts}${rand}`;
}

export default function FaqWizard({ onDone }: Props) {
  const navigate = useNavigate();
  const [step, setStep] = useState<1 | 2 | 3>(1);

  const [category, setCategory] = useState(CATEGORIES[0]);
  const [campus, setCampus] = useState("공통");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const [docId, setDocId] = useState<string | null>(null);

  const [variants, setVariants] = useState<string[]>([]);
  const [keywords, setKeywords] = useState<string[]>([]);
  const [negatives, setNegatives] = useState<string[]>([]);

  const create = useCreateChunk();
  const patch = usePatchChunk(docId ?? "");
  const genVariants = useGenerateVariants();
  const genKeywords = useGenerateKeywords();
  const genNegatives = useGenerateNegatives();

  const onStep1Next = async () => {
    if (!question.trim() || !answer.trim()) return;
    const newId = genDocId(category);
    const created = await create.mutateAsync({
      doc_id: newId,
      path: `FAQ/${category}/${question.slice(0, 40)}`,
      source_collection: "FAQ",
      metadata: {
        title: question,
        question,
        answer,
        category,
        campus,
        source_collection: "FAQ",
        status: "Draft",
      },
      contents: answer,
      raw_content: answer,
      status: "Draft",
    });
    setDocId(created.doc_id);
    try {
      const r = await genVariants.mutateAsync(created.doc_id);
      setVariants(r.variants);
    } catch {
      setVariants(["", "", "", "", ""]);
    }
    setStep(2);
  };

  const onStep2Next = async () => {
    if (!docId) return;
    const cleaned = variants.map((v) => v.trim()).filter(Boolean);
    await patch.mutateAsync({ metadata: { question_variants: cleaned } });
    try {
      const [kw, neg] = await Promise.all([
        genKeywords.mutateAsync(docId),
        genNegatives.mutateAsync(docId),
      ]);
      setKeywords(kw.keywords);
      setNegatives(neg.negative_examples);
    } catch {
      setKeywords([]);
      setNegatives([]);
    }
    setStep(3);
  };

  const onFinish = async () => {
    if (!docId) return;
    const k = keywords.map((v) => v.trim()).filter(Boolean);
    const n = negatives.map((v) => v.trim()).filter(Boolean);
    await patch.mutateAsync({
      metadata: { keywords: k, negative_examples: n },
      status: "Published",
    });
    if (onDone) onDone(docId);
    else navigate(`/chunks/${encodeURIComponent(docId)}`);
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold text-zinc-900">FAQ 신규 작성</h1>
      <Stepper step={step} />

      {step === 1 && (
        <section className="space-y-3 rounded border border-zinc-200 bg-white p-4">
          <h2 className="text-sm font-medium text-zinc-700">Step 1 — 표준 질문 + 답변</h2>

          <div className="grid grid-cols-2 gap-2">
            <label className="block text-sm">
              <span className="text-zinc-600">카테고리</span>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="mt-1 w-full rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
              >
                {CATEGORIES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-sm">
              <span className="text-zinc-600">캠퍼스</span>
              <select
                value={campus}
                onChange={(e) => setCampus(e.target.value)}
                className="mt-1 w-full rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
              >
                {CAMPUSES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="block text-sm">
            <span className="text-zinc-600">표준 질문</span>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="졸업학점은 몇 학점이에요?"
              className="mt-1 w-full rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
            />
          </label>

          <label className="block text-sm">
            <span className="text-zinc-600">답변</span>
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              rows={6}
              placeholder="졸업을 위해서는 130학점 이상을 취득해야 합니다…"
              className="mt-1 w-full rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
            />
          </label>

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onStep1Next}
              disabled={!question.trim() || !answer.trim() || create.isPending}
              className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
            >
              다음 — Solar 변형 질문 생성 ▶
            </button>
          </div>
        </section>
      )}

      {step === 2 && (
        <section className="space-y-3 rounded border border-zinc-200 bg-white p-4">
          <h2 className="text-sm font-medium text-zinc-700">Step 2 — 변형 질문 5개 검수</h2>
          {genVariants.isPending && (
            <p className="text-sm text-zinc-500">Solar 호출 중…</p>
          )}
          <ul className="space-y-2">
            {variants.map((v, i) => (
              <li key={i} className="flex gap-2">
                <span className="w-5 text-xs text-zinc-500">{i + 1}.</span>
                <input
                  type="text"
                  value={v}
                  onChange={(e) => {
                    const next = [...variants];
                    next[i] = e.target.value;
                    setVariants(next);
                  }}
                  className="flex-1 rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
                />
              </li>
            ))}
          </ul>
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={async () => {
                if (!docId) return;
                const r = await genVariants.mutateAsync(docId);
                setVariants(r.variants);
              }}
              className="rounded border border-zinc-300 bg-white px-3 py-1 text-xs hover:bg-zinc-50"
            >
              ✨ 다시 생성
            </button>
            <button
              type="button"
              onClick={onStep2Next}
              disabled={patch.isPending}
              className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
            >
              다음 — 키워드/네거티브 추출 ▶
            </button>
          </div>
        </section>
      )}

      {step === 3 && (
        <section className="space-y-3 rounded border border-zinc-200 bg-white p-4">
          <h2 className="text-sm font-medium text-zinc-700">
            Step 3 — 키워드 / 오인 가능 질문 검수
          </h2>

          <div>
            <h3 className="text-xs font-medium text-zinc-500">키워드</h3>
            <ul className="mt-1 space-y-1">
              {keywords.map((v, i) => (
                <li key={i} className="flex gap-2">
                  <span className="w-5 text-xs text-zinc-500">{i + 1}.</span>
                  <input
                    type="text"
                    value={v}
                    onChange={(e) => {
                      const next = [...keywords];
                      next[i] = e.target.value;
                      setKeywords(next);
                    }}
                    className="flex-1 rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
                  />
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="text-xs font-medium text-zinc-500">오인 가능 질문</h3>
            <ul className="mt-1 space-y-1">
              {negatives.map((v, i) => (
                <li key={i} className="flex gap-2">
                  <span className="w-5 text-xs text-zinc-500">{i + 1}.</span>
                  <input
                    type="text"
                    value={v}
                    onChange={(e) => {
                      const next = [...negatives];
                      next[i] = e.target.value;
                      setNegatives(next);
                    }}
                    className="flex-1 rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
                  />
                </li>
              ))}
            </ul>
          </div>

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onFinish}
              disabled={patch.isPending}
              className="rounded bg-emerald-600 px-3 py-2 text-sm text-white hover:bg-emerald-700 disabled:opacity-50"
            >
              ✓ 완료 (Published)
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

function Stepper({ step }: { step: 1 | 2 | 3 }) {
  const labels = ["질문/답변", "변형 질문", "키워드/네거티브"];
  return (
    <ol className="flex gap-2 text-xs">
      {labels.map((label, i) => {
        const idx = (i + 1) as 1 | 2 | 3;
        const active = idx === step;
        const done = idx < step;
        return (
          <li
            key={label}
            className={
              "rounded-full px-3 py-1 " +
              (done
                ? "bg-emerald-100 text-emerald-700"
                : active
                ? "bg-blue-600 text-white"
                : "bg-zinc-100 text-zinc-500")
            }
          >
            {idx}. {label}
          </li>
        );
      })}
    </ol>
  );
}
