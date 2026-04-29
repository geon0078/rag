import { useState } from "react";
import type { ChunkRow } from "@/api/chunks";

// 운영웹통합명세서 §6.5 — 우측 메타 패널. 3-Layer 아코디언.
// Day 4: 표시 + onChange 콜백. Day 5+ 에 컬렉션별 폼 강화 + Domain별 검증.

type Meta = Record<string, unknown>;

export default function MetaPanel({
  chunk,
  onChange,
}: {
  chunk: ChunkRow;
  onChange?: (next: Meta) => void;
}) {
  const meta = (chunk.metadata || {}) as Meta;
  return (
    <div className="space-y-3 text-sm">
      <Section title="Layer 1: Core (필수)">
        <Field label="doc_id" value={chunk.doc_id} readOnly />
        <Field label="parent_doc_id" value={chunk.parent_doc_id || "—"} readOnly />
        <Field label="path" value={chunk.path} readOnly />
        <Field label="schema_version" value={chunk.schema_version} readOnly />
        <Field
          label="title"
          value={String(meta.title ?? "")}
          onChange={(v) => onChange?.({ ...meta, title: v })}
        />
        <SelectField
          label="campus"
          value={String(meta.campus ?? "전체")}
          options={["성남", "의정부", "대전", "전체"]}
          onChange={(v) => onChange?.({ ...meta, campus: v })}
        />
        <Field label="category" value={String(meta.category ?? "")} readOnly />
        <Field
          label="subcategory"
          value={String(meta.subcategory ?? "")}
          onChange={(v) => onChange?.({ ...meta, subcategory: v })}
        />
      </Section>

      <Section title={`Layer 2: Domain (${chunk.source_collection})`}>
        <DomainFields collection={chunk.source_collection} meta={meta} onChange={onChange} />
      </Section>

      <Section title="Layer 3: Operations (자동 관리)">
        <Field label="created_at" value={String(meta.created_at ?? chunk.created_at ?? "—")} readOnly />
        <Field label="indexed_at" value={String(meta.indexed_at ?? "—")} readOnly />
        <Field
          label="last_verified_at"
          value={String(meta.last_verified_at ?? "")}
          onChange={(v) => onChange?.({ ...meta, last_verified_at: v })}
        />
        <Field
          label="effective_start"
          value={String(meta.effective_start ?? "")}
          onChange={(v) => onChange?.({ ...meta, effective_start: v })}
        />
        <Field
          label="effective_end"
          value={String(meta.effective_end ?? "")}
          onChange={(v) => onChange?.({ ...meta, effective_end: v })}
        />
        <Field label="version" value={String(meta.version ?? 1)} readOnly />
        <Field
          label="owner"
          value={String(meta.owner ?? "")}
          onChange={(v) => onChange?.({ ...meta, owner: v })}
        />
        <SelectField
          label="confidence"
          value={String(meta.confidence ?? "medium")}
          options={["high", "medium", "low"]}
          onChange={(v) => onChange?.({ ...meta, confidence: v })}
        />
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className="rounded border border-zinc-200 bg-white"
    >
      <summary className="cursor-pointer select-none border-b border-zinc-200 bg-zinc-50 px-2 py-1 text-xs font-medium uppercase tracking-wide text-zinc-600">
        {title}
      </summary>
      <div className="space-y-1 p-2">{children}</div>
    </details>
  );
}

function Field({
  label,
  value,
  onChange,
  readOnly,
}: {
  label: string;
  value: string;
  onChange?: (v: string) => void;
  readOnly?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-xs text-zinc-500">{label}</span>
      <input
        type="text"
        value={value}
        readOnly={readOnly}
        onChange={(e) => onChange?.(e.target.value)}
        className={
          "mt-0.5 block w-full rounded border px-2 py-1 text-sm " +
          (readOnly
            ? "border-zinc-200 bg-zinc-50 text-zinc-500"
            : "border-zinc-300 bg-white")
        }
      />
    </label>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange?: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-xs text-zinc-500">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        className="mt-0.5 block w-full rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function DomainFields({
  collection,
  meta,
  onChange,
}: {
  collection: string;
  meta: Meta;
  onChange?: (next: Meta) => void;
}) {
  const set = (k: string, v: string) => onChange?.({ ...meta, [k]: v });
  switch (collection) {
    case "학칙_조항":
      return (
        <>
          <Field label="chapter" value={String(meta.chapter ?? "")} onChange={(v) => set("chapter", v)} />
          <Field label="chapter_title" value={String(meta.chapter_title ?? "")} onChange={(v) => set("chapter_title", v)} />
          <Field label="article_number" value={String(meta.article_number ?? "")} onChange={(v) => set("article_number", v)} />
          <Field label="article_title" value={String(meta.article_title ?? "")} onChange={(v) => set("article_title", v)} />
          <Field label="paragraph" value={String(meta.paragraph ?? "")} onChange={(v) => set("paragraph", v)} />
        </>
      );
    case "학사일정":
      return (
        <>
          <Field label="start_date" value={String(meta.start_date ?? "")} onChange={(v) => set("start_date", v)} />
          <Field label="end_date" value={String(meta.end_date ?? "")} onChange={(v) => set("end_date", v)} />
          <Field label="semester" value={String(meta.semester ?? "")} onChange={(v) => set("semester", v)} />
          <Field label="event_type" value={String(meta.event_type ?? "")} onChange={(v) => set("event_type", v)} />
        </>
      );
    case "강의평가":
      return (
        <>
          <Field label="lecture_id" value={String(meta.lecture_id ?? "")} onChange={(v) => set("lecture_id", v)} />
          <Field label="lecture_title" value={String(meta.lecture_title ?? "")} onChange={(v) => set("lecture_title", v)} />
          <Field label="section" value={String(meta.section ?? "")} onChange={(v) => set("section", v)} />
          <Field label="subject_area" value={String(meta.subject_area ?? "")} onChange={(v) => set("subject_area", v)} />
        </>
      );
    case "FAQ":
      return (
        <>
          <Field label="question_canonical" value={String(meta.question_canonical ?? "")} onChange={(v) => set("question_canonical", v)} />
          <Field
            label="keywords (comma)"
            value={Array.isArray(meta.keywords) ? (meta.keywords as string[]).join(",") : String(meta.keywords ?? "")}
            onChange={(v) => onChange?.({ ...meta, keywords: v.split(",").map((s) => s.trim()).filter(Boolean) })}
          />
          <Field
            label="question_variants (comma)"
            value={Array.isArray(meta.question_variants) ? (meta.question_variants as string[]).join(",") : ""}
            onChange={(v) => onChange?.({ ...meta, question_variants: v.split(",").map((s) => s.trim()).filter(Boolean) })}
          />
        </>
      );
    case "시설_연락처":
      return (
        <>
          <Field label="phone" value={String(meta.phone ?? "")} onChange={(v) => set("phone", v)} />
          <Field label="building" value={String(meta.building ?? "")} onChange={(v) => set("building", v)} />
          <Field label="floor" value={String(meta.floor ?? "")} onChange={(v) => set("floor", v)} />
          <Field label="facility_type" value={String(meta.facility_type ?? "")} onChange={(v) => set("facility_type", v)} />
        </>
      );
    case "장학금":
      return (
        <>
          <Field label="scholarship_type" value={String(meta.scholarship_type ?? "")} onChange={(v) => set("scholarship_type", v)} />
          <Field label="application_period_start" value={String(meta.application_period_start ?? "")} onChange={(v) => set("application_period_start", v)} />
          <Field label="application_period_end" value={String(meta.application_period_end ?? "")} onChange={(v) => set("application_period_end", v)} />
          <Field label="eligibility_grade" value={String(meta.eligibility_grade ?? "")} onChange={(v) => set("eligibility_grade", v)} />
        </>
      );
    case "교육과정":
    case "학과정보":
      return (
        <Field label="department" value={String(meta.department ?? "")} onChange={(v) => set("department", v)} />
      );
    default:
      return <p className="text-xs text-zinc-400">이 컬렉션의 도메인 필드 없음.</p>;
  }
}
