import { useState } from "react";
import { useUploadCsv, type UploadCsvResp } from "@/api/upload";

// 운영웹통합명세서 §11 Day 7 — CSV 일괄 업로드 화면.

const COLLECTIONS = [
  "FAQ",
  "학칙_조항",
  "학사일정",
  "강의평가",
  "시설_연락처",
  "장학금",
  "학사정보",
  "학과정보",
  "교육과정",
];

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [collection, setCollection] = useState(COLLECTIONS[0]);
  const [skipValidation, setSkipValidation] = useState(false);
  const [result, setResult] = useState<UploadCsvResp | null>(null);

  const upload = useUploadCsv();

  const onSubmit = async () => {
    if (!file) return;
    const r = await upload.mutateAsync({ file, collection, skipValidation });
    setResult(r);
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold text-zinc-900">CSV 일괄 업로드</h1>
      <p className="text-sm text-zinc-500">
        필수 컬럼: <code>doc_id, path, contents</code>. 그 외 컬럼은 metadata 로 들어갑니다.
      </p>

      <section className="space-y-3 rounded border border-zinc-200 bg-white p-4">
        <div className="grid grid-cols-2 gap-3">
          <label className="block text-sm">
            <span className="text-zinc-600">대상 컬렉션</span>
            <select
              value={collection}
              onChange={(e) => setCollection(e.target.value)}
              className="mt-1 w-full rounded border border-zinc-300 bg-white px-2 py-1 text-sm"
            >
              {COLLECTIONS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-end gap-2 text-sm">
            <input
              type="checkbox"
              checked={skipValidation}
              onChange={(e) => setSkipValidation(e.target.checked)}
            />
            <span className="text-zinc-600">MetadataV3 검증 스킵 (Draft 적재만)</span>
          </label>
        </div>

        <label className="block text-sm">
          <span className="text-zinc-600">CSV 파일</span>
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="mt-1 block w-full text-sm"
          />
        </label>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={onSubmit}
            disabled={!file || upload.isPending}
            className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
          >
            ⬆ 업로드 시작
          </button>
          {file && (
            <span className="self-center text-xs text-zinc-500">
              {file.name} · {(file.size / 1024).toFixed(1)} KB
            </span>
          )}
        </div>

        {upload.isError && (
          <p className="text-sm text-red-600">
            오류: {(upload.error as Error).message}
          </p>
        )}
      </section>

      {result && (
        <section className="rounded border border-zinc-200 bg-white p-4">
          <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">
            결과
          </h2>
          <div className="mt-2 grid grid-cols-3 gap-3 text-center">
            <Stat
              label="신규"
              value={result.created}
              className="bg-emerald-50 text-emerald-700"
            />
            <Stat
              label="갱신"
              value={result.updated}
              className="bg-blue-50 text-blue-700"
            />
            <Stat
              label="오류"
              value={result.error_count}
              className={
                result.error_count > 0
                  ? "bg-red-50 text-red-700"
                  : "bg-zinc-50 text-zinc-600"
              }
            />
          </div>
          {result.errors.length > 0 && (
            <details className="mt-3 text-xs">
              <summary className="cursor-pointer text-red-700">
                오류 목록 ({result.errors.length}개)
              </summary>
              <ul className="mt-2 space-y-1">
                {result.errors.map((e, i) => (
                  <li key={i} className="rounded bg-red-50 p-2">
                    <span className="font-medium">row {e.row}</span>
                    {e.doc_id && <> · {e.doc_id}</>}
                    <pre className="mt-1 whitespace-pre-wrap text-[11px] text-zinc-700">
                      {typeof e.error === "string"
                        ? e.error
                        : JSON.stringify(e.error, null, 2)}
                    </pre>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  className,
}: {
  label: string;
  value: number;
  className: string;
}) {
  return (
    <div className={"rounded p-3 " + className}>
      <div className="text-2xl font-semibold">{value}</div>
      <div className="text-xs">{label}</div>
    </div>
  );
}
