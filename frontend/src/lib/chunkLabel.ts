// 운영자가 doc_id (예: si_static_info_289) 대신 직관적인 위치/제목을 보도록.
// 우선순위: title > metadata.title > path 마지막 세그먼트 > doc_id.

type Source = {
  doc_id?: string | null;
  path?: string | null;
  title?: string | null;
  source_collection?: string | null;
  metadata?: Record<string, unknown> | null;
};

export function chunkTitle(s: Source): string {
  const direct = (s.title ?? "").toString().trim();
  if (direct) return direct;
  const fromMeta =
    s.metadata && typeof s.metadata === "object" && s.metadata.title
      ? String(s.metadata.title).trim()
      : "";
  if (fromMeta) return fromMeta;
  const path = (s.path ?? "").toString().trim();
  if (path) {
    const last = path.split("/").filter(Boolean).pop();
    if (last) return last;
  }
  return s.doc_id ?? "(미지정)";
}

/**
 * 브레드크럼 형태의 위치 표시.
 * 예: "학과정보 / 학점 / 졸업요건 / 졸업요건: 졸업논문/시험"
 */
export function chunkPath(s: Source): string {
  const path = (s.path ?? "").toString().trim();
  if (path) return path.replaceAll("/", " / ");
  if (s.source_collection) return s.source_collection;
  return "";
}
