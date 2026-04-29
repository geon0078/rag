// Solar OCR API client — V4 backend POST /api/ocr.
// 사용처: ChatPage 의 파일 업로드 버튼.

export type OcrContent = {
  markdown?: string;
  text?: string;
  html?: string;
};

export type OcrResponse = {
  filename: string;
  model: string;
  content: OcrContent;
  elements?: unknown[];
  usage?: Record<string, unknown>;
};

export const ALLOWED_OCR_EXTS = [
  ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp",
  ".pdf", ".docx", ".pptx", ".xlsx", ".hwp", ".hwpx",
];

export const MAX_OCR_BYTES = 50 * 1024 * 1024; // 50 MB

export async function ocrFile(file: File): Promise<OcrResponse> {
  if (file.size > MAX_OCR_BYTES) {
    throw new Error(`파일 크기 50MB 초과: ${(file.size / 1024 / 1024).toFixed(1)}MB`);
  }
  const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
  if (!ALLOWED_OCR_EXTS.includes(ext)) {
    throw new Error(`지원하지 않는 파일 형식: ${ext}`);
  }
  const fd = new FormData();
  fd.append("document", file);
  fd.append("output_formats", "text,markdown");

  const r = await fetch("/api/ocr", {
    method: "POST",
    body: fd,
  });
  if (!r.ok) {
    const errText = await r.text();
    throw new Error(`OCR 실패 ${r.status}: ${errText.slice(0, 200)}`);
  }
  return (await r.json()) as OcrResponse;
}

export async function ocrExtractText(file: File): Promise<string> {
  const fd = new FormData();
  fd.append("document", file);
  const r = await fetch("/api/ocr/extract-text", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`OCR 실패 ${r.status}`);
  const d = (await r.json()) as { filename: string; text: string };
  return d.text;
}
