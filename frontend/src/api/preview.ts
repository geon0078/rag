import { useMutation } from "@tanstack/react-query";
import { api } from "./client";

// 운영웹통합명세서 §8.4 — RAG 미리보기 API.

export type PreviewSearchCandidate = {
  doc_id: string;
  score: number | null;
  title: string | null;
  category: string | null;
  campus: string | null;
  snippet: string;
};

export type PreviewSearchResp = {
  query: string;
  hyde_doc: string | null;
  candidates: PreviewSearchCandidate[];
};

export type PreviewAnswerSource = {
  doc_id: string;
  score: number | null;
  category: string | null;
  campus: string | null;
};

export type PreviewAnswerResp = {
  query: string;
  answer: string;
  grounded: boolean;
  verdict: string;
  retry: boolean;
  sources: PreviewAnswerSource[];
  elapsed_ms: number;
};

export function usePreviewSearch() {
  return useMutation({
    mutationFn: async (
      payload: { query: string; top_k?: number }
    ): Promise<PreviewSearchResp> => {
      const r = await api.post("/api/preview/search", payload);
      return r.data;
    },
  });
}

export function usePreviewAnswer() {
  return useMutation({
    mutationFn: async (
      payload: { query: string; top_k?: number }
    ): Promise<PreviewAnswerResp> => {
      const r = await api.post("/api/preview/answer", payload);
      return r.data;
    },
  });
}
