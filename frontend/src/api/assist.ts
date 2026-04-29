// 운영웹통합명세서 §8.3 + §13 — Solar Pro 자동 보조 API (운영자 검수 전제).

import { useMutation } from "@tanstack/react-query";
import { api } from "./client";

export type VariantsResp = { variants: string[] };
export type KeywordsResp = { keywords: string[] };
export type NegativesResp = { negative_examples: string[] };
export type SummaryResp = { answer_short: string };
export type ParseArticleResp = {
  chapter: string | null;
  article_number: string | null;
  section: string | null;
  raw_match: string | null;
};

export function useGenerateVariants() {
  return useMutation({
    mutationFn: async (docId: string): Promise<VariantsResp> => {
      const r = await api.post(`/api/chunks/${encodeURIComponent(docId)}/generate-variants`);
      return r.data;
    },
  });
}

export function useGenerateKeywords() {
  return useMutation({
    mutationFn: async (docId: string): Promise<KeywordsResp> => {
      const r = await api.post(`/api/chunks/${encodeURIComponent(docId)}/generate-keywords`);
      return r.data;
    },
  });
}

export function useGenerateNegatives() {
  return useMutation({
    mutationFn: async (docId: string): Promise<NegativesResp> => {
      const r = await api.post(`/api/chunks/${encodeURIComponent(docId)}/generate-negatives`);
      return r.data;
    },
  });
}

export function useGenerateSummary() {
  return useMutation({
    mutationFn: async (docId: string): Promise<SummaryResp> => {
      const r = await api.post(`/api/chunks/${encodeURIComponent(docId)}/generate-summary`);
      return r.data;
    },
  });
}

export function useParseArticle() {
  return useMutation({
    mutationFn: async (text: string): Promise<ParseArticleResp> => {
      const r = await api.post("/api/chunks/parse-article", { text });
      return r.data;
    },
  });
}
