// 운영웹통합명세서 §11 Day 7 — 전역 변경 이력 피드.

import { useQuery } from "@tanstack/react-query";
import { api } from "./client";

export type GlobalHistoryItem = {
  id: number;
  doc_id: string;
  version: number;
  changed_at: string | null;
  diff: Record<string, [unknown, unknown]>;
  source_collection: string;
  path: string;
  title: string | null;
};

export type GlobalHistoryResp = {
  items: GlobalHistoryItem[];
};

export function useGlobalHistory(params: { limit?: number; collection?: string } = {}) {
  return useQuery<GlobalHistoryResp>({
    queryKey: ["history", "recent", params],
    queryFn: async () => {
      const r = await api.get("/api/history/recent", { params });
      return r.data;
    },
    refetchInterval: 10_000,
  });
}
