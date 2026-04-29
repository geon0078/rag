import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";

// 운영웹통합명세서 §8.1 — chunk read/write API hooks.

export type ChunkRow = {
  doc_id: string;
  parent_doc_id: string | null;
  path: string;
  schema_version: string;
  source_collection: string;
  metadata: Record<string, unknown>;
  contents: string;
  raw_content: string | null;
  status: string;
  created_at: string | null;
  updated_at: string | null;
};

export type ChunkListResp = {
  items: ChunkRow[];
  limit: number;
  offset: number;
  count: number;
};

export type RelatedResp = {
  self: ChunkRow;
  siblings: ChunkRow[];
  children: ChunkRow[];
};

export function useChunkList(params: {
  collection?: string;
  campus?: string;
  status?: string;
  q?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: ["chunks", params],
    queryFn: async (): Promise<ChunkListResp> => {
      const r = await api.get("/api/chunks", { params });
      return r.data;
    },
    placeholderData: (prev) => prev,
  });
}

export function useChunk(docId: string | undefined) {
  return useQuery({
    queryKey: ["chunk", docId],
    queryFn: async (): Promise<ChunkRow> => {
      const r = await api.get(`/api/chunks/${encodeURIComponent(docId!)}`);
      return r.data;
    },
    enabled: !!docId,
  });
}

export function useChunkRelated(docId: string | undefined) {
  return useQuery({
    queryKey: ["chunk-related", docId],
    queryFn: async (): Promise<RelatedResp> => {
      const r = await api.get(`/api/chunks/${encodeURIComponent(docId!)}/related`);
      return r.data;
    },
    enabled: !!docId,
  });
}

export type ChunkPatch = {
  metadata?: Record<string, unknown>;
  contents?: string;
  raw_content?: string;
  status?: string;
  expected_version?: number;
};

export function usePatchChunk(docId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (payload: ChunkPatch): Promise<ChunkRow> => {
      const r = await api.patch(
        `/api/chunks/${encodeURIComponent(docId)}`,
        payload
      );
      return r.data;
    },
    onSuccess: (data) => {
      qc.setQueryData(["chunk", docId], data);
      qc.invalidateQueries({ queryKey: ["chunk-related", docId] });
      qc.invalidateQueries({ queryKey: ["chunks"] });
    },
  });
}

export type HistoryItem = {
  id: number;
  version: number;
  changed_at: string | null;
  diff: Record<string, [unknown, unknown]>;
};

export function useChunkHistory(docId: string | undefined) {
  return useQuery({
    queryKey: ["chunk-history", docId],
    queryFn: async (): Promise<{ doc_id: string; items: HistoryItem[] }> => {
      const r = await api.get(`/api/chunks/${encodeURIComponent(docId!)}/history`);
      return r.data;
    },
    enabled: !!docId,
  });
}

export async function validateMetadata(metadata: Record<string, unknown>) {
  const r = await api.post("/api/chunks/validate", { metadata });
  return r.data as { ok: boolean };
}

export type ChunkCreate = {
  doc_id: string;
  parent_doc_id?: string | null;
  path: string;
  source_collection: string;
  metadata: Record<string, unknown>;
  contents: string;
  raw_content?: string | null;
  status?: string;
};

export function useCreateChunk() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (payload: ChunkCreate): Promise<ChunkRow> => {
      const r = await api.post("/api/chunks", payload);
      return r.data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["chunks"] });
      qc.invalidateQueries({ queryKey: ["tree"] });
    },
  });
}
