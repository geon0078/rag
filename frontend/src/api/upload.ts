// 운영웹통합명세서 §11 Day 7 — CSV 일괄 업로드 API.

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";

export type UploadCsvResp = {
  ok: boolean;
  filename: string;
  collection: string;
  created: number;
  updated: number;
  errors: Array<{ row: number; doc_id?: string; error: unknown }>;
  error_count: number;
};

export function useUploadCsv() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (args: {
      file: File;
      collection: string;
      skipValidation?: boolean;
    }): Promise<UploadCsvResp> => {
      const fd = new FormData();
      fd.append("file", args.file);
      const r = await api.post("/api/upload/csv", fd, {
        headers: { "Content-Type": "multipart/form-data" },
        params: {
          collection: args.collection,
          skip_validation: args.skipValidation ? "true" : "false",
        },
      });
      return r.data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["chunks"] });
      qc.invalidateQueries({ queryKey: ["tree"] });
      qc.invalidateQueries({ queryKey: ["history"] });
    },
  });
}
