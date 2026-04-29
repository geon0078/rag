// 운영웹통합명세서 §8.5 — 인덱싱 작업 API + SSE 진행률.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api } from "./client";

export type IndexingJob = {
  id: number;
  job_type: "incremental" | "full";
  status: "queued" | "running" | "success" | "failed" | "cancelled";
  started_at: string | null;
  completed_at: string | null;
  chunks_total: number | null;
  chunks_processed: number | null;
  error_message: string | null;
};

export type JobListResp = {
  items: IndexingJob[];
};

export function useJobList(limit = 30) {
  return useQuery<JobListResp>({
    queryKey: ["indexing", "jobs", limit],
    queryFn: async () => {
      const r = await api.get(`/api/indexing/jobs?limit=${limit}`);
      return r.data;
    },
    refetchInterval: 5_000,
  });
}

export function useTriggerIncremental() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (): Promise<IndexingJob> => {
      const r = await api.post("/api/indexing/incremental");
      return r.data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["indexing", "jobs"] }),
  });
}

export function useTriggerFull() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (): Promise<IndexingJob> => {
      const r = await api.post("/api/indexing/full");
      return r.data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["indexing", "jobs"] }),
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: number): Promise<IndexingJob> => {
      const r = await api.post(`/api/indexing/jobs/${jobId}/cancel`);
      return r.data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["indexing", "jobs"] }),
  });
}

/**
 * SSE — `/api/indexing/jobs/{id}/stream` 을 EventSource 로 구독.
 * 단말 status (success/failed/cancelled) 도달 시 자동 close.
 */
export function useJobStream(jobId: number | null) {
  const [job, setJob] = useState<IndexingJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (jobId == null) return;
    setJob(null);
    setError(null);
    const es = new EventSource(`/api/indexing/jobs/${jobId}/stream`);
    esRef.current = es;
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as IndexingJob;
        setJob(data);
        if (
          data.status === "success" ||
          data.status === "failed" ||
          data.status === "cancelled"
        ) {
          es.close();
        }
      } catch {
        setError("스트림 파싱 실패");
      }
    };
    es.addEventListener("error", () => {
      setError("스트림 연결 오류");
      es.close();
    });
    return () => {
      es.close();
      esRef.current = null;
    };
  }, [jobId]);

  return { job, error };
}
