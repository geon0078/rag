import { useQuery } from "@tanstack/react-query";
import { api } from "./client";

// 운영웹통합명세서 §8.2 — tree read API hook.

export type TreeNode = {
  id: string;
  name: string;
  doc_id: string | null;
  depth: number | null;
  status: string | null;
  children: TreeNode[];
  count?: number;
};

export type TreeResp = {
  collection: string | null;
  tree: TreeNode[];
  count: number;
};

export function useTree(collection?: string) {
  return useQuery({
    queryKey: ["tree", collection ?? "_root"],
    queryFn: async (): Promise<TreeResp> => {
      const r = await api.get("/api/tree", {
        params: collection ? { collection } : undefined,
      });
      return r.data;
    },
  });
}
