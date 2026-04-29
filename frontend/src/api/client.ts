// 운영웹통합명세서 §10.1 — axios 인스턴스 (Vite proxy 가 /api → backend 로 forward).

import axios from "axios";

export const api = axios.create({
  baseURL: "/",
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err?.response?.data) {
      console.error("API error:", err.response.status, err.response.data);
    }
    return Promise.reject(err);
  }
);
